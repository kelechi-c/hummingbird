"""Fused Pallas TPU qk-prep for DiT self-attention (D3): split -> heads -> RMSNorm -> RoPE -> mask.

The XLA chain between the `to_qkv` projection and SDPA is five separate HBM round trips per block:
`jnp.split` into the differential sections, five `_heads` transposes, four `_rms` (fp32 mean of
squares -> rsqrt -> fp32 scale -> one cast), four `_rope` (fp32 partial rotary -> one cast) and the
`v * v_mask` multiply. This kernel reads the `(B, L, S*H*D)` projection once and writes
q, k, v[, q_diff, k_diff] as `(B, H, L, D)` — the layout the flash kernel consumes — with the head
transpose folded into the output index map, exactly as flicker's `qk_prep` does.

One head per program, so the tile is `(rows, HEAD_DIM)`: that is the width the official TPU flash
kernel's own q/k/v BlockSpecs use, and it lets the output land in `(B, H, L, D)` directly. (A
128-lane two-head tile, as `band_attention` uses for SAME-L, is only writable into a
head-pair-contiguous layout and would need a host transpose — exactly the `_heads` cost this
kernel exists to remove.) `L` is padded up to `ROWS` and the pad rows are sliced off the outputs,
so the pad never reaches the softmax. `lane_freqs` mirrors `band_attention.rope_tables` at 64
lanes; the `pltpu.roll` partner select is the same rotate-half trick.

Numerics follow `dit._rms` / `dit._rope` cast point for cast point: fp32 mean of squares + eps ->
rsqrt -> fp32 multiply by the norm scale -> cast to the compute dtype; fp32 `cos*x + sin*partner`
-> cast, with the first `2*half` dims rotary (half-half `rotate_half`) and the rest a pass-through.
`v` is multiplied by `v_mask` and nothing else.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax

HEAD_DIM = 64
# Row tile: the input is padded up to a multiple of it and sliced back, so it need not divide L.
ROWS = 512


def lane_freqs(length: int, inv_freq: Array) -> tuple[Array, Array]:
    """`(length, HEAD_DIM)` fp32 cos / signed-sin lanes for one head, matching `dit._rope_tables`.

    `band_attention.rope_tables` at 64 lanes: dim `d < 2F` rotates with `inv_freq[d % F]`
    (`F = len(inv_freq)`, i.e. `half`), the sign of `sin` is negative for `d < F` (that lane's
    partner is `d + F`), and dims `>= 2F` pass through (`cos = 1, sin = 0`).
    """
    half = inv_freq.shape[0]
    dim = jnp.arange(HEAD_DIM)
    rotary = dim < 2 * half
    freq = jnp.where(rotary, inv_freq.astype(jnp.float32)[dim % half], 0.0)
    angles = jnp.arange(length, dtype=jnp.float32)[:, None] * freq[None, :]
    return jnp.cos(angles), jnp.sin(angles) * jnp.where(dim < half, -1.0, 1.0) * rotary


def _body(q_in, k_in, v_in, cos, sin, q_scale, k_scale, v_mask, qd_in, kd_in,
          q_out, k_out, v_out, qd_out, kd_out, *, half, eps):
    """One `(rows, HEAD_DIM)` tile of one head: RMSNorm + partial RoPE on the q/k streams, mask v."""
    from jax.experimental.pallas import tpu as pltpu

    c, s = cos[...], sin[...]
    low = lax.broadcasted_iota(jnp.int32, c.shape, 1) < half
    # rotate-half: a low dim pairs with d + half, a high one with d - half (roll semantics)
    roll_low, roll_high = HEAD_DIM - half, half

    def prep(x, scale):
        wide = x.astype(jnp.float32)
        normed = wide * lax.rsqrt(jnp.mean(wide * wide, axis=-1, keepdims=True) + eps)
        normed = (normed * scale.astype(jnp.float32)).astype(x.dtype).astype(jnp.float32)
        partner = jnp.where(low, pltpu.roll(normed, roll_low, 1), pltpu.roll(normed, roll_high, 1))
        return (normed * c + s * partner).astype(x.dtype)

    q_out[0, 0] = prep(q_in[0], q_scale[...])
    k_out[0, 0] = prep(k_in[0], k_scale[...])
    v_out[0, 0] = v_in[0] * v_mask[0, 0]
    if qd_in is not None:
        qd_out[0, 0] = prep(qd_in[0], q_scale[...])
        kd_out[0, 0] = prep(kd_in[0], k_scale[...])


def qk_prep(
    qkv: Array,
    q_scale: Array,
    k_scale: Array,
    cos: Array,
    sin: Array,
    v_mask: Array,
    num_heads: int,
    half: int,
    eps: float,
    *,
    rows: int = ROWS,
    interpret: bool = False,
) -> tuple[Array, Array, Array, Array | None, Array | None]:
    """`(B, L, S*H*D)` `to_qkv` output -> `q, k, v[, q_diff, k_diff]` `(B, H, L, D)`.

    `S` is 3 (`[q | k | v]`) or 5 (`[q | k | v | q_diff | k_diff]`, `differential`), inferred from
    the width. `q_scale`/`k_scale` are the block's RMSNorm vectors `(HEAD_DIM,)`; `cos`/`sin` are
    `lane_freqs` tables covering at least `L` rows; `half` is their rotary pair count; `v_mask` is
    the `(B, 1, L, 1)` compute-dtype mask. The outputs are sliced back to `L` — the pad rows must
    never reach the softmax. `rows` overrides the tile (the test sweeps it); `interpret` runs on CPU.
    """
    import jax.experimental.pallas as pl

    batch, length, width = qkv.shape
    if width % (num_heads * HEAD_DIM) != 0 or width // (num_heads * HEAD_DIM) not in (3, 5):
        raise ValueError(f"qk_prep needs 3 or 5 sections of {num_heads} x {HEAD_DIM}: width {width}")
    differential = width // (num_heads * HEAD_DIM) == 5
    sections = 5 if differential else 3
    padded = -(-length // rows) * rows
    shape = jax.ShapeDtypeStruct((batch, num_heads, padded, HEAD_DIM), qkv.dtype)

    def column(section: int) -> "pl.BlockSpec":
        """Head `h`'s `(rows, HEAD_DIM)` slice of section `section` (`num_heads` blocks wide)."""
        return pl.BlockSpec((1, rows, HEAD_DIM), lambda b, i, h: (b, i, section * num_heads + h))

    def out(b: int, i: int, h: int) -> tuple[int, int, int, int]:
        """`(B, H, L, D)` slot — the DMA does the head transpose."""
        return (b, h, i, 0)

    def body(*refs):
        """Bridge Pallas' (in-refs, out-refs) call to `_body`'s fixed differential arity."""
        if differential:
            _body(*refs, half=half, eps=eps)
        else:
            _body(*refs[:8], None, None, *refs[8:], None, None, half=half, eps=eps)

    table = pl.BlockSpec((rows, HEAD_DIM), lambda b, i, h: (i, 0))
    scale = pl.BlockSpec((1, HEAD_DIM), lambda b, i, h: (0, 0))
    mask = pl.BlockSpec((1, 1, rows, 1), lambda b, i, h: (b, 0, i, 0))
    # in-refs: the sections, the shared tables/scales/mask, then the differential streams
    ins = [column(s) for s in range(3)] + [table, table, scale, scale, mask]
    outs = [pl.BlockSpec((1, 1, rows, HEAD_DIM), out)] * 3
    if differential:
        ins += [column(3), column(4)]
        outs += [pl.BlockSpec((1, 1, rows, HEAD_DIM), out)] * 2
    pad_rows = ((0, 0), (0, padded - length), (0, 0))
    wide = jnp.pad(qkv, pad_rows)
    shared = [jnp.pad(cos, pad_rows[:2]), jnp.pad(sin, pad_rows[:2]), q_scale.reshape(1, HEAD_DIM),
              k_scale.reshape(1, HEAD_DIM), jnp.pad(v_mask, ((0, 0), (0, 0), (0, padded - length), (0, 0)))]
    kernel = pl.pallas_call(body, out_shape=[shape] * sections, grid=(batch, padded // rows, num_heads),
                            in_specs=ins, out_specs=outs, interpret=interpret, name="dit_qk_prep")
    got = kernel(*([wide] * 3 + shared + ([wide, wide] if differential else [])))
    q, k, v = (o[:, :, :length] for o in got[:3])
    if not differential:
        return q, k, v, None, None
    q_diff, k_diff = (o[:, :, :length] for o in got[3:])
    return q, k, v, q_diff, k_diff
