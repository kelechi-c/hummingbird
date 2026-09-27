"""Fused SAME-L attention on TPU (Pallas): DyT -> RoPE -> differential band attention, one pass.

The XLA path (`same_l._attention`) spends ~150 of the decoder's ~227 ms per 120 s clip on
plumbing, not math: 3x-materialised K/V windows (`_windows`), `(…, 17, 51)` logits padded to TPU
tiles, RoPE's slice/negate/concat, and head transposes — each an HBM round trip at 23 086 tokens x
12 blocks. The matmuls themselves take ~38 ms. This kernel reads the `to_qkv` projection once and
writes the merged attention output once; everything in between lives in VMEM.

Layout. `qkv` is `(B, S, 5 * H * 64)` = `[q | k | v | q_diff | k_diff]`, each section 12 lane
tiles of 128 = 12 head *pairs* (head_dim is 64, so one 128-lane tile holds two heads). The grid is
`(batch, row block i, head pair h)` with `h` innermost, so the per-row-block RoPE tables are DMA'd
once per `i`, not once per pair. The output tile `(i, h)` lands in `(B, S, H * 64)` directly — no
merge transpose.

The band. A query at absolute position `p` sees keys `|k - p| <= window` (17). With `rows >=
window`, those keys all sit in row blocks `i - 1`, `i`, `i + 1`, so K, V and K_diff each get three
BlockSpecs (clamped at the edges) and the logits come in three `(rows, rows)` pieces joined by one
softmax. The mask is built from *nominal* positions, so a clamped duplicate block and the padding
rows past `S` are masked out rather than read as real keys. Nothing is gathered or concatenated.

Two heads per tile. Head A's logits are `(q * lanes_A) @ k^T`: zeroing head B's lanes makes the
128-lane dot sum only A's 64 dims, exactly. The output selects A's lanes from A's `p @ v` and B's
from B's. This doubles the (small) attention matmuls but keeps every op on plain 2-D tiles, the
Mosaic rule flicker learned (`ml/flicker/src/flicker/kernels.py`).

Numerics follow `same_l` under `matmul_precision=default`, cast point for cast point: DyT and RoPE
in fp32; q/k rounded to bf16 for `QK^T` with fp32 accumulation (what XLA:TPU does to an fp32 dot at
`default`); fp32 softmax, normalised, then bf16 for `P @ V`; the differential subtraction in fp32
and a single cast to the compute dtype.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

LANES = 128
_NEG = -1e30  # a finite -inf: exp underflows to exactly 0 without inf - inf = nan


def rope_tables(length: int, inv_freq: Array, head_dim: int) -> tuple[Array, Array]:
    """`(length, 128)` fp32 cos and signed-sin lanes for two 64-dim heads per tile.

    `same_l._rope` rotates the first `2 * len(inv_freq)` (32) dims of each head with half-half
    `rotate_half`: dim `d < 16` takes `-x[d + 16]`, `16 <= d < 32` takes `x[d - 16]`. So lane `l`
    (dim `d = l % head_dim`) uses frequency `d % 16` when rotary, and the sign lives in `sin`.
    Non-rotary dims get `cos = 1, sin = 0` and pass through. Angles are `fp32(arange) * inv_freq`,
    exactly `_rope_tables`, so the tables are bit-identical to the XLA path's.
    """
    half = inv_freq.shape[0]
    dim = jnp.arange(LANES) % head_dim
    rotary = dim < 2 * half
    freq = jnp.where(rotary, inv_freq.astype(jnp.float32)[dim % half], 0.0)
    sign = jnp.where(dim < half, -1.0, 1.0) * rotary
    angles = jnp.arange(length, dtype=jnp.float32)[:, None] * freq[None, :]
    return jnp.cos(angles), jnp.sin(angles) * sign


def _body(norms, cos_p, sin_p, cos_c, sin_c, cos_n, sin_n, q, qd,
          k_p, k_c, k_n, v_p, v_c, v_n, kd_p, kd_c, kd_n, out, *, rows, length, window, head_dim, half, compute):
    """One `(rows, 128)` output tile: row block `i`, head pair `h`, both differential branches."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    f32 = jnp.float32
    i = pl.program_id(1)
    lane = lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
    head_a = lane < head_dim
    rot_low = lane % head_dim < half  # dims [0, half) pair with d + half; [half, 2 half) with d - half

    def dyt(x, alpha, gamma, beta):
        return gamma * jnp.tanh(alpha * x[0].astype(f32)) + beta

    def rope(x, cos, sin):
        partner = jnp.where(rot_low, pltpu.roll(x, LANES - half, 1), pltpu.roll(x, half, 1))
        return x * cos[...] + partner * sin[...]

    q_norm = (norms[0:1, :], norms[1:2, :], norms[2:3, :])
    k_norm = (norms[3:4, :], norms[4:5, :], norms[5:6, :])
    tables = ((cos_p, sin_p), (cos_c, sin_c), (cos_n, sin_n))
    q_pos = i * rows + lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    k_pos = [(i + j - 1) * rows + lax.broadcasted_iota(jnp.int32, (1, rows), 1) for j in range(3)]
    band = [(jnp.abs(p - q_pos) <= window) & (p >= 0) & (p < length) for p in k_pos]
    nt = (((1,), (1,)), ((), ()))  # contract the lane dim of both: q @ k^T

    def branch(q_ref, k_refs):
        qf = rope(dyt(q_ref, *q_norm), cos_c, sin_c).astype(jnp.bfloat16)
        ks = [rope(dyt(ref, *k_norm), *tables[j]).astype(jnp.bfloat16) for j, ref in enumerate(k_refs)]
        vs = [ref[0] for ref in (v_p, v_c, v_n)]
        heads = []
        for lanes in (head_a, ~head_a):
            qh = jnp.where(lanes, qf, jnp.zeros_like(qf))
            logits = [jnp.where(band[j], lax.dot_general(qh, ks[j], nt, preferred_element_type=f32)
                                * (1.0 / head_dim ** 0.5), _NEG) for j in range(3)]
            peak = jnp.maximum(jnp.maximum(logits[0].max(-1, keepdims=True), logits[1].max(-1, keepdims=True)),
                               logits[2].max(-1, keepdims=True))
            weights = [jnp.exp(s - peak) for s in logits]
            total = weights[0].sum(-1, keepdims=True) + weights[1].sum(-1, keepdims=True) + weights[2].sum(-1, keepdims=True)
            heads.append(sum(jnp.dot((w / total).astype(jnp.bfloat16), vs[j].astype(jnp.bfloat16),
                                     preferred_element_type=f32) for j, w in enumerate(weights)))
        return jnp.where(head_a, heads[0], heads[1])

    out[0] = (branch(q, (k_p, k_c, k_n)) - branch(qd, (kd_p, kd_c, kd_n))).astype(compute)


def band_attention(qkv: Array, q_norm: dict[str, Array], k_norm: dict[str, Array], inv_freq: Array,
                   num_heads: int, head_dim: int, window: int, *, rows: int = 128,
                   interpret: bool = False) -> Array:
    """`(B, S, 5 * H * D)` projection output -> `(B, S, H * D)` differential band attention.

    `q_norm`/`k_norm` are the block's DyT params (`alpha` scalar, `gamma`/`beta` `(D,)`), shared by
    both differential branches as in `same_l._attention`. `rows` is the row tile (a multiple of 8
    and at least `window`); `S` is zero-padded to a multiple of it and the pad rows are masked.
    """
    import jax.experimental.pallas as pl

    if head_dim * 2 != LANES or rows % 8 or rows < window:
        raise ValueError(f"band_attention needs head_dim 64, rows % 8 == 0 and rows >= window: {head_dim}, {rows}")
    batch, length, width = qkv.shape
    pairs = num_heads * head_dim // LANES
    blocks = -(-length // rows)
    padded = jnp.pad(qkv, ((0, 0), (0, blocks * rows - length), (0, 0)))
    cos, sin = rope_tables(blocks * rows, inv_freq, head_dim)

    def lanes(value: Array) -> Array:
        return jnp.broadcast_to(jnp.tile(jnp.ravel(value).astype(jnp.float32), LANES // jnp.size(value)), (1, LANES))

    norms = jnp.concatenate([lanes(q_norm["alpha"]), lanes(q_norm["gamma"]), lanes(q_norm["beta"]),
                             lanes(k_norm["alpha"]), lanes(k_norm["gamma"]), lanes(k_norm["beta"]),
                             jnp.zeros((2, LANES), jnp.float32)])
    last = blocks - 1
    shifts = (lambda i: jnp.maximum(i - 1, 0), lambda i: i, lambda i: jnp.minimum(i + 1, last))

    def section(offset: int, shift=shifts[1]) -> "pl.BlockSpec":
        return pl.BlockSpec((1, rows, LANES), lambda b, i, h: (b, shift(i), offset * pairs + h))

    def table(shift) -> "pl.BlockSpec":
        return pl.BlockSpec((rows, LANES), lambda b, i, h: (shift(i), 0))

    in_specs = [pl.BlockSpec((8, LANES), lambda b, i, h: (0, 0))]
    in_specs += [spec for shift in shifts for spec in (table(shift), table(shift))]
    in_specs += [section(0), section(3)]
    in_specs += [section(offset, shift) for offset in (1, 2, 4) for shift in shifts]
    kernel = pl.pallas_call(
        partial(_body, rows=rows, length=length, window=window, head_dim=head_dim,
                half=inv_freq.shape[0], compute=qkv.dtype),
        out_shape=jax.ShapeDtypeStruct((batch, blocks * rows, pairs * LANES), qkv.dtype),
        grid=(batch, blocks, pairs),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((1, rows, LANES), lambda b, i, h: (b, i, h)),
        interpret=interpret,
        name="same_l_band_attention",
    )
    tables = [t for _ in shifts for t in (cos, sin)]
    sections = [padded] * 11
    return kernel(norms, *tables, *sections)[:, :length]
