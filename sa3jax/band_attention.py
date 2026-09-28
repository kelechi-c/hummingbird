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
          k_p, k_c, k_n, v_p, v_c, v_n, kd_p, kd_c, kd_n, out, *, rows, halo, sub, length, window, head_dim,
          half, compute):
    """One `(rows, 128)` output tile: row block `i`, head pair `h`, both differential branches.

    Keys are the contiguous `rows + 2 * halo` positions `i * rows - halo + r`: a halo before, the
    block, a halo after. Queries go in `sub`-row slices, each against the `sub + 2 * halo` keys that
    can reach it, so no query multiplies keys outside its band neighbourhood.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    f32, bf16 = jnp.float32, jnp.bfloat16
    i = pl.program_id(1)
    lane = lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
    head_a = lane < head_dim
    rot_low = lane % head_dim < half  # dims [0, half) pair with d + half; [half, 2 half) with d - half
    span = sub + 2 * halo
    scale = 1.0 / head_dim ** 0.5
    nt = (((1,), (1,)), ((), ()))  # contract the lane dim of both: q @ k^T

    def prep(ref, norm, cos, sin, factor=1.0):
        x = norm[1] * jnp.tanh(norm[0] * ref[0].astype(f32)) + norm[2]  # DyT, fp32
        partner = jnp.where(rot_low, pltpu.roll(x, LANES - half, 1), pltpu.roll(x, half, 1))
        return ((x * cos[...] + partner * sin[...]) * factor).astype(bf16)  # RoPE, fp32, then bf16 for the MXU

    q_norm = (norms[0:1, :], norms[1:2, :], norms[2:3, :])
    k_norm = (norms[3:4, :], norms[4:5, :], norms[5:6, :])
    tables = ((cos_p, sin_p), (cos_c, sin_c), (cos_n, sin_n))
    k_start = i * rows - halo
    k_row = k_start + lax.broadcasted_iota(jnp.int32, (rows + 2 * halo, 1), 0)
    # rows past the end (ragged last block) or before the start are undefined on read: zero v there,
    # because a masked weight of 0 times a NaN would still be NaN
    v = jnp.where((k_row >= 0) & (k_row < length), jnp.concatenate([r[0] for r in (v_p, v_c, v_n)]), 0).astype(bf16)

    def branch(q_ref, k_refs):
        qf = prep(q_ref, q_norm, cos_c, sin_c, scale)  # 1/sqrt(64) = 2^-3 folded into q: exact in bf16
        kf = jnp.concatenate([prep(ref, k_norm, *tables[j]) for j, ref in enumerate(k_refs)])
        slices = []
        for s in range(rows // sub):
            q_pos = i * rows + s * sub + lax.broadcasted_iota(jnp.int32, (sub, 1), 0)
            k_pos = k_start + s * sub + lax.broadcasted_iota(jnp.int32, (1, span), 1)
            band = (jnp.abs(k_pos - q_pos) <= window) & (k_pos >= 0) & (k_pos < length)
            qs, ks, vs = qf[s * sub:(s + 1) * sub], kf[s * sub:s * sub + span], v[s * sub:s * sub + span]
            heads = []
            for lanes in (head_a, ~head_a):
                logits = lax.dot_general(jnp.where(lanes, qs, jnp.zeros_like(qs)), ks, nt, preferred_element_type=f32)
                logits = jnp.where(band, logits, _NEG)  # the max must see only in-band (and finite) scores
                weights = jnp.exp(logits - logits.max(-1, keepdims=True))
                # normalise after P @ V: the division runs on (sub, 128), not on (sub, span)
                total = weights.sum(-1, keepdims=True)
                heads.append(jnp.dot(weights.astype(bf16), vs, preferred_element_type=f32) / total)
            slices.append(jnp.where(head_a, heads[0], heads[1]))
        return jnp.concatenate(slices)

    out[0] = (branch(q, (k_p, k_c, k_n)) - branch(qd, (kd_p, kd_c, kd_n))).astype(compute)


def band_attention(qkv: Array, q_norm: dict[str, Array], k_norm: dict[str, Array], inv_freq: Array,
                   num_heads: int, head_dim: int, window: int, *, rows: int = 1024, halo: int = 32,
                   sub: int = 256, interpret: bool = False) -> Array:
    """`(B, S, 5 * H * D)` projection output -> `(B, S, H * D)` differential band attention.

    `q_norm`/`k_norm` are the block's DyT params (`alpha` scalar, `gamma`/`beta` `(D,)`), shared by
    both differential branches as in `same_l._attention`. `rows` is the row tile; `halo` (>= window,
    a multiple of 8 dividing `rows`) the neighbour rows each side; `sub` the query slice (dividing
    `rows`). `S` need not divide `rows`: the grid is ragged and out-of-range rows are masked/dropped.
    Defaults are the v6e-1 optimum at 120 s (decode only: 256x64 120.1, 256x128 84.8, 256x256 73.8,
    512x256 63.3, 1024x256 60.3, 512x512 66.9 ms): larger slices win until 256 because the kernel is
    bound by per-op vector work, not FLOPs; taller row blocks read each K halo for more queries.
    """
    import jax.experimental.pallas as pl

    if head_dim * 2 != LANES or halo < window or halo % 8 or rows % halo or rows % sub or sub % 8:
        raise ValueError(f"band_attention: need head_dim 64, halo >= window, 8 | halo | rows, 8 | sub | rows "
                         f"(head_dim={head_dim}, rows={rows}, halo={halo}, sub={sub}, window={window})")
    batch, length, _ = qkv.shape
    pairs = num_heads * head_dim // LANES
    blocks = -(-length // rows)
    per = rows // halo
    cos, sin = rope_tables((blocks + 1) * rows, inv_freq, head_dim)  # one spare block: every halo index exists

    def lanes(value: Array) -> Array:
        return jnp.broadcast_to(jnp.tile(jnp.ravel(value).astype(jnp.float32), LANES // jnp.size(value)), (1, LANES))

    norms = jnp.concatenate([lanes(q_norm["alpha"]), lanes(q_norm["gamma"]), lanes(q_norm["beta"]),
                             lanes(k_norm["alpha"]), lanes(k_norm["gamma"]), lanes(k_norm["beta"]),
                             jnp.zeros((2, LANES), jnp.float32)])
    last_halo = -(-length // halo) - 1

    def column(offset: int, size: int, index) -> "pl.BlockSpec":
        return pl.BlockSpec((1, size, LANES), lambda b, i, h: (b, index(i), offset * pairs + h))

    def keys(offset: int) -> list["pl.BlockSpec"]:
        """The halo before, the block, the halo after (halo indices in units of `halo` rows)."""
        return [column(offset, halo, lambda i: jnp.maximum(i * per - 1, 0)), column(offset, rows, lambda i: i),
                column(offset, halo, lambda i: jnp.minimum((i + 1) * per, last_halo))]

    tables = [pl.BlockSpec((halo, LANES), lambda b, i, h: (jnp.maximum(i * per - 1, 0), 0)),
              pl.BlockSpec((rows, LANES), lambda b, i, h: (i, 0)),
              pl.BlockSpec((halo, LANES), lambda b, i, h: ((i + 1) * per, 0))]
    in_specs = [pl.BlockSpec((8, LANES), lambda b, i, h: (0, 0))]
    in_specs += [spec for spec in tables for _ in range(2)]
    in_specs += [column(0, rows, lambda i: i), column(3, rows, lambda i: i)]
    in_specs += [spec for offset in (1, 2, 4) for spec in keys(offset)]
    kernel = pl.pallas_call(
        partial(_body, rows=rows, halo=halo, sub=sub, length=length, window=window, head_dim=head_dim,
                half=inv_freq.shape[0], compute=qkv.dtype),
        out_shape=jax.ShapeDtypeStruct((batch, length, pairs * LANES), qkv.dtype),
        grid=(batch, blocks, pairs),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((1, rows, LANES), lambda b, i, h: (b, i, h)),
        interpret=interpret,
        name="same_l_band_attention",
    )
    return kernel(norms, cos, sin, cos, sin, cos, sin, *[qkv] * 11)
