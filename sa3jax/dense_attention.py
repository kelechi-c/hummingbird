"""Fused DiT self-attention on TPU (Pallas): RMSNorm -> RoPE -> differential softmax attention, one pass.

The XLA path spends ~94 of the DiT's ~150 ms per clip in attention plumbing: each of 192 block
evaluations (24 blocks x 8 steps) writes a `(24, 1422, 1422)` score tensor per differential branch to
HBM, reads it back for an fp32 softmax, writes the weights, reads them for `P @ V`, and around that
splits, transposes, normalises and rotates q/k/v in separate round trips. This kernel reads the
`to_qkv` projection once and writes the merged `(B, L, H * 64)` output once; scores never leave VMEM.

Layout, as `band_attention`: `qkv` is `(B, L, 5 * H * 64)` = `[q | k | v | q_diff | k_diff]`, 12 head
*pairs* per section (two 64-dim heads per 128-lane tile). The grid is `(batch, head pair, query block)`
with the query block innermost, so each pair's full-length K, V, K_diff blocks — `(1, L, 128)`, legal
because a block dim equal to the array dim needs no 8-multiple — are DMA'd once and reused. L = 64 + T
(1422 at 120 s) fits VMEM whole, so there is no online softmax: one pass per query block over all keys.
The last query block is ragged; its out-of-range rows are computed on undefined data and dropped.

Semantics, cast point for cast point with `dit._attend` except where noted:
- RMSNorm per head: fp32 mean of squares over that head's 64 lanes (two masked lane sums), fp32 scale,
  cast to the compute dtype — `dit._rms`.
- RoPE: fp32 rotate-half over the first `2 * half` dims (`pltpu.roll` + select, signed-sin table), cast
  back — `dit._rope`. Non-rotary dims have cos 1, sin 0.
- `v * v_mask` in the compute dtype; no key mask (the reference's softmax sees every key).
- Scores: bf16 operands, fp32 accumulation, **kept fp32** through the softmax. XLA rounds them to bf16
  first; torch's flash kernels do not, so this is a deliberate move toward the reference.
- The differential subtraction in fp32, one cast (XLA subtracts in bf16).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from sa3jax.band_attention import LANES, rope_tables

FAST_ROWS = 256
# One whole-sequence query block wins only near 120 s (v6e-1, DiT ms at 1422 tokens: 256-row 105.1, 712-row 100.7,
# one block 99.0); at 1162 tokens 256-row wins (82.7 vs 87.8 at 712), and past ~1536 the fp32 scores outgrow VMEM.
SINGLE_BLOCK = (1400, 1536)
VMEM_LIMIT = 100 * 2**20  # v6e has 128 MiB of VMEM; the default scoped limit is far lower


def _body(norms, cos_q, sin_q, cos_k, sin_k, v_mask, q, qd, k, v, kd, out, k_s, kd_s, v_s, *, head_dim, half,
          eps, compute, stack):
    """One `(rows, 128)` output tile: query block `i` of head pair `h`, both differential branches.

    K, K_diff and masked V do not depend on the query block, so they are prepared once per head pair
    (at `i == 0`) into VMEM scratch and reused by the pair's other query blocks. Recomputing them per
    block was about as much vector work as the scores themselves.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    f32, bf16 = jnp.float32, jnp.bfloat16
    lane = lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
    head_a = lane < head_dim
    rot_low = lane % head_dim < half
    scale = 1.0 / head_dim ** 0.5
    nt = (((1,), (1,)), ((), ()))  # contract the lane dim of both: q @ k^T

    def prep(ref, weight, cos, sin):
        x = ref[0].astype(f32)
        squares = x * x
        mean_a = jnp.sum(jnp.where(head_a, squares, 0.0), -1, keepdims=True) / head_dim
        mean_b = jnp.sum(jnp.where(head_a, 0.0, squares), -1, keepdims=True) / head_dim
        x = (x * jnp.where(head_a, lax.rsqrt(mean_a + eps), lax.rsqrt(mean_b + eps)) * weight).astype(compute)
        x = x.astype(f32)
        partner = jnp.where(rot_low, pltpu.roll(x, LANES - half, 1), pltpu.roll(x, half, 1))
        return (x * cos[...] + partner * sin[...]).astype(compute)

    q_weight, k_weight = norms[0:1, :], norms[1:2, :]

    @pl.when(pl.program_id(2) == 0)
    def _():
        k_s[...] = prep(k, k_weight, cos_k, sin_k).astype(bf16)
        kd_s[...] = prep(kd, k_weight, cos_k, sin_k).astype(bf16)
        v_s[...] = (v[0] * v_mask[0]).astype(bf16)

    def attend(queries, keys):
        """softmax(queries @ keys^T) @ V, normalised after P @ V (on 128 lanes, not on L)."""
        logits = lax.dot_general(queries, keys, nt, preferred_element_type=f32)
        weights = jnp.exp(logits - logits.max(-1, keepdims=True))
        return jnp.dot(weights.astype(bf16), v_s[...], preferred_element_type=f32) / weights.sum(-1, keepdims=True)

    def branch(q_ref, keys):
        # the 1/sqrt(64) = 2^-3 scale folded into q: exact, and one full-score-matrix multiply fewer
        qf = prep(q_ref, q_weight, cos_q, sin_q).astype(f32) * scale
        own, other = jnp.where(head_a, qf, 0.0), jnp.where(head_a, 0.0, qf)
        if stack:  # both heads as one (2 rows, 128) operand: one matmul and one softmax, half the ops
            both = attend(jnp.concatenate([own, other]).astype(bf16), keys)
            return jnp.where(head_a, both[: qf.shape[0]], both[qf.shape[0]:])
        return jnp.where(head_a, attend(own.astype(bf16), keys), attend(other.astype(bf16), keys))

    out[0] = (branch(q, k_s[...]) - branch(qd, kd_s[...])).astype(compute)


def tables(length: int, inv_freq: Array, valid: Array, head_dim: int, compute: jnp.dtype) -> tuple[Array, Array, Array]:
    """Per-clip inputs: lane RoPE cos / signed-sin `(L, 128)` fp32 and the lane V-mask `(B, L, 128)` in the
    compute dtype. Built once in `dit.prepare_static`, reused by all 192 calls."""
    cos, sin = rope_tables(length, inv_freq, head_dim)
    return cos, sin, jnp.broadcast_to(valid.astype(compute)[:, :, None], (*valid.shape, LANES))


def dense_attention(qkv: Array, q_weight: Array, k_weight: Array, cos: Array, sin: Array, mask: Array,
                    num_heads: int, head_dim: int, half: int, eps: float, *, rows: int | None = None,
                    stack: bool = True, interpret: bool = False) -> Array:
    """`(B, L, 5 * H * D)` DiT `to_qkv` output -> `(B, L, H * D)` differential self-attention.

    `q_weight`/`k_weight` are the q/k RMSNorm scales `(D,)`; `cos`/`sin`/`mask` come from `tables`;
    `half` is the rotate-half width (16 for a 32-dim rotary). `rows` is the query block (a multiple of
    8; default: one block near 120 s, else 256 — see `SINGLE_BLOCK`); the last block may be ragged — its
    out-of-range rows read undefined data, are row-independent through the softmax, and are dropped on
    write. `stack` (default on, measured faster) runs both heads of a tile as one operand.
    """
    import jax.experimental.pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = rows or (-(-qkv.shape[1] // 8) * 8 if SINGLE_BLOCK[0] <= qkv.shape[1] <= SINGLE_BLOCK[1] else FAST_ROWS)
    if head_dim * 2 != LANES or rows % 8:
        raise ValueError(f"dense_attention needs head_dim 64 and rows % 8 == 0 (head_dim={head_dim}, rows={rows})")
    batch, length, _ = qkv.shape
    pairs = num_heads * head_dim // LANES
    lanes = lambda w: jnp.tile(w.astype(jnp.float32), LANES // w.shape[0])[None]  # noqa: E731
    norms = jnp.concatenate([lanes(q_weight), lanes(k_weight), jnp.zeros((6, LANES), jnp.float32)])

    def column(offset: int, size: int, block) -> "pl.BlockSpec":
        return pl.BlockSpec((1, size, LANES), lambda b, h, i: (b, block(i), offset * pairs + h))

    whole = lambda i: 0  # noqa: E731 - the full-length K/V/K_diff block
    in_specs = [
        pl.BlockSpec((8, LANES), lambda b, h, i: (0, 0)),
        pl.BlockSpec((rows, LANES), lambda b, h, i: (i, 0)),
        pl.BlockSpec((rows, LANES), lambda b, h, i: (i, 0)),
        pl.BlockSpec((length, LANES), lambda b, h, i: (0, 0)),
        pl.BlockSpec((length, LANES), lambda b, h, i: (0, 0)),
        pl.BlockSpec((1, length, LANES), lambda b, h, i: (b, 0, 0)),
        column(0, rows, lambda i: i), column(3, rows, lambda i: i),
        column(1, length, whole), column(2, length, whole), column(4, length, whole),
    ]
    kernel = pl.pallas_call(
        partial(_body, head_dim=head_dim, half=half, eps=eps, compute=qkv.dtype, stack=stack),
        out_shape=jax.ShapeDtypeStruct((batch, length, pairs * LANES), qkv.dtype),
        grid=(batch, pairs, -(-length // rows)),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((1, rows, LANES), lambda b, h, i: (b, i, h)),
        scratch_shapes=[pltpu.VMEM((length, LANES), jnp.bfloat16)] * 3,
        # the query-block axis must run in order on one core: block 0 fills the K/V scratch the rest read
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"),
                                             vmem_limit_bytes=VMEM_LIMIT),
        interpret=interpret,
        name="dit_dense_attention",
    )
    return kernel(norms, cos, sin, cos, sin, mask, *[qkv] * 5)
