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


def _body(norms, cos_q, sin_q, cos_k, sin_k, v_mask, q, qd, k, v, kd, out, *, head_dim, half, eps, compute):
    """One `(rows, 128)` output tile: query block `i` of head pair `h`, both differential branches."""
    from jax.experimental.pallas import tpu as pltpu

    f32 = jnp.float32
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
    values = (v[0] * v_mask[0]).astype(jnp.bfloat16)

    def branch(q_ref, k_ref):
        # the 1/sqrt(64) = 2^-3 scale folded into q: exact in bf16, and one full-score-matrix
        # multiply fewer — the kernel is bound by vector ops on the (rows, L) scores, not the MXU
        qf = (prep(q_ref, q_weight, cos_q, sin_q).astype(f32) * scale).astype(jnp.bfloat16)
        kf = prep(k_ref, k_weight, cos_k, sin_k).astype(jnp.bfloat16)
        heads = []
        for lanes in (head_a, ~head_a):
            logits = lax.dot_general(jnp.where(lanes, qf, jnp.zeros_like(qf)), kf, nt, preferred_element_type=f32)
            weights = jnp.exp(logits - logits.max(-1, keepdims=True))
            # normalise after P @ V: the division runs on (rows, 128), not on (rows, L)
            total = weights.sum(-1, keepdims=True)
            heads.append(jnp.dot(weights.astype(jnp.bfloat16), values, preferred_element_type=f32) / total)
        return jnp.where(head_a, heads[0], heads[1])

    out[0] = (branch(q, k) - branch(qd, kd)).astype(compute)


def tables(length: int, inv_freq: Array, valid: Array, head_dim: int, compute: jnp.dtype,
           rows: int = FAST_ROWS) -> tuple[Array, Array, Array]:
    """Per-clip inputs: lane RoPE cos / signed-sin `(blocks * rows, 128)` fp32 and the lane V-mask
    `(B, L, 128)` in the compute dtype. Built once in `dit.prepare_static`, reused by all 192 calls."""
    cos, sin = rope_tables(-(-length // rows) * rows, inv_freq, head_dim)
    return cos, sin, jnp.broadcast_to(valid.astype(compute)[:, :, None], (*valid.shape, LANES))


def dense_attention(qkv: Array, q_weight: Array, k_weight: Array, cos: Array, sin: Array, mask: Array,
                    num_heads: int, head_dim: int, half: int, eps: float, *, rows: int = FAST_ROWS,
                    interpret: bool = False) -> Array:
    """`(B, L, 5 * H * D)` DiT `to_qkv` output -> `(B, L, H * D)` differential self-attention.

    `q_weight`/`k_weight` are the q/k RMSNorm scales `(D,)`; `cos`/`sin`/`mask` come from `tables`
    (built with the same `rows`); `half` is the rotate-half width (16 for a 32-dim rotary).
    """
    import jax.experimental.pallas as pl

    if head_dim * 2 != LANES or rows % 8 or cos.shape[0] % rows:
        raise ValueError(f"dense_attention needs head_dim 64, rows % 8 == 0 and tables built with rows "
                         f"(head_dim={head_dim}, rows={rows}, table rows={cos.shape[0]})")
    batch, length, _ = qkv.shape
    pairs = num_heads * head_dim // LANES
    blocks = -(-length // rows)
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
        partial(_body, head_dim=head_dim, half=half, eps=eps, compute=qkv.dtype),
        out_shape=jax.ShapeDtypeStruct((batch, length, pairs * LANES), qkv.dtype),
        grid=(batch, pairs, blocks),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((1, rows, LANES), lambda b, h, i: (b, i, h)),
        interpret=interpret,
        name="dit_dense_attention",
    )
    return kernel(norms, cos, sin, cos[:length], sin[:length], mask, *[qkv] * 5)
