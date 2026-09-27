"""Stable Audio 3 diffusion transformer (`small-music`), pure functional JAX.

Ground truth: `models/dit.py:179-309` (`DiffusionTransformer._forward`) and `models/transformer.py`
(`ContinuousTransformer.forward:1161-1272`, `TransformerBlock.forward:997-1068`,
`Attention:598-800`, `RotaryEmbedding:240-322`, `RMSNorm:392-410`, `GLU:421-444`).

Two entry points, split so that per-step work is hoisted rather than repeated: `prepare_static`
folds everything that does not depend on `x`/`t`, and `dit_forward` is the denoiser itself, shaped
for `jax.lax.scan` — fixed shapes, no branching on traced values, no side effects.

Out of scope, deliberately (dead or off at `cfg_scale=1.0` for the shipped checkpoint): CFG + APG +
norm-threshold + rescale, LoRA, `prepend_cond`, `input_concat_cond`, `modular_local_cond`, causal
mode, `timestep_cond_type="input_concat"`, and the inpaint path. `differential` attention is kept as
a flag: `medium` sets it and retrofitting it would be an attention rewrite, not a config change.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import Array

from sa3jax import qk_prep
from sa3jax.config import DiTConfig
from sa3jax.dtypes import Dtype

Params = dict[str, Any]
Static = dict[str, Array]
FlashKernel = Callable[[Array, Array, Array, Array], Array]  # (q, k, v, segment_ids) -> out

# The Pallas TPU flash kernel's legal sequence tiles (multiples of its 128 lanes): self-attention
# pads 64 + T up to the largest that fits, so the kernel gets as few key blocks as it can.
_FLASH_TILES: Final[tuple[int, ...]] = (512, 256, 128)


def _linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    """`nn.Linear`; the checkpoint stores `[out, in]`."""
    y = x @ jnp.swapaxes(weight, -1, -2)
    return y if bias is None else y + bias


def _sequential(seq: Params, x: Array) -> Array:
    """`nn.Sequential(Linear, SiLU, Linear)`; the two `to_*_embed` projections carry no bias."""
    inner, outer = seq["in"], seq["out"]
    return _linear(jax.nn.silu(_linear(x, inner["weight"], inner.get("bias"))), outer["weight"], outer.get("bias"))


def _rms(x: Array, weight: Array, eps: float, dtype: Dtype) -> Array:
    """`RMSNorm`: the stored vector is a *scale*, not Gemma's `(1 + w)` (`transformer.py:392-410`).

    The mean of squares runs in `dtype.norm` (fp32 in every preset) and the result is cast back to
    the input dtype. The block norm asks for fp32 via `force_fp32`; the q/k norm does not, but
    torch's CUDA RMSNorm kernel accumulates in fp32 `opmath_type` for half inputs either way — and
    in fp16 the difference is not subtle: the cross-attention keys reach 1075, so `k * k` overflows
    fp16 and 553 of 4112 key rows come back as zeros (measured 9.1 dB against the fp32 opmath).
    """
    in_dtype = x.dtype
    x = x.astype(dtype.norm)
    normed = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return (normed * weight.astype(dtype.norm)).astype(in_dtype)


def _conv1x1(x: Array, kernel: Array) -> Array:
    """1x1 `Conv1d` on a channel-first tensor — a channel matmul, so it is written as one.

    `jax.lax.conv_general_dilated` is *not* row-symmetric across the batch on CPU: identical rows
    in one batch differ by 2^-16, while the same arithmetic as a matmul is bit-equal. That is
    invisible at B=1 but would make an 8-chip sharded batch differ from eight single-clip runs, so
    the matmul form is used deliberately. The kernel is stored `[out, in, k]` with `k == 1`.
    """
    return jnp.einsum("bct,oc->bot", x, kernel[:, :, 0])


def _expo_fourier(t: Array, dim: int) -> Array:
    """`ExpoFourierFeatures(256, 0.5, 10000.0)` (`blocks.py:50-82`): fp32 throughout, result cast
    back to the input dtype. `t` arrives already rounded to the model dtype (`dit.py:238`)."""
    in_dtype = t.dtype
    args = t.astype(jnp.float32)[:, None] * jnp.exp(
        jnp.linspace(0.0, 1.0, dim // 2, dtype=jnp.float32) * (math.log(10000.0) - math.log(0.5)) + math.log(0.5)
    )
    args = args * (2.0 * math.pi)
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(in_dtype)


def _rope_tables(seq_len: int, inv_freq: Array) -> tuple[Array, Array]:
    """`RotaryEmbedding(max(head_dim // 2, 32))` over the full `64 + T` sequence. `cat(freqs,
    freqs)` is the partial-rotary duplication: `rot_dim = 32` of the 64 head dims."""
    angles = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv_freq.astype(jnp.float32)[None, :]
    freqs = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(freqs), jnp.sin(freqs)


def _rope(x: Array, cos: Array, sin: Array, dtype: Dtype) -> Array:
    """Partial rotary with half-half `rotate_half` — Llama-style, **not** interleaved — computed in
    fp32 and cast back (`transformer.py:296-322`). `cos`/`sin` are `(S, rot_dim)`."""
    out_dtype = x.dtype
    rot_dim = cos.shape[-1]
    x = x.astype(dtype.rope)
    rot, rest = x[..., :rot_dim], x[..., rot_dim:]
    rotated = jnp.concatenate([-rot[..., rot_dim // 2 :], rot[..., : rot_dim // 2]], axis=-1)
    return jnp.concatenate([rot * cos + rotated * sin, rest], axis=-1).astype(out_dtype)


def _heads(x: Array, cfg: DiTConfig) -> Array:
    """`b n (h d) -> b h n d`."""
    batch, seq, _ = x.shape
    return x.reshape(batch, seq, cfg.num_heads, cfg.head_dim).transpose(0, 2, 1, 3)


def _merge(x: Array) -> Array:
    """`b h n d -> b n (h d)`."""
    batch, heads, seq, dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq, heads * dim)


def _flash_padded(length: int) -> int:
    """`S` padded up to the largest `_FLASH_TILES` tile that fits; the kernel needs a divisor."""
    tile = next(size for size in _FLASH_TILES if size <= max(length, _FLASH_TILES[-1]))
    return -(-length // tile) * tile


def _pallas_flash(q: Array, k: Array, v: Array, seg: Array) -> Array:
    """The Pallas TPU flash kernel (`jax.experimental.pallas.ops.tpu.flash_attention`).

    `seg` carries the kernel-padding boundary: real positions `1`, padded `0`. Block tiles must
    divide the padded length and be multiples of 128; `block_k == kv_seq_len` is legal (the
    single-step kernel), and head_dim 64 is accepted — the kernel's `l_broadcast` handles
    `head_dim < 128` by slicing. Imported lazily so a CPU import stays free of the TPU module.
    """
    from jax.experimental.pallas.ops.tpu import flash_attention as flash

    tile = next(size for size in _FLASH_TILES if q.shape[2] % size == 0)
    return flash.flash_attention(
        q,
        k,
        v,
        segment_ids=flash.SegmentIds(q=seg, kv=seg),
        causal=False,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        block_sizes=flash.BlockSizes(block_q=tile, block_k_major=tile, block_k=tile, block_b=1),
    )


def _flash_sdpa(q: Array, k: Array, v: Array, kernel: FlashKernel | None = None) -> Array:
    """Self-attention via the Pallas TPU flash kernel, padded to a legal tile, sliced back.

    The kernel needs `S % 128 == 0` and cannot express the caller's `v_mask` by segment: a
    segment-masked key leaves the softmax denominator, while `v_mask` zeroes V and *keeps* the
    key (`_sdpa`). So only the kernel's own padding past the real length is segment-masked —
    positions `[0, S)` get id 1, the extra `padded - S` get id 0, and the padded queries' outputs
    (which attend padded keys only) are sliced away. `v_mask` is applied by `_attend` exactly as
    in the XLA path. `kernel` is the dependency-injected `(q, k, v, seg) -> out` callable, so the
    padding/segment/slice logic is testable on CPU with a pure-jnp attention in its place.
    """
    length = q.shape[2]
    padded = _flash_padded(length)
    seg = jnp.broadcast_to(
        jnp.concatenate([jnp.ones(length, jnp.int32), jnp.zeros(padded - length, jnp.int32)]),
        (q.shape[0], padded),
    )
    pad = ((0, 0), (0, 0), (0, padded - length), (0, 0))
    kernel = _pallas_flash if kernel is None else kernel
    return kernel(jnp.pad(q, pad), jnp.pad(k, pad), jnp.pad(v, pad), seg)[:, :, :length]


def _sdpa(q: Array, k: Array, v: Array, cfg: DiTConfig, *, flash: bool = False) -> Array:
    """Unmasked softmax attention at `1/sqrt(head_dim)`. Padding is V-zeroing, not an additive
    mask (`transformer.py:675-699`), so the softmax sees every key; it runs in fp32, as the
    reference kernels do.

    `flash=True` (`cfg.attention == "flash"`, self-attention only) replaces the two einsums with
    the Pallas TPU kernel. The XLA path rounds the logits to the compute dtype before the fp32
    softmax; the kernel keeps them fp32 throughout — a deliberate, documented difference.
    """
    if flash:
        return _flash_sdpa(q, k, v)
    logits = jnp.einsum("bhsd,bhtd->bhst", q, k) * (1.0 / math.sqrt(cfg.head_dim))
    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(q.dtype)
    return jnp.einsum("bhst,bhtd->bhsd", weights, v)


def _k_norm(attn: Params, k: Array, k_diff: Array | None, cfg: DiTConfig, dtype: Dtype) -> tuple[Array, Array | None]:
    """`k`'s RMSNorm (and its differential pair), shared by self- and cross-attention. `_attend`
    only norms `q`, because cross-attention's k is hoisted into `static` already normalised."""
    k = _rms(k, attn["k_norm"]["weight"], cfg.qk_norm_eps, dtype)
    if k_diff is not None:
        k_diff = _rms(k_diff, attn["k_norm"]["weight"], cfg.qk_norm_eps, dtype)
    return k, k_diff


def _attend_out(
    attn: Params,
    q: Array,
    k: Array,
    v: Array,
    q_diff: Array | None,
    k_diff: Array | None,
    cfg: DiTConfig,
    *,
    flash: bool = False,
) -> Array:
    """SDPA -> `to_out` (with the differential subtraction) for already normed/roped q/k/v.

    `out = SDPA(q, k, v) - SDPA(q_diff, k_diff, v)` with a shared `v` when `differential`.
    """
    out = _sdpa(q, k, v, cfg, flash=flash)
    if q_diff is not None:
        out = out - _sdpa(q_diff, k_diff, v, cfg, flash=flash)
    return _linear(_merge(out), attn["to_out"]["weight"])


def _attend(
    attn: Params,
    q: Array,
    k: Array,
    v: Array,
    q_diff: Array | None,
    k_diff: Array | None,
    rope: tuple[Array, Array] | None,
    mask: Array | None,
    cfg: DiTConfig,
    dtype: Dtype,
    *,
    flash: bool = False,
) -> Array:
    """
    q RMSNorm -> RoPE -> SDPA -> `to_out`. `k`/`k_diff` arrive already k-normed: self-attention
    norms them inline, cross-attention hoists them into `static`. `q_diff`/`k_diff` are `None`
    unless `differential` (`medium`), in which case the two SDPAs share `v` and are subtracted.
    """
    q = _rms(q, attn["q_norm"]["weight"], cfg.qk_norm_eps, dtype)
    if q_diff is not None:
        q_diff = _rms(q_diff, attn["q_norm"]["weight"], cfg.qk_norm_eps, dtype)
    if rope is not None:
        q, k = _rope(q, *rope, dtype), _rope(k, *rope, dtype)
        if q_diff is not None:
            q_diff, k_diff = _rope(q_diff, *rope, dtype), _rope(k_diff, *rope, dtype)
    if mask is not None:
        v = v * mask
    return _attend_out(attn, q, k, v, q_diff, k_diff, cfg, flash=flash)


def _self_attention(h: Array, block: Params, static: Static, cfg: DiTConfig, dtype: Dtype) -> Array:
    """Fused `to_qkv` self-attention: the XLA split/norm/RoPE chain, or the fused qk-prep kernel.

    The kernel (`cfg.qk_prep == "pallas"`) applies RMSNorm, RoPE and the V-mask itself and returns
    `(B, H, 64 + T, D)`, so it lands on `_attend_out`; the XLA path normalises and rotates here and
    goes through `_attend`.
    """
    attn = block["self_attn"]
    qkv = _linear(h, attn["to_qkv"]["weight"])
    flash = cfg.attention == "flash"
    if cfg.qk_prep == "pallas":
        q, k, v, q_diff, k_diff = qk_prep.qk_prep(
            qkv, attn["q_norm"]["weight"], attn["k_norm"]["weight"], static["qk_cos"], static["qk_sin"],
            static["v_mask"], cfg.num_heads, cfg.rope_dim // 2, cfg.qk_norm_eps,
        )
        return _attend_out(attn, q, k, v, q_diff, k_diff, cfg, flash=flash)
    parts = jnp.split(qkv, 5 if cfg.differential else 3, axis=-1)
    q, k, v = (_heads(t, cfg) for t in parts[:3])
    q_diff, k_diff = (_heads(t, cfg) for t in parts[3:]) if cfg.differential else (None, None)
    k, k_diff = _k_norm(attn, k, k_diff, cfg, dtype)
    return _attend(attn, q, k, v, q_diff, k_diff, (static["cos"], static["sin"]), static["v_mask"],
                   cfg, dtype, flash=flash)


def _cross_attention(h: Array, static: Static, block: Params, index: int, cfg: DiTConfig, dtype: Dtype) -> Array:
    """`to_q` plus the hoisted `to_kv` K/V. No RoPE and no mask (`dit.py:1033-1035`): the
    reference only forwards `cross_attn_rotary_pos_emb` when it exists, and it is `None` here."""
    attn = block["cross_attn"]
    q = _linear(h, attn["to_q"]["weight"])
    if cfg.differential:
        q, q_diff = (_heads(t, cfg) for t in jnp.split(q, 2, axis=-1))
        k_diff = static["cross_k_diff"][index]
    else:
        q, q_diff, k_diff = _heads(q, cfg), None, None
    return _attend(
        attn, q, static["cross_k"][index], static["cross_v"][index], q_diff, k_diff, None, None, cfg, dtype
    )


def _feed_forward(x: Array, ff: Params) -> Array:
    """SwiGLU: `Linear(1024->8192)` -> `value * SiLU(gate)` -> `Linear(4096->1024)` (`GLU`)."""
    value, gate = jnp.split(_linear(x, ff["glu_proj"]["weight"], ff["glu_proj"]["bias"]), 2, axis=-1)
    return _linear(value * jax.nn.silu(gate), ff["proj_out"]["weight"], ff["proj_out"]["bias"])


def _block(h: Array, block: Params, static: Static, gcond: Array, index: int, cfg: DiTConfig, dtype: Dtype) -> Array:
    """`TransformerBlock.forward` with adaLN, in the reference's order (`transformer.py:1018-1049`).

    The gate is `sigmoid(1 - gate)` — the `1 -` is *inside* the sigmoid. `layer_scale` is off, so
    `self_attn_scale`/`cross_attn_scale`/`ff_scale` are `Identity` and do not appear.
    """
    scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = jnp.split(
        (block["to_scale_shift_gate"].astype(dtype.compute) + gcond)[:, None, :], 6, axis=-1
    )
    residual = h
    x = _rms(h, block["pre_norm"]["weight"], cfg.norm_eps, dtype)
    x = x * (1 + scale_self) + shift_self
    x = _self_attention(x, block, static, cfg, dtype)
    h = residual + x * jax.nn.sigmoid(1 - gate_self)

    x = _rms(h, block["cross_attend_norm"]["weight"], cfg.norm_eps, dtype)
    h = h + _cross_attention(x, static, block, index, cfg, dtype)

    # `_apply_local_conditioning` left-pads `(1, T, 1024)` to `64 + T` at the add site
    # (`transformer.py:982`), so the memory tokens receive nothing. See `_local_constants`.
    h = h.at[:, cfg.num_memory_tokens :, :].add(static["local"][index])

    residual = h
    x = _rms(h, block["ff_norm"]["weight"], cfg.norm_eps, dtype)
    x = x * (1 + scale_ff) + shift_ff
    x = _feed_forward(x, block["ff"])
    return residual + x * jax.nn.sigmoid(1 - gate_ff)


def _local_constants(blocks: tuple[Params, ...], dtype: Dtype) -> Array:
    """The 20 `to_local_embed` outputs for an all-zero `local_add_cond`, as `(depth, 1024)`.

    For text-to-audio `local_add_cond` is exactly zero (fixture `cond/local_add_cond`, absmax 0.0),
    so `Linear(0) == bias` exactly and the block output is bitwise constant along T — measured 0.0
    deviation across all 20 blocks in both the fp32 and fp16 groups. The closed form
    `W2 · SiLU(b1) + b2` is therefore **exact, not an approximation**; the `(64+T, 1024)`
    activation never exists. A non-zero `local_add_cond` (inpaint) would invalidate this.
    """
    return jnp.stack(
        [
            _linear(
                jax.nn.silu(block["to_local_embed"]["in"]["bias"].astype(dtype.compute)),
                block["to_local_embed"]["out"]["weight"],
                block["to_local_embed"]["out"]["bias"],
            )
            for block in blocks
        ]
    )


def _cross_kv(block: Params, ctx: Array, cfg: DiTConfig, dtype: Dtype) -> tuple[Array, ...]:
    """One block's cross-attention K/V: `to_kv` -> split -> heads -> `k_norm`, hoisted out of the
    sampler loop (`ctx` is constant, so this whole chain ran 24 x 8 times needlessly). Returns
    `(k, k_diff, v)` when `differential`, `(k, v)` otherwise; `v` is never normed."""
    attn = block["cross_attn"]
    kv = _linear(ctx, attn["to_kv"]["weight"])
    if cfg.differential:
        k, k_diff, v = (_heads(t, cfg) for t in jnp.split(kv, 3, axis=-1))
        return (*_k_norm(attn, k, k_diff, cfg, dtype), v)
    k, v = (_heads(t, cfg) for t in jnp.split(kv, 2, axis=-1))
    return _k_norm(attn, k, None, cfg, dtype)[0], v


def prepare_static(params: Params, cond: Params, padding_mask: Array, cfg: DiTConfig, dtype: Dtype) -> Static:
    """Everything `dit_forward` needs that does not depend on `x` or `t`.

    `cond` is `sa3jax.conditioner.assemble_conditioning`'s dict: `cross_attn_cond` `(B, 257, 768)`
    and `global_cond` `(B, 768)`. `cross_attn_mask` is discarded by the reference
    (`dit.py:411-414`) and `local_add_cond` is ignored because it is identically zero here.
    Per-block cross-attention K/V is hoisted here, already k-normed, as a stacked `(depth, ...)`
    pytree of arrays the sampler reuses every step.
    """
    net = params["transformer"]
    batch, seq_len = padding_mask.shape
    memory = cfg.num_memory_tokens
    local = cond.get("local_add_cond")
    if local is not None and float(jnp.max(jnp.abs(local))) != 0.0:
        raise ValueError(
            "local_add_cond is non-zero (inpaint): the constant `to_local_embed` shortcut holds "
            "only for an all-zero input. Implement the full per-position path before enabling it"
        )
    cos, sin = _rope_tables(memory + seq_len, net["rope_inv_freq"])
    # The mask is extended True over the 64 memory tokens (`transformer.py:1213-1225`) and cast to
    # the compute dtype: it multiplies V, it is not an additive attention mask.
    valid = jnp.concatenate([jnp.ones((batch, memory), dtype=bool), padding_mask.astype(bool)], axis=-1)
    ctx = _sequential(params["to_cond_embed"], cond["cross_attn_cond"].astype(dtype.compute))
    kv = [_cross_kv(block, ctx, cfg, dtype) for block in net["blocks"]]
    cross: Static = {
        "cross_k": jnp.stack([t[0] for t in kv]),
        "cross_v": jnp.stack([t[-1] for t in kv]),
    }
    if cfg.differential:
        cross["cross_k_diff"] = jnp.stack([t[1] for t in kv])
    static: Static = {
        "gcond_proj": _sequential(params["to_global_embed"], cond["global_cond"].astype(dtype.compute)),
        "local": _local_constants(net["blocks"], dtype),
        "cos": cos,
        "sin": sin,
        "v_mask": valid.astype(dtype.compute)[:, None, :, None],
        **cross,
    }
    if cfg.qk_prep == "pallas":
        # The kernel's padded length and lane-layout tables; `v_mask` is zero-extended with it so
        # the kernel's pad rows read a real mask (they are sliced off before SDPA either way).
        if net["rope_inv_freq"].shape[0] * 2 != cfg.rope_dim:
            raise ValueError(f"qk_prep's rotary width comes from cfg.rope_dim: {net['rope_inv_freq'].shape[0]}")
        padded = -(-(memory + seq_len) // qk_prep.ROWS) * qk_prep.ROWS
        static["qk_cos"], static["qk_sin"] = qk_prep.lane_freqs(padded, net["rope_inv_freq"])
        static["v_mask"] = jnp.pad(static["v_mask"], ((0, 0), (0, 0), (0, padded - (memory + seq_len)), (0, 0)))
    return static


def dit_forward(params: Params, x: Array, t: Array, static: Static, cfg: DiTConfig, dtype: Dtype) -> Array:
    """One denoiser evaluation: `(B, 256, T)` latents and `(B,)` timesteps -> `v`, same shape."""
    net = params["transformer"]
    compute = dtype.compute

    h = x.astype(compute)
    h = _conv1x1(h, params["preprocess_conv"]["weight"].astype(compute)) + h
    h = _linear(h.transpose(0, 2, 1), net["project_in"]["weight"].astype(compute))
    memory = jnp.broadcast_to(
        net["memory_tokens"].astype(compute), (h.shape[0], cfg.num_memory_tokens, cfg.embed_dim)
    )
    h = jnp.concatenate([memory, h], axis=1)

    timestep = _sequential(params["to_timestep_embed"], _expo_fourier(t.astype(compute), cfg.fourier_dim))
    gcond = _sequential(net["global_cond_embedder"], static["gcond_proj"] + timestep)

    for index, block in enumerate(net["blocks"]):
        h = _block(h, block, static, gcond, index, cfg, dtype)

    h = _linear(h[:, cfg.num_memory_tokens :, :], net["project_out"]["weight"].astype(compute))
    v = h.transpose(0, 2, 1)
    return _conv1x1(v, params["postprocess_conv"]["weight"].astype(compute)) + v
