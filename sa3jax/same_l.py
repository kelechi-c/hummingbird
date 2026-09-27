"""SAME-L audio decoder (the `medium` codec), pure functional JAX.

Ground truth: `models/autoencoders.py` — `SAMEDecoder:290-348`, `TransformerResamplingBlock`
`:34-222` (only the `sliding_window is not None` paths), `_get_sliding_window_size:84-88` — plus
`models/transformer.py`: `Attention:523-820`, `DynamicTanh:325-334`, `RotaryEmbedding:239-294`,
`GLU:421-444`, `FeedForward:453-521`, `Sin:446-451`, `TransformerBlock.forward:1051-1065`.

SAME-L is a different execution strategy, not a bigger SAME-S. At `sliding_window=[1,1]` the whole
chunk-fold branch is skipped (`autoencoders.py:156`), so none of SAME-S's fold, midpoint-shift or
`effective_chunk_size` machinery exists here; locality comes from a sliding-window attention band
instead. `conv_mapping=False` selects `nn.Conv1d(kernel_size=1)`, and `chunk_midpoint_shift` is
inert for the same reason. What is shared with SAME-S is DyT, differential attention and SwiGLU.

The band. `_get_sliding_window_size` scales the latent-unit window by `stride + 1` = 17 (one latent
plus its 16 new-token positions), so the reference hands `window_size=(17, 17)` to flash attention
and `_sliding_window_additive_mask` builds `|k - q| <= 17` on absolute positions. Queries are laid
out in groups of 17, and one group's union of bands is exactly 3 consecutive KV groups = 51 keys —
the form implemented here, `(groups, 17, 51)`, built once per sequence length and shared by all 12
blocks. Softmax over the 51-key window with the out-of-band entries at `-inf` is the same function
as the reference's 35-key band, not an approximation of it.

The differential pair accumulates in `dtype.norm` (fp32 in every preset) and is subtracted there
before a single cast back: rounding both branches to bf16/fp16 first would make the subtraction
itself the lossy step that the plan's fp32 island exists to avoid (`docs/jax-port.md` §5). Under
`FP32_PARITY` this is exactly the reference; under bf16/fp16 it is a deliberate deviation from
torch's own `out - out_diff` in the compute dtype, unmeasurable without a `medium` fixture.

Out of scope, deliberately: chunked decode (this is the unchunked path), the AE encoder,
`override_new_tokens` / `prepend_cond` / cross-attention / `feat_scale` (all off or `None` for the
decoder), and the output-side mask zeroing, clamp and duration truncation, which belong to the
pipeline.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from sa3jax.config import SameConfig
from sa3jax.dtypes import Dtype

Params = dict[str, Any]

_AUDIO_CHANNELS = 2  # `patched` pretransform `channels: 2`; patch_size = 512 // 2 = 256
_DITHER_SCALE = 1e-3  # `SoftNormBottleneck.decode`'s eval-path `scale`


def _linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    """`nn.Linear`; the checkpoint stores `[out, in]`."""
    y = x @ jnp.swapaxes(weight, -1, -2)
    return y if bias is None else y + bias


def _dyt(x: Array, norm: Params) -> Array:
    """`DynamicTanh`: `gamma * tanh(alpha * x) + beta`, `alpha` a scalar (`transformer.py:325-334`).

    An fp32 island, returned in fp32. There is no reduction to overflow; the loss is rounding: bf16
    rounds four times here, the trained gammas are large (logits reach absmax 2269) and the band
    softmax is near-argmax, so each rounding flips winners. Measured on trained medium weights: bf16
    DyT alone costs the decode 16.7 dB SNR vs 36.3 dB at fp32 (MERT-v2 0.960 vs 0.9987), for +7 %
    decode time. Callers cast back to the compute dtype before the next GEMM, or fp32 promotion
    would silently turn every following matmul fp32; q/k stay fp32 through RoPE into `QK^T`.
    """
    f32 = jnp.float32
    return norm["gamma"].astype(f32) * jnp.tanh(norm["alpha"].astype(f32) * x.astype(f32)) + norm["beta"].astype(f32)


def _heads(x: Array, cfg: SameConfig) -> Array:
    """`b n (h d) -> b h n d`."""
    batch, seq, _ = x.shape
    return x.reshape(batch, seq, cfg.num_heads, cfg.dim_heads).transpose(0, 2, 1, 3)


def _merge(x: Array) -> Array:
    """`b h n d -> b n (h d)`."""
    batch, heads, seq, dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq, heads * dim)


def _rope_tables(seq_len: int, inv_freq: Array, dtype: Dtype) -> tuple[Array, Array]:
    """`RotaryEmbedding(dim_heads // 2)` over the full `T * 17` sequence. `cat(freqs, freqs)` is
    the partial-rotary duplication: `rot_dim = 32` of the 64 head dims."""
    angles = jnp.arange(seq_len, dtype=dtype.rope)[:, None] * inv_freq.astype(dtype.rope)[None, :]
    freqs = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(freqs), jnp.sin(freqs)


def _rope(x: Array, cos: Array, sin: Array, dtype: Dtype) -> Array:
    """Half-half `rotate_half` — Llama-style, **not** interleaved — computed in fp32 and cast back
    (`transformer.py:296-322`). `cos`/`sin` are `(S, rot_dim)`."""
    out_dtype = x.dtype
    rot_dim = cos.shape[-1]
    x = x.astype(dtype.rope)
    rot, rest = x[..., :rot_dim], x[..., rot_dim:]
    rotated = jnp.concatenate([-rot[..., rot_dim // 2 :], rot[..., : rot_dim // 2]], axis=-1)
    return jnp.concatenate([rot * cos + rotated * sin, rest], axis=-1).astype(out_dtype)


def _swa_mask(groups: int, cfg: SameConfig) -> Array:
    """The static `(groups, 17, 51)` boolean sliding-window band.

    Query `q` of group `g` is absolute position `17g + q`; window slot `w` is absolute position
    `17g + w - 17`. The reference's `|k - q| <= 17` is therefore `q <= w <= q + 34`, and the guard
    drops window slots that fall outside `[0, 17 * groups)` — the reference needs no such guard
    because it builds the band on the unpadded `(seq_q, seq_k)` pair directly.
    """
    stride = cfg.sub_chunk_size
    q = jnp.arange(stride)[None, :, None]
    w = jnp.arange(3 * stride)[None, None, :]
    pos = jnp.arange(groups)[:, None, None] * stride + w - stride
    return (w >= q) & (w <= q + 2 * stride) & (pos >= 0) & (pos < groups * stride)


def _windows(t: Array, groups: int, stride: int) -> Array:
    """`(B, H, S, D) -> (B, H, groups, 3 * stride, D)`: each query group's 3 consecutive KV groups.

    The sequence is zero-padded by one group on each side and three overlapping `(groups, stride)`
    views of it are concatenated — a stride-`stride` sliding window written without a scan. Padding
    slots are always out of band, so their values never reach the softmax.
    """
    batch, heads, seq, dim = t.shape
    padded = jnp.pad(t, ((0, 0), (0, 0), (stride, stride), (0, 0)))
    parts = [
        padded[:, :, off : off + seq, :].reshape(batch, heads, groups, stride, dim)
        for off in (0, stride, 2 * stride)
    ]
    return jnp.concatenate(parts, axis=3)


def _sdpa(q: Array, k: Array, v: Array, mask: Array, cfg: SameConfig, dtype: Dtype) -> Array:
    """Grouped sliding-window attention at `1/sqrt(head_dim)`, fp32 accumulation.

    Softmax runs in fp32 the way torch's SDPA does; the weighted sum and the caller's differential
    subtraction run in `dtype.norm` — the plan's fp32 island. Returns `dtype.norm` arrays.
    """
    groups, stride = mask.shape[0], cfg.sub_chunk_size
    batch, heads, seq, dim = q.shape
    qg = q.reshape(batch, heads, groups, stride, dim)
    kw, vw = _windows(k, groups, stride), _windows(v, groups, stride)
    logits = jnp.einsum("bhgqd,bhgwd->bhgqw", qg, kw) * (1.0 / math.sqrt(cfg.dim_heads))
    logits = jnp.where(mask[None, None], logits.astype(dtype.norm), -jnp.inf)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhgqw,bhgwd->bhgqd", weights, vw.astype(dtype.norm))
    return out.reshape(batch, heads, seq, dim)


def _attention(x: Array, block: Params, mask: Array, cfg: SameConfig, dtype: Dtype) -> Array:
    """Differential self-attention: fused `to_qkv` split 5 ways, shared `v`, both branches at the
    same band (`transformer.py:738-741, 790-795`). q/k DyT and RoPE precede attention, and the two
    differential halves reuse the same `q_norm`/`k_norm` (`apply_qk_layernorm` runs on the stacked
    pair)."""
    attn = block["self_attn"]
    q, k, v, q_diff, k_diff = (_heads(t, cfg) for t in jnp.split(_linear(x, attn["to_qkv"]["weight"]), 5, axis=-1))
    cos, sin = _rope_tables(x.shape[1], block["rope_inv_freq"], dtype)
    q = _rope(_dyt(q, attn["q_norm"]), cos, sin, dtype)
    k = _rope(_dyt(k, attn["k_norm"]), cos, sin, dtype)
    q_diff = _rope(_dyt(q_diff, attn["q_norm"]), cos, sin, dtype)
    k_diff = _rope(_dyt(k_diff, attn["k_norm"]), cos, sin, dtype)
    out = _sdpa(q, k, v, mask, cfg, dtype) - _sdpa(q_diff, k_diff, v, mask, cfg, dtype)
    return _linear(_merge(out.astype(x.dtype)), attn["to_out"]["weight"])


def _feed_forward(x: Array, block: Params, index: int, cfg: SameConfig) -> Array:
    """`GLU` gated by `SiLU`, or by `sin(pi * x)` on the sinusoidal blocks.

    `sinusoidal_blocks` counts from the END (`autoencoders.py:61`: `depth - i < sinusoidal_blocks`),
    so the sine covers `i >= depth - sinusoidal_blocks + 1` — blocks 5..11 of 12 at the medium
    preset's 8. `GLU.forward` takes `value` as the first half of the projection and `gate` as the
    second, then returns `value * act(gate)`.
    """
    ff = block["ff"]
    value, gate = jnp.split(_linear(x, ff["glu_proj"]["weight"], ff["glu_proj"]["bias"]), 2, axis=-1)
    sinusoidal = index >= cfg.transformer_depth - cfg.sinusoidal_blocks + 1
    activated = jnp.sin(math.pi * gate) if sinusoidal else jax.nn.silu(gate)
    return _linear(value * activated, ff["proj_out"]["weight"], ff["proj_out"]["bias"])


def _block(x: Array, block: Params, index: int, mask: Array, cfg: SameConfig, dtype: Dtype) -> Array:
    """`TransformerBlock.forward`'s non-adaLN branch (`transformer.py:1051-1065`). No cross-attn, no
    conformer, no local conditioning, and `layer_scale` is off so `self_attn_scale`/`ff_scale` are
    `Identity` and drop out of the two residual adds."""
    h = x + _attention(_dyt(x, block["pre_norm"]).astype(dtype.compute), block, mask, cfg, dtype)
    return h + _feed_forward(_dyt(h, block["ff_norm"]).astype(dtype.compute), block, index, cfg)


def _patch(x: Array, cfg: SameConfig) -> Array:
    """`PatchedPretransform.decode`'s `b (c h) l -> b c (l h)` with `h = 512 // 2 = 256`."""
    batch, _, length = x.shape
    patch = cfg.out_channels // _AUDIO_CHANNELS
    return x.reshape(batch, _AUDIO_CHANNELS, patch, length).transpose(0, 1, 3, 2).reshape(
        batch, _AUDIO_CHANNELS, length * patch
    )


def decode(
    params: Params, latents: Array, dither: Array, mask_noise: Array, cfg: SameConfig, dtype: Dtype
) -> Array:
    """`(B, 256, T)` latents -> `(B, 2, T * 4096)` audio, in one call over the whole sequence.

    `dither` is the verbatim `torch.randn_like` draw from `SoftNormBottleneck.decode` and
    `mask_noise` the one from the resampling block's `new_tokens`; both are injected, never drawn
    (`docs/jax-port.md` D2). `mask_noise` is `(T, 16, dim)` — the reference draws it per
    `(batch, latent)` row, so a batch axis would carry independent draws; a `(B, T, 16, dim)` input
    is also accepted and broadcasts correctly. The bottleneck's `scaling_factor`/`bias` are encode
    -only and unused here.
    """
    if cfg.sliding_window is None or cfg.conv_mapping:
        raise ValueError(
            "same_l.decode implements the SAME-L path only: it needs sliding_window set (the fold "
            "branch is skipped) and conv_mapping False (a 1x1 mapping). Use sa3jax.same for SAME-S."
        )
    compute = dtype.compute

    std = params["bottleneck"]["running_std"].astype(compute)
    z = latents.astype(compute) * std
    z = z + dither.astype(compute) * std * _DITHER_SCALE

    h = _linear(z.transpose(0, 2, 1), params["project_in"]["weight"], params["project_in"]["bias"])
    batch, latents_len, _ = h.shape
    new = params["new_tokens"].astype(compute) + mask_noise.astype(compute) * cfg.mask_noise
    h = jnp.concatenate(
        [h[:, :, None, :], jnp.broadcast_to(new, (batch, latents_len, cfg.new_tokens_per_latent, cfg.dim))],
        axis=2,
    ).reshape(batch, latents_len * cfg.sub_chunk_size, cfg.dim)

    mask = _swa_mask(latents_len, cfg)
    for index, block in enumerate(params["blocks"]):
        h = _block(h, block, index, mask, cfg, dtype)

    kept = h.reshape(batch, latents_len, cfg.sub_chunk_size, cfg.dim)[:, :, 1:]
    kept = kept.reshape(batch, latents_len * cfg.new_tokens_per_latent, cfg.dim)
    # `mapping` is `WNConv1d(kernel_size=1)`: the checkpoint stores `[out, in, 1]`, and the bias is
    # per output channel, so it broadcasts on axis 1 rather than on the trailing length axis.
    mapping = params["mapping"]
    audio = jnp.einsum("bld,od->bol", kept, mapping["weight"][:, :, 0]) + mapping["bias"][None, :, None]
    return _patch(audio, cfg)
