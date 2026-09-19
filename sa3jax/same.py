"""SAME-S audio decoder (`small-music`), pure functional JAX — **unchunked**.

Ground truth: `models/autoencoders.py` (`TransformerResamplingBlock.forward:99-222`, decoder
branch; `SAMEDecoder:290-348`; `_zero_pad_modulo_sequence:25-32`), `models/bottleneck.py:50-71`
(`SoftNormBottleneck.decode`), `models/transformer.py` (`TransformerBlock.forward:997-1068`,
`Attention:704-820`, `DynamicTanh:325-334`, `GLU:421-444`, `RotaryEmbedding:239-322`) and
`models/pretransforms.py` (`PatchedPretransform.decode`).

`decode` returns **audio, not patches**: the patched pretransform's `b (c h) l -> b c (l h)`
rearrange (`h = 256`) is part of the decoder's output contract, so the caller gets
`(B, 2, T * 4096)` in one call.

**Both noises are injected, never drawn** (decision D2). The reference is stochastic in eval mode —
`SoftNormBottleneck.decode` adds `randn_like(x) * running_std * 1e-3` (training uses `5e-2`) and the
resampling block adds `randn_like(new_tokens) * mask_noise` — and torch's Philox stream is not JAX's
Threefry, so consuming the fixture's verbatim draws is the only way to compare. The function is
pure: no RNG, no side effects, both noises are ordinary arguments.

`jax.jit` compatibility is a hard requirement, so `params` must be a jit **argument**: closing over
them lets XLA constant-fold 54.5 M parameters (754 s of compile time instead of 3.4 s, M3).

Two fp32 islands, both measured rather than assumed: the differential subtraction (`fp16` cancels
catastrophically, `[R]`) and the attention's QK^T/PV accumulation, whose absence costs the bf16 tier
**36 dB** (see `_sdpa`). The fp32 tier is bit-identical with and without either.

Out of scope, one line each: the AE **encoder** (dead at inference, its 120 checkpoint keys are
dropped by `weights.py`); the **chunked/overlapped** decode path (`autoencoders.py:596-638`) — this
module is one call over the whole `T`, no overlap and no stitching, and chunked and unchunked decode
are *different functions*, so it is only comparable with an unchunked reference group.
"""
from __future__ import annotations

import math
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import Array

from sa3jax.config import SameConfig
from sa3jax.dtypes import Dtype

Params = dict[str, Any]

# `PatchedPretransform.patch_size`: the decoder emits `(B, out_channels, L)` patches which the
# pretransform folds back into `(B, out_channels // 256, L * 256)` audio samples.
PATCH_SIZE: Final[int] = 256


def _linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    """`nn.Linear`; the checkpoint stores `[out, in]`."""
    y = x @ jnp.swapaxes(weight, -1, -2)
    return y if bias is None else y + bias


def _dyt(x: Array, norm: Params) -> Array:
    """`DynamicTanh`: `gamma * tanh(alpha * x) + beta` — a bounded squash, **not** a normaliser.

    `dyt: true` puts it in place of RMSNorm/LayerNorm, so there is no mean-square anywhere in this
    decoder. `alpha` is a scalar `(1,)`, `gamma`/`beta` are `(dim,)`.
    """
    alpha, gamma, beta = (norm[name].astype(x.dtype) for name in ("alpha", "gamma", "beta"))
    return gamma * jnp.tanh(alpha * x) + beta


def _rope_tables(seq_len: int, inv_freq: Array) -> tuple[Array, Array]:
    """`RotaryEmbedding(dim_heads // 2)`: 16 frequencies, duplicated to 32 rotary dims.

    The tables are built for the **chunk** length, so positions restart at 0 inside every 34-token
    chunk — the fold is positional only within a chunk (`transformer.py:1015-1016`).
    """
    angles = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv_freq.astype(jnp.float32)[None, :]
    freqs = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(freqs), jnp.sin(freqs)


def _rope(x: Array, cos: Array, sin: Array, dtype: Dtype) -> Array:
    """Partial rotary with half-half `rotate_half` — Llama-style, **not** interleaved.

    `rot_dim = 32` of the 64 head dims. The angle math runs in the fp32 `dtype.rope` island and the
    result is cast back (`transformer.py:302-322`).
    """
    out_dtype = x.dtype
    rot_dim = cos.shape[-1]
    x = x.astype(dtype.rope)
    rot, rest = x[..., :rot_dim], x[..., rot_dim:]
    rotated = jnp.concatenate([-rot[..., rot_dim // 2 :], rot[..., : rot_dim // 2]], axis=-1)
    return jnp.concatenate([rot * cos + rotated * sin, rest], axis=-1).astype(out_dtype)


def _sdpa(q: Array, k: Array, v: Array, cfg: SameConfig) -> Array:
    """Unmasked softmax attention at `1/sqrt(head_dim)`, **both products accumulated in fp32**.

    There is no padding mask inside the decoder: every chunk is a dense 34-token window. The fp32
    accumulation is not decoration. The SAME-S logits are enormous — rms 426, absmax 2269, because
    the DyT `gamma`s are large — so the softmax is a near-hard argmax (0.7 % of rows above 0.99,
    mean top weight 0.52) and a bf16 accumulation, which is measurably worse than fp32 accumulation
    of the same bf16 products (50.88 vs 52.64 dB from fp64), flips ~1 % of the argmaxes and costs
    **36 dB**: the bf16 tier decodes to absmax 16.9 instead of 1.4. Upcasting q/k/v is exact (they
    are already bf16-representable) and reproduces torch's SDPA contract, whose half-precision
    kernels accumulate QK^T and PV in fp32; in the fp32 tier the casts are no-ops, so the fp32
    parity numbers are bit-identical with and without this.
    """
    scale = 1.0 / math.sqrt(cfg.dim_heads)
    logits = jnp.einsum("nhsd,nhtd->nhst", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("nhst,nhtd->nhsd", weights, v.astype(jnp.float32)).astype(q.dtype)


def _attention(x: Array, attn: Params, rope: tuple[Array, Array], cfg: SameConfig, dtype: Dtype) -> Array:
    """Differential self-attention: `SDPA(q, k, v) - SDPA(q_diff, k_diff, v)`, **shared v**.

    `to_qkv` is `Linear(768, 5 * 768)` and chunks in the reference's order `q, k, v, q_diff,
    k_diff` (`transformer.py:738`). Both q halves go through the same DyT and RoPE, likewise k.
    """
    compute = dtype.compute
    parts = jnp.split(_linear(x, attn["to_qkv"]["weight"].astype(compute)), 5, axis=-1)
    q, k, v, q_diff, k_diff = (
        t.reshape(t.shape[0], t.shape[1], cfg.num_heads, cfg.dim_heads).transpose(0, 2, 1, 3)
        for t in parts
    )
    q = _rope(_dyt(q, attn["q_norm"]), *rope, dtype)
    k = _rope(_dyt(k, attn["k_norm"]), *rope, dtype)
    q_diff = _rope(_dyt(q_diff, attn["q_norm"]), *rope, dtype)
    k_diff = _rope(_dyt(k_diff, attn["k_norm"]), *rope, dtype)

    main = _sdpa(q, k, v, cfg)
    diff = _sdpa(q_diff, k_diff, v, cfg)
    # fp32 island: in fp16/bf16 the two branches cancel catastrophically, so the subtraction runs at
    # full precision and rounds once (`docs/jax-port.md` §5, TensorRT/MLX port notes).
    out = (main.astype(jnp.float32) - diff.astype(jnp.float32)).astype(compute)
    out = out.transpose(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], cfg.dim)
    return _linear(out, attn["to_out"]["weight"].astype(compute))


def _feed_forward(x: Array, ff: Params, dtype: Dtype) -> Array:
    """SwiGLU (`sinusoidal_blocks = 0` for SAME-S): `Linear(768->4608)`, `value * SiLU(gate)`,
    `Linear(2304->768)`. The activation is on the **second** half (`GLU.forward:443`)."""
    compute = dtype.compute
    glu = _linear(x, ff["glu_proj"]["weight"].astype(compute), ff["glu_proj"]["bias"].astype(compute))
    value, gate = jnp.split(glu, 2, axis=-1)
    out = value * jax.nn.silu(gate)
    return _linear(out, ff["proj_out"]["weight"].astype(compute), ff["proj_out"]["bias"].astype(compute))


def _block(x: Array, block: Params, cfg: SameConfig, dtype: Dtype) -> Array:
    """`TransformerBlock.forward` in the decoder's configuration: `global_cond_dim`,
    `local_add_cond_dim`, `cross_attend`, `conformer` and `layer_scale` are all `None`/off, so the
    adaLN branch is skipped and the block is exactly two residual branches (`transformer.py:1051-1065`).
    """
    rope = _rope_tables(x.shape[-2], block["rope_inv_freq"])
    h = x + _attention(_dyt(x, block["pre_norm"]), block["self_attn"], rope, cfg, dtype)
    return h + _feed_forward(_dyt(h, block["ff_norm"]), block["ff"], dtype)


def _mapping(x: Array, mapping: Params, dtype: Dtype) -> Array:
    """Weight-norm-folded `Conv1d(768 -> 512, k=3, padding='same')` on a channel-first tensor.

    `weights.py` has already folded `weight_g`/`weight_v` into `weight`; the kernel is stored
    `[out, in, k]` and JAX's `"OIW"` reads it as-is — no permutation (`docs/jax-port.md` M0).
    """
    compute = dtype.compute
    kernel = mapping["weight"].astype(compute)
    radius = kernel.shape[-1] // 2
    y = jax.lax.conv_general_dilated(
        x.transpose(0, 2, 1), kernel, (1,), ((radius, radius),), dimension_numbers=("NWC", "OIW", "NWC")
    )
    return (y + mapping["bias"].astype(compute)).transpose(0, 2, 1)


def _unpatch(x: Array, cfg: SameConfig) -> Array:
    """The patched pretransform's `b (c h) l -> b c (l h)`, `h = 256` (`pretransforms.py:76`)."""
    channels = cfg.out_channels // PATCH_SIZE
    return (
        x.reshape(x.shape[0], channels, PATCH_SIZE, x.shape[-1])
        .transpose(0, 1, 3, 2)
        .reshape(x.shape[0], channels, -1)
    )


def _forward(
    params: Params, latents: Array, dither: Array, mask_noise: Array, cfg: SameConfig, dtype: Dtype
) -> dict[str, Array]:
    """The decoder, plus the taps the fixture records as `dec/chunk0/*` (rungs 11-13).

    `latents` is `(B, 256, T)` channel-first; `dither` is the bottleneck's `randn_like(x)` draw
    `(B, 256, T)`; `mask_noise` is the resampling block's `randn_like(new_tokens)` draw
    `(B * T, 16, dim)` — the frozen interface's `(T, 16, dim)` at `B = 1`, which is the reference's
    own shape because it draws over the expanded `(B * T, 16, dim)` tensor.
    """
    compute = dtype.compute
    batch, _, length = latents.shape

    # bottleneck first: `x * running_std`, then the dither at the **eval** scale. `scaling_factor`
    # and `bias` are encode-only and do not appear in `SoftNormBottleneck.decode`.
    running_std = params["bottleneck"]["running_std"].astype(compute)
    bottleneck_out = latents.astype(compute) * running_std
    bottleneck_out = bottleneck_out + (dither.astype(compute) * running_std) * 1e-3

    project = params["project_in"]
    x = _linear(
        bottleneck_out.transpose(0, 2, 1), project["weight"].astype(compute), project["bias"].astype(compute)
    )
    block_in = x.transpose(0, 2, 1)

    # `_zero_pad_modulo_sequence(x, chunk_size // stride)`: SAME-S needs an even latent count. The
    # pad is applied *after* `project_in`, as the reference does — a zero latent would project to
    # `bias`, not to zero. A padded slot's own noise draw is unknowable to the caller, so it is
    # zero; even `T` (the only case the fixture exercises) never enters this branch.
    pad = (-length) % (cfg.chunk_size // cfg.stride)
    if pad:
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
        mask_noise = jnp.pad(mask_noise, ((0, pad), (0, 0), (0, 0)))
        length += pad

    # one learned `new_tokens` vector broadcast to `stride` positions after each latent slot,
    # plus the injected block noise (`mask_noise = 0.01` for SAME-S)
    sub = cfg.sub_chunk_size
    tokens = jnp.broadcast_to(
        params["new_tokens"].astype(compute), (batch * length, cfg.stride, cfg.dim)
    ) + mask_noise.reshape(batch * length, cfg.stride, cfg.dim).astype(compute) * cfg.mask_noise
    x = jnp.concatenate([x.reshape(batch * length, 1, cfg.dim), tokens], axis=1).reshape(
        batch, length * sub, cfg.dim
    )

    # Non-overlapping 34-token chunks for the first half of the stack ...
    effective, shift = cfg.effective_chunk_size, cfg.effective_chunk_size // 2
    split = cfg.transformer_depth // 2
    chunks = x.reshape(batch * (x.shape[1] // effective), effective, cfg.dim)
    taps: dict[str, Array] = {}
    for index, block in enumerate(params["blocks"][:split]):
        chunks = _block(chunks, block, cfg, dtype)
        taps[f"layer{index:02d}"] = chunks

    # ... then the 17-token midpoint shift: half a chunk is repeated at both ends and the sequence
    # is refolded, so the second half sees the neighbouring sub-chunk across every original chunk
    # boundary. The shift is undone by the same slice afterwards (`autoencoders.py:160-197`).
    x = chunks.reshape(batch, -1, cfg.dim)
    x = jnp.concatenate([x[:, :shift], x, x[:, -shift:]], axis=1)
    chunks = x.reshape(batch * (x.shape[1] // effective), effective, cfg.dim)
    for index, block in enumerate(params["blocks"][split:], start=split):
        chunks = _block(chunks, block, cfg, dtype)
        taps[f"layer{index:02d}"] = chunks
    x = chunks.reshape(batch, -1, cfg.dim)[:, shift:-shift]

    # drop the latent slot, keep the 16 new-token positions of each 17-token sub-chunk
    x = x.reshape(batch, length, sub, cfg.dim)[:, :, 1:].reshape(batch, length * cfg.stride, cfg.dim)
    block_out = _mapping(x.transpose(0, 2, 1), params["mapping"], dtype)
    return {
        "bottleneck_out": bottleneck_out,
        "block_in": block_in,
        "block_out": block_out,
        "audio": _unpatch(block_out, cfg),
        **taps,
    }


def decode_features(
    params: Params, latents: Array, dither: Array, mask_noise: Array, cfg: SameConfig, dtype: Dtype
) -> dict[str, Array]:
    """`decode` plus the intermediate taps, keyed by the fixture's `dec/chunk0/*` names."""
    return _forward(params, latents, dither, mask_noise, cfg, dtype)


def decode(
    params: Params, latents: Array, dither: Array, mask_noise: Array, cfg: SameConfig, dtype: Dtype
) -> Array:
    """`(B, 256, T)` latents -> `(B, 2, T * 4096)` audio, in one unchunked call.

    `T` must be even for SAME-S (the reference pads internally to 2 latents otherwise, and then the
    output is `(T + 1) * 4096` samples long). Returns `dec/audio_out_raw`: the mask zeroing, clamp
    and duration truncation are the caller's output handling, not the decoder's.
    """
    return _forward(params, latents, dither, mask_noise, cfg, dtype)["audio"]
