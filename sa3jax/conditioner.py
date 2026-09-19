"""Number conditioner + conditioning assembly — the non-text half of `cond/*` (rungs 3-5).

Ground truth: `models/conditioners.py:41-68` (`apply_padding`) and `:121-155`
(`NumberConditioner`), `models/blocks.py:50-82` (`ExpoFourierFeatures`), assembled by
`models/diffusion.py:97-194` (`get_conditioning_inputs`) for the text-to-audio case.

The prompt encoder is deliberately not imported: `assemble_conditioning` takes its output as an
argument, so this module is independent of the T5Gemma port and testable on its own.

Text-to-audio only. The inpaint path is out of scope, which is exactly why `local_add_cond` is
zeros rather than something built: `model.py:282-297` makes both halves (`inpaint_mask`,
`inpaint_masked_input`) `torch.zeros` when no inpaint audio is supplied.
"""
from __future__ import annotations

import math
from typing import Final

import jax.numpy as jnp
from jax import Array

from sa3jax.config import ConditionerConfig
from sa3jax.dtypes import Dtype

ConditionerParams = dict[str, Array]

# The only conditioner tensors the shipped checkpoint carries, loaded by `weights.py` (the single
# checkpoint-to-pytree mapper — this module deliberately has no loader of its own).
# `prompt.proj_out` is absent by construction (`conditioners.py:35`): `dim == output_dim == 768`, so
# the projection is Identity and the prompt path is `t5_last_hidden` unchanged. `embedding.0` (the
# Fourier features) has no parameters, hence the `.1.` in the seconds key.
# `dit.local_add_cond_dim` = inpaint mask (1) + inpaint masked input (`io_channels` = 256).
LOCAL_ADD_COND_CHANNELS: Final[int] = 257


def expo_fourier_features(t: Array, dim: int, min_freq: float, max_freq: float) -> Array:
    """`blocks.py:50-82`: geometric frequencies `min_freq -> max_freq` over `dim // 2` bands.

    `t` is `(B,)` or `(B, 1)`. Every step runs in fp32 (`autocast` is disabled upstream) and the
    result is cast back to the input dtype, so a bf16/fp16 caller still gets fp32-precision
    frequencies. The `2 * pi` factor is applied last, as upstream.
    """
    in_dtype = t.dtype
    t = t.astype(jnp.float32)
    if t.ndim == 1:
        t = t[:, None]
    ramp = jnp.linspace(0.0, 1.0, dim // 2, dtype=jnp.float32)
    freqs = jnp.exp(ramp * (math.log(max_freq) - math.log(min_freq)) + math.log(min_freq))
    args = t * freqs * (2.0 * math.pi)
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(in_dtype)


def number_conditioner(
    seconds: Array, params: ConditionerParams, cfg: ConditionerConfig, dtype: Dtype
) -> tuple[Array, Array]:
    """`conditioners.py:121-155`: clamp -> normalise -> expo Fourier -> Linear -> `unsqueeze(1)`.

    Returns the `(B, 1, cond_dim)` embedding and the `(B, 1)` all-ones mask. Clamping and
    normalising happen in fp32 and are only then cast to the embedder's dtype, matching the
    reference order; the mask is `torch.ones`, i.e. fp32 in every dtype group.
    """
    x = jnp.clip(seconds.astype(jnp.float32), cfg.seconds_min, cfg.seconds_max)
    x = ((x - cfg.seconds_min) / (cfg.seconds_max - cfg.seconds_min)).astype(dtype.compute)
    features = expo_fourier_features(x, cfg.fourier_dim, cfg.fourier_min_freq, cfg.fourier_max_freq)
    weight = params["seconds_weight"].astype(dtype.compute)
    bias = params["seconds_bias"].astype(dtype.compute)
    embed = jnp.einsum("...i,oi->...o", features, weight, preferred_element_type=dtype.norm) + bias
    return embed.astype(dtype.compute)[:, None, :], jnp.ones((x.shape[0], 1), dtype=jnp.float32)


def apply_padding(embeddings: Array, attention_mask: Array, padding_embedding: Array) -> Array:
    """`conditioners.py:41-68` with `padding_mode="learned"`: pad rows are *replaced* by one
    learned `(cond_dim,)` vector. It runs after `proj_out`, which is Identity here.

    `where` promotes the way `torch.where` does, so a bf16 prompt and an fp32 padding embedding
    produce an fp32 result — the dtype the fixture records for `prompt_padded`.
    """
    return jnp.where(attention_mask[:, :, None], embeddings, padding_embedding)


def assemble_conditioning(
    prompt_hidden: Array,
    prompt_mask: Array,
    seconds: Array,
    params: ConditionerParams,
    cfg: ConditionerConfig,
    dtype: Dtype,
    latent_len: int | None = None,
) -> dict[str, Array]:
    """`get_conditioning_inputs` for text-to-audio (`diffusion.py:97-194`).

    `cross_attn_cond` concatenates `[prompt(256) || seconds(1)]` on the token axis and
    `global_cond` is the seconds embedding alone (the `(B, 1, D)` -> `(B, D)` squeeze).
    `cross_attn_mask` is fp32 `{0,1}` — `torch.cat` of the bool prompt mask and a fp32 ones mask
    — and is emitted for parity only: the DiT discards it (`dit.py:411-414`), so no masking is
    implemented anywhere in the port.

    `latent_len` is `T`, the sampler's latent length. It is the only input that comes from
    outside the conditioner; when given, `local_add_cond` is emitted as `(B, 257, T)` zeros and
    the DiT left-pads it to `64 + T` (`transformer.py:77-88`, `_left_pad_to_match`). When it is
    `None` the key is omitted — the `cond_expanded` reference has no sampler stage and no
    `local_add_cond`.
    """
    # Identity `proj_out` (asserted at load). The fixture stores this bf16 hidden state upcast to
    # fp32, so emit it the same way rather than in the encoder's dtype.
    prompt_proj = prompt_hidden.astype(jnp.float32)
    prompt_padded = apply_padding(
        prompt_proj, prompt_mask, params["padding_embedding"].astype(dtype.compute)
    )
    seconds_embed, seconds_mask = number_conditioner(seconds, params, cfg, dtype)

    assembled = {
        "prompt_proj": prompt_proj,
        "prompt_padded": prompt_padded,
        "seconds_embed": seconds_embed,
        "global_cond": seconds_embed[:, 0, :],
        "cross_attn_cond": jnp.concatenate([prompt_padded, seconds_embed], axis=1),
        "cross_attn_mask": jnp.concatenate(
            [prompt_mask.astype(jnp.float32), seconds_mask], axis=1
        ),
    }
    if latent_len is not None:
        assembled["local_add_cond"] = jnp.zeros(
            (prompt_hidden.shape[0], LOCAL_ADD_COND_CHANNELS, latent_len), dtype=jnp.float32
        )
    return assembled