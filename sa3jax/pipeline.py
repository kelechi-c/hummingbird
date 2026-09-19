"""End-to-end text-to-audio: prompt -> `(1, 2, seconds*44100)` — `StableAudioModel.generate`.

Ground truth: `model.py:76-367` (`generate`), `:400-429` (`_adapt_sample_size`),
`inference/sampling.py:445-452` (padding mask), `:509-521` (decode + mask zeroing) and
`model.py:351-365` (clamp + truncate).

    generate()      host-side: tokenise -> `_condition` (jit) -> `render` -> `postprocess`
    render()        `prepare_static` (eager) then `_render` (jit); `(latents, raw audio)` out
    postprocess()   mask zeroing -> fp32 -> clamp -> truncate

Both jits take `params`/`static` as *arguments*. Closing over them let XLA constant-fold all
459 M parameters: 754 s of compile against 3.4 s (measured, M3) — a hard constraint, not style.
`T_lat` is static by construction (it is an array shape), so the executable cache holds exactly
one entry per `(steps, T_lat, cfg, dtype)`; bucket durations rather than recompiling per second.

Nothing here draws RNG implicitly: `seed` feeds `jax.random.key` for the initial latent only, the
per-step renoise and both decoder noises are explicit arguments (D2) defaulting to zeros.

Out of scope, one line each: CFG/APG (`cfg_scale != 1.0` raises, D3), inpaint, init-audio /
variation, LoRA, `euler`/`rk4`/`dpmpp`, chunked decode, `soft_clip`, `prepend_cond`.
"""
from __future__ import annotations

import logging
import math
import pathlib as pl
from functools import partial
from typing import Final, Literal

import jax
import jax.numpy as jnp
from jax import Array

from sa3jax import conditioner, dit, same, sampling, text_encoder, weights
from sa3jax.config import ConditionerConfig, ModelConfig, SMALL_MUSIC
from sa3jax.dtypes import BF16, Dtype

logger = logging.getLogger(__name__)

# Staged weights + tokenizer on this host; `weights.load_all` takes the two checkpoints.
ROOT: Final[pl.Path] = pl.Path("/home/tensor/.cache/hummingbird/small-music")
TOKENIZER_REL: Final[str] = "t5gemma-b-b-ul2/tokenizer.json"
# `model.sampling_dist_shift` (`diffusion.py:79`): LogSNRShift(anchor_logsnr=-6.2, logsnr_end=2.0).
ANCHOR_LOGSNR: Final[float] = -6.2
LOGSNR_END: Final[float] = 2.0

Params = dict[str, dict]
Cond = dict[str, Array]
Noise = Literal["zeros"] | Array


def latent_len(seconds: float, cfg: ModelConfig = SMALL_MUSIC) -> int:
    """`_adapt_sample_size` (`model.py:400-429`) then `// downsampling_ratio`.

    The 6 s headroom exists so the model can render past the requested duration — the tail is
    where the distilled sampler lands, and it is truncated afterwards. 8192 is the chunked
    encoder's alignment (`downsampling_ratio * chunk_size // stride`), and `sample_size` clamps
    120 s, which would otherwise want 1358 latents, down to 1292.
    """
    samples = int((seconds + cfg.audio.headroom_seconds) * cfg.audio.sample_rate)
    aligned = -(-samples // cfg.audio.align) * cfg.audio.align  # round up to 8192
    return min(aligned, cfg.audio.sample_size) // cfg.audio.downsampling_ratio


def padding_mask(seconds: float, cfg: ModelConfig = SMALL_MUSIC) -> Array:
    """`sampling.py:445-452` + `data/utils.py:195-212`: `(1, T_lat)` bool, a strict prefix.

    `valid = ceil(seconds*44100/4096) + int(6*44100/4096)`, clamped to `T_lat`. It is a no-op at
    the 120 s default (1292/1292) and bites at short durations: 118/120 at 5 s, 387/388 at 30 s.
    """
    t_lat = latent_len(seconds, cfg)
    rate, ds = cfg.audio.sample_rate, cfg.audio.downsampling_ratio
    headroom = int(cfg.audio.headroom_seconds * rate / ds)
    valid = math.ceil(int(seconds * rate) / ds) + headroom
    return (jnp.arange(t_lat) < min(valid, t_lat))[None]


@partial(jax.jit, static_argnums=(5, 6, 7))
def _condition(
    text_params: Params,
    ids: Array,
    mask: Array,
    seconds: Array,
    cond_params: Params,
    cfg: ConditionerConfig,
    dtype: Dtype,
    latent_len: int,
) -> Cond:
    """T5Gemma -> `assemble_conditioning`, jitted. `latent_len` is static so `local_add_cond` has
    the reference's `(1, 257, T)` shape and the DiT's zero-guard sees a concrete array."""
    hidden = text_encoder.encode(text_params, ids, mask, cfg, dtype)
    return conditioner.assemble_conditioning(hidden, mask, seconds, cond_params, cfg, dtype, latent_len)


@partial(jax.jit, static_argnums=(7, 8))
def _render(
    dit_params: Params,
    decoder_params: Params,
    static: dit.Static,
    x0: Array,
    noise: Array,
    dither: Array,
    mask_noise: Array,
    cfg: ModelConfig,
    dtype: Dtype,
) -> tuple[Array, Array]:
    """Denoise then decode, one compile per `(steps, T_lat, cfg, dtype)`.

    `noise` is `(steps, 1, 256, T_lat)`, so `steps` comes from its leading axis and stays Python
    at trace time. `dither` is `(1, 256, T_lat)` and `mask_noise` `(T_lat, 16, 768)`; both are
    injected because torch's Philox stream cannot be reproduced across frameworks (`[R]`).
    """
    schedule = sampling.build_schedule(noise.shape[0], cfg.sigma_max, ANCHOR_LOGSNR, LOGSNR_END)
    latents, _ = sampling.pingpong(dit_params, x0, schedule, noise, static, cfg, dtype, trajectory=False)
    return latents, same.decode(decoder_params, latents, dither, mask_noise, cfg.decoder, dtype)


def render(
    params: Params,
    cond: Cond,
    mask: Array,
    x0: Array,
    noise: Array,
    dither: Array,
    mask_noise: Array,
    cfg: ModelConfig = SMALL_MUSIC,
    dtype: Dtype = BF16,
) -> tuple[Array, Array]:
    """The pure core: conditioning and every stochastic draw in, `(latents, raw audio)` out.

    `prepare_static` stays eager here, between the two jits: it hoists the DiT's x/t-invariant
    work, and it is the one place that must see a *concrete* `local_add_cond` to reject inpaint.
    """
    static = dit.prepare_static(params["dit"], cond, mask, cfg.dit, dtype)
    return _render(params["dit"], params["decoder"], static, x0, noise, dither, mask_noise, cfg, dtype)


def postprocess(
    audio: Array, mask: Array, seconds: float, cfg: ModelConfig = SMALL_MUSIC, dtype: Dtype = BF16
) -> Array:
    """`sampling.py:509-521` then `model.py:351-365`: mask zeroing -> fp32 -> clamp -> truncate.

    The mask is upsampled 4096x on the audio axis, so latent padding positions decode to garbage
    and are silenced. Invisible at 30 s (the zeroed window starts at sample 1 585 152, past the
    1 323 000-sample cut) and audible in the `_full` fixture group, whose tail is exactly zero.
    """
    audio = audio * jnp.repeat(mask, cfg.audio.downsampling_ratio, axis=-1)[:, None, :].astype(audio.dtype)
    return jnp.clip(audio.astype(dtype.audio), -1.0, 1.0)[:, :, : int(seconds * cfg.audio.sample_rate)]


def compile_count() -> tuple[int, int]:
    """Executable-cache sizes `(conditioning, render)`. The render one is one entry per
    `(steps, T_lat, cfg, dtype)` — the contract the sampler's static shapes impose."""
    return _condition._cache_size(), _render._cache_size()


def generate(
    prompt: str,
    seconds: float,
    *,
    steps: int = 8,
    seed: int = 0,
    noise: Noise = "zeros",
    dtype: Dtype = BF16,
    cfg: ModelConfig = SMALL_MUSIC,
    cfg_scale: float = 1.0,
    root: str | pl.Path | None = None,
    params: Params | None = None,
    conditioning: Cond | None = None,
    x0: Array | None = None,
    dither: Array | None = None,
    mask_noise: Array | None = None,
) -> Array:
    """`generate` for text-to-audio: `(1, 2, int(seconds*44100))` fp32 in `[-1, 1]`.

    `noise` is the per-step renoise: `"zeros"` (deterministic, what the ports ship) or the
    reference's verbatim draws `(steps, 1, 256, T_lat)` for parity. `x0`, `dither` and
    `mask_noise` are the remaining draws; `x0` defaults to the seeded initial latent, the other
    two to zeros. `conditioning`/`params` bypass the text path and the loader, so the parity
    harness can inject the fixture's own `cond/*` and noise without a tokenizer round trip.
    """
    if cfg_scale != 1.0:
        raise NotImplementedError(f"cfg_scale={cfg_scale}: CFG/APG is out of scope (D3); rendering unguided audio would be silently wrong")
    t_lat = latent_len(seconds, cfg)
    mask = padding_mask(seconds, cfg)

    if isinstance(noise, str):
        if noise != "zeros":
            raise ValueError(f"noise must be 'zeros' or an array, got {noise!r}")
        renoise = jnp.zeros((steps, 1, cfg.dit.io_channels, t_lat), dtype=dtype.compute)
    else:
        renoise = jnp.asarray(noise)
        if renoise.shape[0] != steps:
            raise ValueError(f"noise has {renoise.shape[0]} steps, generate was asked for {steps}")
    if x0 is None:
        x0 = jax.random.normal(jax.random.key(seed), (1, cfg.dit.io_channels, t_lat), dtype=dtype.compute)
    if dither is None:
        dither = jnp.zeros_like(x0)
    if mask_noise is None:
        mask_noise = jnp.zeros((t_lat, cfg.decoder.new_tokens_per_latent, cfg.decoder.dim), dtype=dtype.compute)

    root = pl.Path(root) if root is not None else ROOT
    if params is None:
        params = weights.load_all(root / "model.safetensors", root / "t5gemma.safetensors", cfg, dtype)
    if conditioning is None:
        ids, token_mask = text_encoder.tokenize(prompt, root / TOKENIZER_REL, cfg.conditioner)
        conditioning = _condition(
            params["text"],
            ids,
            token_mask,
            jnp.asarray([seconds], jnp.float32),
            params["conditioner"],
            cfg.conditioner,
            dtype,
            t_lat,
        )

    _, audio = render(params, conditioning, mask, x0, renoise, dither, mask_noise, cfg, dtype)
    logger.info(
        "generated %.3g s: %d latents (%d valid), %d steps, %s", seconds, t_lat, int(mask.sum()), steps, dtype.name
    )
    return postprocess(audio, mask, seconds, cfg, dtype)
