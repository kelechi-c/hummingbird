"""Opt-in int8 W8A8 GEMM tier for the linear layers.

v6e's MXU runs int8 at 2x its bf16 rate, and the bf16 GEMMs are already at the chip's measured
ceiling (`docs/optimization-log.md`, `docs/roofline-ops.md` §5), so a lower-precision multiply is
the only lever left on them. This module supplies that tier without touching the arithmetic of any
other path: a `_linear` weight leaf is either a plain array (the bf16 path, unchanged bit for bit)
or the `{"q", "scale"}` pair built here, and `_linear` dispatches on which one it received.

The scheme is symmetric W8A8 — **per-output-channel** weight scales (absmax over the input axis /
127) computed once at load by `quantize_params`, and **dynamic per-token** activation scales (absmax
over the contraction axis / 127) computed per call. The int8 products accumulate in int32, which is
exact for any K here, so the tier's entire error is two roundings: one per weight element and one
per activation element. No SmoothQuant calibration: flicker's cost ~2x its LPIPS floor for 16 % of
its DiT speed, and this tier is deliberately simpler until it is measured on the TPU.
"""
from __future__ import annotations

from typing import Any, Final

import jax.numpy as jnp
from jax import Array, lax

from sa3jax.config import ModelConfig

Params = dict[str, Any]
QuantLeaf = dict[str, Array]  # {"q": int8 [out, in], "scale": fp32 [out]}

# symmetric int8: 127 is the positive ceiling, -127 the negative one, so the round trip is unbiased.
_INT8_MAX: Final[float] = 127.0
# a zero weight row (or a zero activation row) would divide by zero; the floor only fires there,
# where the quantised value is 0 whatever the scale.
_SCALE_FLOOR: Final[float] = 1e-6

# The quantised GEMM set, written down in exactly one place. Templates are `weights.py` pytree
# paths with `{i}` the block index. DiT: the big per-token projections only — `cross_attn.to_kv`
# is hoisted into `prepare_static` (once per clip, not per step), and the adaLN/embedder/conv
# linears are small or memory-bound, so neither is in the tier. Decoder: the four per-block GEMMs,
# not `project_in`/`mapping` (head and tail, once per clip).
_DIT_TEMPLATES: Final[tuple[tuple[str, ...], ...]] = (
    ("dit", "transformer", "blocks", "{i}", "self_attn", "to_qkv", "weight"),
    ("dit", "transformer", "blocks", "{i}", "self_attn", "to_out", "weight"),
    ("dit", "transformer", "blocks", "{i}", "cross_attn", "to_q", "weight"),
    ("dit", "transformer", "blocks", "{i}", "cross_attn", "to_out", "weight"),
    ("dit", "transformer", "blocks", "{i}", "ff", "glu_proj", "weight"),
    ("dit", "transformer", "blocks", "{i}", "ff", "proj_out", "weight"),
)
_DECODER_TEMPLATES: Final[tuple[tuple[str, ...], ...]] = (
    ("decoder", "blocks", "{i}", "self_attn", "to_qkv", "weight"),
    ("decoder", "blocks", "{i}", "self_attn", "to_out", "weight"),
    ("decoder", "blocks", "{i}", "ff", "glu_proj", "weight"),
    ("decoder", "blocks", "{i}", "ff", "proj_out", "weight"),
)


def quantize_weight(w: Array) -> QuantLeaf:
    """Symmetric per-output-channel int8 weight: `scale = absmax(in) / 127`, `q = round(w / scale)`.

    `w` is `[out, in]` (`nn.Linear`'s layout), so the absmax is over the input axis and `scale` is
    `(out,)`. The clip cannot bite on a leaf quantised against its own absmax; it is there so a
    rounded value can never wrap sign. Pure; `w` is untouched.
    """
    wide = w.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(wide), axis=-1), _SCALE_FLOOR) / _INT8_MAX
    q = jnp.clip(jnp.round(wide / scale[..., None]), -_INT8_MAX, _INT8_MAX).astype(jnp.int8)
    return {"q": q, "scale": scale}


def quantize_rows(x: Array) -> tuple[Array, Array]:
    """Dynamic per-token (per-row) activation quant: absmax over the contraction axis / 127.

    Per-token, not per-tensor: one outlier token would otherwise cost every other token its int8
    range. The reduction is over the GEMM's contraction axis, so it is one elementwise pass. The
    returned scale is `(..., 1)`, broadcastable against the GEMM output.
    """
    wide = x.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(wide), axis=-1, keepdims=True), _SCALE_FLOOR) / _INT8_MAX
    return jnp.clip(jnp.round(wide / scale), -_INT8_MAX, _INT8_MAX).astype(jnp.int8), scale


def linear_int8(x: Array, w: QuantLeaf, bias: Array | None = None) -> Array:
    """W8A8 `nn.Linear`: dynamic per-token activation quant, int8 x int8 -> int32, fp32 dequant.

    `preferred_element_type=int32` is the exact accumulate; the dequant `acc * x_scale * w_scale`
    is fp32 and casts once to `x.dtype`, so the result is a drop-in `_linear`. `bias` is added
    after the cast. `x` is `(..., in)`, `w["q"]` `[out, in]`: x's last axis contracts with w's
    input axis (1).
    """
    qx, x_scale = quantize_rows(x)
    dims = (((x.ndim - 1,), (1,)), ((), ()))
    acc = lax.dot_general(qx, w["q"], dims, preferred_element_type=jnp.int32)
    y = (acc.astype(jnp.float32) * x_scale * w["scale"]).astype(x.dtype)
    return y if bias is None else y + bias


def is_quant(w: Array | QuantLeaf) -> bool:
    """Whether a `_linear` weight leaf is the int8 tier's `{"q", "scale"}` pair (vs a plain array)."""
    return isinstance(w, dict)


def _expand(templates: tuple[tuple[str, ...], ...], blocks: int) -> set[tuple[str | int, ...]]:
    """Every concrete leaf path of a template set, `{i}` expanded over the block count."""
    return {
        tuple(i if seg == "{i}" else seg for seg in template)
        for template in templates
        for i in (range(blocks) if "{i}" in template else (None,))
    }


def int8_targets(cfg: ModelConfig) -> set[tuple[str | int, ...]]:
    """The tier's concrete leaf paths for this config; empty when both `gemm` fields are `"bf16"`."""
    targets: set[tuple[str | int, ...]] = set()
    if cfg.dit.gemm == "int8":
        targets |= _expand(_DIT_TEMPLATES, cfg.dit.depth)
    if cfg.decoder.gemm == "int8":
        targets |= _expand(_DECODER_TEMPLATES, cfg.decoder.transformer_depth)
    return targets


def _rewrite(node: Any, path: tuple[str | int, ...], targets: set[tuple[str | int, ...]]) -> Any:
    """Rebuild `node` with every leaf whose path is in `targets` replaced by its quantised pair."""
    if isinstance(node, dict):
        return {key: _rewrite(value, path + (key,), targets) for key, value in node.items()}
    if isinstance(node, tuple):
        return tuple(_rewrite(value, path + (index,), targets) for index, value in enumerate(node))
    return quantize_weight(node) if path in targets else node


def quantize_params(params: Params, cfg: ModelConfig) -> Params:
    """Quantise the tier's weight leaves in a `weights.load_all` tree; every other leaf is returned
    unchanged.

    Pure and once-per-load: the selected leaves become `{"q": int8, "scale": fp32}`, a pytree the
    sampler still takes as a jit **argument**. Never call this inside the per-step loop — only the
    activations quantise there, dynamically, inside `linear_int8`. With both `gemm` fields at their
    `"bf16"` default the target set is empty and the tree comes back bit-for-bit identical.
    """
    return _rewrite(params, (), int8_targets(cfg))
