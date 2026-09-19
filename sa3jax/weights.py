"""PyTorch safetensors -> JAX parameter pytrees.

One entry point, `load_all`. Every checkpoint key is either consumed into the pytree or named in
the explicit drop list at the top of this file, and the file's key set is asserted *equal* to
`consumed | dropped` — there is no `strict=False` equivalent anywhere here, because a silent skip
is exactly the bug this module exists to prevent.

Conv layout: `lax.conv_general_dilated` with `dimension_numbers=("NWC", "OIW", "NWC")` reads a
kernel as `[out, in, k]`, which is PyTorch's `nn.Conv1d.weight`, so **no permutation**. Measured
against torch 2.13 on victoria with identical inputs and the real checkpoint tensors (fp32, CPU):

    preprocess_conv   [256,256,1]   max|d| 3.43e-05   rel_rms 2.8e-07
    postprocess_conv  [256,256,1]   max|d| 9.54e-06   rel_rms 2.8e-07
    decoder mapping   [512,768,3]   max|d| 1.01e-06   rel_rms 3.3e-07

All three are at the fp32 reduction-order floor: each side sits ~2.5e-07 (rel_rms) from an fp64
numpy reference. The `[out,in,k] -> [out,k,in]` permutation recorded in
`docs/sa3-inference-pytorch-deepseek.md` §12.3 is the **MLX** convention: fed to JAX's `"OIW"` it
raises (`in != k`), and paired with a matching `"OWI"` spec it is bit-identical to the unpermuted
form. Do not permute. Note also that the rhs spec must be `"OIW"` and not `"OIH"`: jax 0.11
requires the spatial character to match the lhs, so `("NWC", "OIH", "NWC")` is a `TypeError`.

Dtypes: storage follows the dtype, because the reference's `model_half` casts the *whole* module
tree (`model.to(torch.float16)`), buffers included. Every `model.safetensors` parameter — the
bottleneck scalars included — takes `dtype.weight`; `dtype.norm` is a *compute* dtype, not a
storage one. The one exception is the RoPE `inv_freq` buffers, which take `dtype.rope` (fp32 in
every preset): they are the documented fp32 RoPE island (`docs/jax-port.md` §5), 16 constants per
block, and rounding them is the long-sequence failure mode. `model.half()` does round this buffer
too, so the fp16/bf16 groups are where that deliberate deviation would show; the fp32 group cannot
discriminate it (its buffer is fp32 there as well).

Casting the bottleneck with the dtype is not cosmetic. `running_std` is a single trained scalar
that decode multiplies the latents by, and `model.half()` rounds it: fp32 0.06568934 ->
fp16 0.065673828125 (2.36e-04 relative) / bf16 0.06591796875 (3.48e-03 relative). Keeping the fp32
checkpoint value would reproduce the checkpoint rather than the reference and would surface as a
small systematic offset in the decoder rungs.

The T5Gemma group follows `dtype.text` (the checkpoint is bf16 and sits outside `model_half`),
with its norm vectors in `dtype.norm`, mirroring `sa3jax.text_encoder`'s own loader.

Weight norm: the SAME decoder's output `WNConv1d` is folded offline (see `_fold_weight_norm`).
"""
from __future__ import annotations

import logging
import pathlib as pl
from typing import Callable, Final, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from safetensors import safe_open

from sa3jax.config import ConditionerConfig, ModelConfig
from sa3jax.dtypes import Dtype

logger = logging.getLogger(__name__)

_Rule = tuple[str, tuple[str | int, ...]]
_Tree = dict[str, object]

# The AE *encoder* is out of scope (decoder-only port): 120 keys under this prefix.
DROPPED_PREFIXES: Final[tuple[str, ...]] = ("pretransform.model.encoder.",)
# Zero-element parameter: `SoftNormBottleneck.decode` never concatenates noise at
# `noise_augment_dim == 0`, so it carries no information. 1 key.
DROPPED_KEYS: Final[dict[str, str]] = {
    "pretransform.model.bottleneck.noise_scaling_factor": "zero-element (noise_augment_dim=0), unread by decode",
}
# The bundled t5gemma file is the full seq2seq checkpoint; SA3 uses T5GemmaEncoderModel only.
_TEXT_PREFIX: Final[str] = "model.encoder."
_TEXT_DROPPED_PREFIX: Final[str] = "model.decoder."
_MAPPING_G: Final[str] = "pretransform.model.decoder.layers.3.mapping.weight_g"
_MAPPING_V: Final[str] = "pretransform.model.decoder.layers.3.mapping.weight_v"

_DIT: Final[str] = "model.model"
_DEC: Final[str] = "pretransform.model.decoder"
_BLK: Final[str] = f"{_DIT}.transformer.layers.{{i}}"
_DBLK: Final[str] = f"{_DEC}.layers.3.transformers.{{i}}"


def _seq(prefix: str, path: tuple[str | int, ...], *, bias: bool) -> tuple[_Rule, ...]:
    """`nn.Sequential(Linear, act, Linear)`: checkpoint `.0` is the input projection and `.2` the
    output one; the activation in between has no parameters."""
    leaves = ("weight", "bias") if bias else ("weight",)
    return tuple(
        (f"{prefix}.{i}.{leaf}", path + (side, leaf)) for i, side in ((0, "in"), (2, "out")) for leaf in leaves
    )


_DIT_RULES: Final[tuple[_Rule, ...]] = (
    (f"{_DIT}.preprocess_conv.weight", ("preprocess_conv", "weight")),
    (f"{_DIT}.postprocess_conv.weight", ("postprocess_conv", "weight")),
    *_seq(f"{_DIT}.to_timestep_embed", ("to_timestep_embed",), bias=True),
    *_seq(f"{_DIT}.to_cond_embed", ("to_cond_embed",), bias=False),
    *_seq(f"{_DIT}.to_global_embed", ("to_global_embed",), bias=False),
    (f"{_DIT}.transformer.project_in.weight", ("transformer", "project_in", "weight")),
    (f"{_DIT}.transformer.project_out.weight", ("transformer", "project_out", "weight")),
    (f"{_DIT}.transformer.memory_tokens", ("transformer", "memory_tokens")),
    (f"{_DIT}.transformer.rotary_pos_emb.inv_freq", ("transformer", "rope_inv_freq")),
    *_seq(f"{_DIT}.transformer.global_cond_embedder", ("transformer", "global_cond_embedder"), bias=True),
    (f"{_BLK}.pre_norm.gamma", ("transformer", "blocks", "{i}", "pre_norm", "weight")),
    (f"{_BLK}.self_attn.to_qkv.weight", ("transformer", "blocks", "{i}", "self_attn", "to_qkv", "weight")),
    (f"{_BLK}.self_attn.to_out.weight", ("transformer", "blocks", "{i}", "self_attn", "to_out", "weight")),
    (f"{_BLK}.self_attn.q_norm.gamma", ("transformer", "blocks", "{i}", "self_attn", "q_norm", "weight")),
    (f"{_BLK}.self_attn.k_norm.gamma", ("transformer", "blocks", "{i}", "self_attn", "k_norm", "weight")),
    (f"{_BLK}.cross_attend_norm.gamma", ("transformer", "blocks", "{i}", "cross_attend_norm", "weight")),
    (f"{_BLK}.cross_attn.to_q.weight", ("transformer", "blocks", "{i}", "cross_attn", "to_q", "weight")),
    (f"{_BLK}.cross_attn.to_kv.weight", ("transformer", "blocks", "{i}", "cross_attn", "to_kv", "weight")),
    (f"{_BLK}.cross_attn.to_out.weight", ("transformer", "blocks", "{i}", "cross_attn", "to_out", "weight")),
    (f"{_BLK}.cross_attn.q_norm.gamma", ("transformer", "blocks", "{i}", "cross_attn", "q_norm", "weight")),
    (f"{_BLK}.cross_attn.k_norm.gamma", ("transformer", "blocks", "{i}", "cross_attn", "k_norm", "weight")),
    (f"{_BLK}.ff_norm.gamma", ("transformer", "blocks", "{i}", "ff_norm", "weight")),
    (f"{_BLK}.ff.ff.0.proj.weight", ("transformer", "blocks", "{i}", "ff", "glu_proj", "weight")),
    (f"{_BLK}.ff.ff.0.proj.bias", ("transformer", "blocks", "{i}", "ff", "glu_proj", "bias")),
    (f"{_BLK}.ff.ff.2.weight", ("transformer", "blocks", "{i}", "ff", "proj_out", "weight")),
    (f"{_BLK}.ff.ff.2.bias", ("transformer", "blocks", "{i}", "ff", "proj_out", "bias")),
    (f"{_BLK}.to_scale_shift_gate", ("transformer", "blocks", "{i}", "to_scale_shift_gate")),
    *_seq(f"{_BLK}.to_local_embed", ("transformer", "blocks", "{i}", "to_local_embed"), bias=True),
)

# `mapping.weight_{g,v}` are absent here: they are folded into `mapping.weight` by
# `_fold_weight_norm` and counted as consumed through `_DECODER_FOLDED`.
_DECODER_RULES: Final[tuple[_Rule, ...]] = (
    (f"{_DEC}.layers.1.weight", ("project_in", "weight")),
    (f"{_DEC}.layers.1.bias", ("project_in", "bias")),
    (f"{_DEC}.layers.3.new_tokens", ("new_tokens",)),
    (f"{_DEC}.layers.3.mapping.bias", ("mapping", "bias")),
    ("pretransform.model.bottleneck.running_std", ("bottleneck", "running_std")),
    ("pretransform.model.bottleneck.scaling_factor", ("bottleneck", "scaling_factor")),
    ("pretransform.model.bottleneck.bias", ("bottleneck", "bias")),
    (f"{_DBLK}.pre_norm.alpha", ("blocks", "{i}", "pre_norm", "alpha")),
    (f"{_DBLK}.pre_norm.gamma", ("blocks", "{i}", "pre_norm", "gamma")),
    (f"{_DBLK}.pre_norm.beta", ("blocks", "{i}", "pre_norm", "beta")),
    (f"{_DBLK}.self_attn.to_qkv.weight", ("blocks", "{i}", "self_attn", "to_qkv", "weight")),
    (f"{_DBLK}.self_attn.to_out.weight", ("blocks", "{i}", "self_attn", "to_out", "weight")),
    (f"{_DBLK}.self_attn.q_norm.alpha", ("blocks", "{i}", "self_attn", "q_norm", "alpha")),
    (f"{_DBLK}.self_attn.q_norm.gamma", ("blocks", "{i}", "self_attn", "q_norm", "gamma")),
    (f"{_DBLK}.self_attn.q_norm.beta", ("blocks", "{i}", "self_attn", "q_norm", "beta")),
    (f"{_DBLK}.self_attn.k_norm.alpha", ("blocks", "{i}", "self_attn", "k_norm", "alpha")),
    (f"{_DBLK}.self_attn.k_norm.gamma", ("blocks", "{i}", "self_attn", "k_norm", "gamma")),
    (f"{_DBLK}.self_attn.k_norm.beta", ("blocks", "{i}", "self_attn", "k_norm", "beta")),
    (f"{_DBLK}.ff_norm.alpha", ("blocks", "{i}", "ff_norm", "alpha")),
    (f"{_DBLK}.ff_norm.gamma", ("blocks", "{i}", "ff_norm", "gamma")),
    (f"{_DBLK}.ff_norm.beta", ("blocks", "{i}", "ff_norm", "beta")),
    (f"{_DBLK}.ff.ff.0.proj.weight", ("blocks", "{i}", "ff", "glu_proj", "weight")),
    (f"{_DBLK}.ff.ff.0.proj.bias", ("blocks", "{i}", "ff", "glu_proj", "bias")),
    (f"{_DBLK}.ff.ff.2.weight", ("blocks", "{i}", "ff", "proj_out", "weight")),
    (f"{_DBLK}.ff.ff.2.bias", ("blocks", "{i}", "ff", "proj_out", "bias")),
    (f"{_DBLK}.rope.inv_freq", ("blocks", "{i}", "rope_inv_freq")),
)
_DECODER_FOLDED: Final[tuple[str, ...]] = (_MAPPING_V, _MAPPING_G)

_CONDITIONER_RULES: Final[tuple[_Rule, ...]] = (
    ("conditioner.conditioners.prompt.padding_embedding", ("padding_embedding",)),
    ("conditioner.conditioners.seconds_total.embedder.embedding.1.weight", ("seconds_weight",)),
    ("conditioner.conditioners.seconds_total.embedder.embedding.1.bias", ("seconds_bias",)),
)

_TEXT_SUFFIXES: Final[tuple[str, ...]] = (
    "pre_self_attn_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_self_attn_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "post_feedforward_layernorm.weight",
)
_TEXT_RULES: Final[tuple[_Rule, ...]] = (
    ("model.encoder.embed_tokens.weight", ("embed_tokens",)),
    ("model.encoder.norm.weight", ("norm",)),
) + tuple(
    (f"model.encoder.layers.{{i}}.{suffix}", ("layers", "{i}", suffix)) for suffix in _TEXT_SUFFIXES
)


def _expand(rules: Sequence[_Rule], blocks: int) -> set[str]:
    """Every checkpoint key a rule table consumes, for the configured block count."""
    return {t.replace("{i}", str(i)) for t, _ in rules for i in (range(blocks) if "{i}" in t else (0,))}


def _put(node: dict | list, path: tuple[str | int, ...], value: Array) -> None:
    """Insert `value` at `path`, creating nested dicts and block lists on the way down."""
    seg, rest = path[0], path[1:]
    if not rest:
        node[seg] = value
        return
    if isinstance(seg, int):
        while len(node) <= seg:
            node.append(None)
    child = node[seg] if isinstance(seg, int) else node.get(seg)
    if child is None:
        child = [] if isinstance(rest[0], int) else {}
        node[seg] = child
    _put(child, rest, value)


def _freeze(node: object) -> object:
    """Lists -> tuples: block stacks are pytrees too, but they are immutable."""
    if isinstance(node, dict):
        return {k: _freeze(v) for k, v in node.items()}
    if isinstance(node, list):
        return tuple(_freeze(v) for v in node)
    return node


def _build(rules: Sequence[_Rule], blocks: int, read: Callable[[str], Array]) -> _Tree:
    """Apply a rule table to the checkpoint: expand `{i}` over the block count, read, nest."""
    tree: dict = {}
    for template, path in rules:
        for i in range(blocks) if "{i}" in template else (None,):
            key = template if i is None else template.replace("{i}", str(i))
            _put(tree, tuple(i if s == "{i}" else s for s in path), read(key))
    return _freeze(tree)  # type: ignore[return-value]


def _cast(key: str, arr: np.ndarray, dtype: Dtype) -> Array:
    """Storage dtype of one checkpoint tensor: the map drives everything, because the reference
    casts the whole module tree — **buffers included**, so the RoPE `inv_freq` buffer is rounded
    too (`model.half()` is `.to()`, which casts parameters *and* buffers). The fp32 island survives
    downstream: `dit._rope_tables` re-casts the buffer for the angle math, so only the rounding of
    the stored frequencies is reproduced, which is what the reference does.
    """
    return jnp.asarray(arr).astype(dtype.weight)


def _reader(get: Callable[[str], np.ndarray], dtype: Dtype) -> Callable[[str], Array]:
    """Bind the precision map to a raw tensor getter: one read path for every rule table."""
    return lambda key: _cast(key, get(key), dtype)


def _fold_weight_norm(v: np.ndarray, g: np.ndarray, dtype: Dtype) -> Array:
    """`torch.nn.utils.weight_norm(dim=0)` -> a plain kernel: `w = v * (g / ||v||)`, the norm taken
    over every axis but 0 (`norm_except_dim`, i.e. `v.reshape(v.shape[0], -1).norm(2, 1)`).

    Folded in fp32 straight from the checkpoint, before any cast. Checked against torch's own
    folded weight on the real 512x768x3 kernel: max|d| 1.49e-08 (1 ulp), and torch's legacy
    `weight_norm` and the modern `parametrizations.weight_norm` are bit-identical to each other.
    """
    norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape((-1,) + (1,) * (v.ndim - 1))
    return jnp.asarray(v * (g / norm)).astype(dtype.weight)


def _assert_keys(
    keys: set[str], expected: set[str], prefixes: Sequence[str], exact: dict[str, str], what: str
) -> set[str]:
    """Exact key-set equality: `file keys == consumed | dropped`. Raises with both sides named."""
    dropped = {k for k in keys if k.startswith(tuple(prefixes))} | {k for k in keys if k in exact}
    if expected & dropped:
        raise ValueError(f"{what}: keys both consumed and dropped: {sorted(expected & dropped)}")
    if keys != expected | dropped:
        raise ValueError(
            f"{what}: missing={sorted(expected - keys)} unexpected={sorted(keys - expected - dropped)}"
        )
    return dropped


def _assert_shapes(dit: _Tree, decoder: _Tree, cfg: ModelConfig) -> None:
    """Directional checks on the tensors whose layout the ports historically got wrong: a
    transposed kernel or a swapped in/out pair fails here rather than at parity time."""
    d, s = cfg.dit, cfg.decoder
    checks = (
        ("dit.preprocess_conv.weight", dit["preprocess_conv"]["weight"], (d.io_channels, d.io_channels, 1)),
        ("dit.postprocess_conv.weight", dit["postprocess_conv"]["weight"], (d.io_channels, d.io_channels, 1)),
        ("dit.transformer.project_in.weight", dit["transformer"]["project_in"]["weight"], (d.embed_dim, d.io_channels)),
        ("dit.transformer.project_out.weight", dit["transformer"]["project_out"]["weight"], (d.io_channels, d.embed_dim)),
        ("dit.transformer.memory_tokens", dit["transformer"]["memory_tokens"], (d.num_memory_tokens, d.embed_dim)),
        ("decoder.project_in.weight", decoder["project_in"]["weight"], (s.dim, s.latent_dim)),
        ("decoder.mapping.weight", decoder["mapping"]["weight"], (s.out_channels, s.dim, 3 if s.conv_mapping else 1)),
    )
    for name, arr, want in checks:
        if tuple(arr.shape) != want:
            raise ValueError(f"{name}: shape {tuple(arr.shape)} != {want}")
    if len(dit["transformer"]["blocks"]) != d.depth or len(decoder["blocks"]) != s.transformer_depth:
        raise ValueError("block count does not match the config")


def load_model(path: str | pl.Path, cfg: ModelConfig, dtype: Dtype) -> dict[str, object]:
    """`model.safetensors` -> `{"dit", "decoder", "conditioner"}` pytrees."""
    with safe_open(str(path), framework="numpy") as f:
        keys = set(f.keys())
        read = _reader(f.get_tensor, dtype)
        dit = _build(_DIT_RULES, cfg.dit.depth, read)
        decoder = _build(_DECODER_RULES, cfg.decoder.transformer_depth, read)
        conditioner = _build(_CONDITIONER_RULES, 0, read)
        expected = (
            _expand(_DIT_RULES, cfg.dit.depth)
            | _expand(_DECODER_RULES, cfg.decoder.transformer_depth)
            | set(_DECODER_FOLDED)
            | _expand(_CONDITIONER_RULES, 0)
        )
        dropped = _assert_keys(keys, expected, DROPPED_PREFIXES, DROPPED_KEYS, "model.safetensors")
        decoder["mapping"]["weight"] = _fold_weight_norm(f.get_tensor(_MAPPING_V), f.get_tensor(_MAPPING_G), dtype)
    logger.info(
        "model.safetensors: %d keys consumed, %d dropped (%d ae encoder, %d dead)",
        len(keys) - len(dropped),
        len(dropped),
        sum(1 for k in dropped if k.startswith(DROPPED_PREFIXES)),
        len(dropped) - sum(1 for k in dropped if k.startswith(DROPPED_PREFIXES)),
    )
    _assert_shapes(dit, decoder, cfg)
    return {"dit": dit, "decoder": decoder, "conditioner": conditioner}


def load_text(path: str | pl.Path, cfg: ConditionerConfig, dtype: Dtype) -> dict[str, object]:
    """`t5gemma.safetensors` -> the encoder pytree `sa3jax.text_encoder.encode` consumes.

    The layers are stacked along a leading axis (the encoder scans over them). This is the single
    loader for that file: the bundled checkpoint is the full seq2seq model, so the 206
    `model.decoder.*` keys are named explicitly as dropped rather than silently ignored.
    """
    with safe_open(str(path), framework="numpy") as f:
        keys = set(f.keys())
        expected = _expand(_TEXT_RULES, cfg.t5_layers)
        dropped = _assert_keys(keys, expected, (_TEXT_DROPPED_PREFIX,), {}, "t5gemma.safetensors")

        def tensor(suffix: str, target: jnp.dtype) -> Array:
            return jnp.asarray(f.get_tensor(f"{_TEXT_PREFIX}{suffix}")).astype(target)

        def stack(suffix: str, target: jnp.dtype) -> Array:
            return jnp.stack([tensor(f"layers.{i}.{suffix}", target) for i in range(cfg.t5_layers)])

        params = {
            "embed_tokens": tensor("embed_tokens.weight", dtype.text),
            "norm": tensor("norm.weight", dtype.norm),
            "layers": {
                "pre_self_attn_norm": stack("pre_self_attn_layernorm.weight", dtype.norm),
                "self_attn": {
                    "q_proj": stack("self_attn.q_proj.weight", dtype.text),
                    "k_proj": stack("self_attn.k_proj.weight", dtype.text),
                    "v_proj": stack("self_attn.v_proj.weight", dtype.text),
                    "o_proj": stack("self_attn.o_proj.weight", dtype.text),
                },
                "post_self_attn_norm": stack("post_self_attn_layernorm.weight", dtype.norm),
                "pre_feedforward_norm": stack("pre_feedforward_layernorm.weight", dtype.norm),
                "mlp": {
                    "gate_proj": stack("mlp.gate_proj.weight", dtype.text),
                    "up_proj": stack("mlp.up_proj.weight", dtype.text),
                    "down_proj": stack("mlp.down_proj.weight", dtype.text),
                },
                "post_feedforward_norm": stack("post_feedforward_layernorm.weight", dtype.norm),
            },
        }
    logger.info("t5gemma.safetensors: %d keys consumed, %d dropped (%s)", len(keys) - len(dropped), len(dropped), _TEXT_DROPPED_PREFIX)
    return params


def load_all(
    model_path: str | pl.Path, text_path: str | pl.Path, cfg: ModelConfig, dtype: Dtype
) -> dict[str, dict]:
    """Both checkpoints -> `{"dit", "decoder", "conditioner", "text"}` pytrees of `jax.Array`."""
    return {
        **load_model(model_path, cfg, dtype),
        "text": load_text(text_path, cfg.conditioner, dtype),
    }


def param_counts(params: dict[str, object]) -> dict[str, int]:
    """Parameter count per top-level group; small-music expects dit 459 130 896 (459.13 M),
    decoder 54 515 321 (54.52 M), conditioner 198 144 (0.20 M), text 281 580 288 (281.58 M).

    The DiT figure is the raw `model.model.*` total, so every key lands exactly once. The decoder
    is not the raw total: two weight-norm keys fold into one (54 515 320 - 1 180 160 + 1 179 648)
    plus 513 bottleneck scalars hoisted in from a sibling module.
    """
    return {k: sum(int(x.size) for x in jax.tree.leaves(v)) for k, v in params.items()}


if __name__ == "__main__":
    import sys

    from sa3jax.config import SMALL_MUSIC
    from sa3jax.dtypes import BF16

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = pl.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/tensor/.cache/hummingbird/small-music")
    loaded = load_all(root / "model.safetensors", root / "t5gemma.safetensors", SMALL_MUSIC, BF16)
    for name, count in param_counts(loaded).items():
        print(f"{name:12s} {count:>12,}")
    print("dit.preprocess_conv.weight", loaded["dit"]["preprocess_conv"]["weight"].dtype)
    print("decoder.bottleneck.running_std", loaded["decoder"]["bottleneck"]["running_std"].dtype)
    print("dit.transformer.rope_inv_freq", loaded["dit"]["transformer"]["rope_inv_freq"].dtype)
