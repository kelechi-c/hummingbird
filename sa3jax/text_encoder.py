"""T5Gemma text encoder (`google/t5gemma-b-b-ul2`) — the prompt conditioner of SA3.

Pure functions over an explicit parameter pytree: no Flax, no module state, no RNG, no side
effects, so the whole encoder traces under `jax.jit`. Ground truth is
`transformers/models/t5gemma/modeling_t5gemma.py` (local clone v5.13.0.dev0; the fixture itself
was captured on victoria with the installed transformers 5.17.0, whose t5gemma encoder is
functionally identical); `conditioners.py:157-269` shows how SA3 calls it.

Structure: `embed x sqrt(768)` -> 12 x (pre-norm self-attn -> post-norm -> residual, pre-norm
GeGLU -> post-norm -> residual) -> final RMSNorm. Gemma2 conventions throughout: RMSNorm is
`(1 + weight)` with the raw vector stored (not a scale), computed in fp32; GeGLU uses
`gelu_pytorch_tanh` (`jax.nn.gelu(approximate=True)`); attention logits are scaled by
`query_pre_attn_scalar ** -0.5` and **not** soft-capped (see `_attention`); no biases anywhere.

Sliding window: `layer_types` alternates `sliding_attention`/`full_attention` with
`sliding_window=4096`, but the conditioner runs at `max_length=256`, so a sliding band is
`kv_idx > q_idx - 4096` over a 256-wide axis: always true, i.e. inert. One full-attention path
is therefore exact here. Per-layer masks would be ANDed into `key_bias` in `encode` (a stacked
`(layers, S, S)` band), which is why the bias is built there rather than inside `_attention`.

Parameter pytree (weights stacked over layers, leading axis `t5_layers`; names are the
checkpoint's with `model.encoder.` and `.weight` dropped):

    {
      "embed_tokens": (vocab, hidden),
      "norm": (hidden,),
      "layers": {
        "pre_self_attn_norm": (L, hidden),
        "self_attn": {"q_proj": (L, hidden, hidden), "k_proj": ..., "v_proj": ..., "o_proj": ...},
        "post_self_attn_norm": (L, hidden),
        "pre_feedforward_norm": (L, hidden),
        "mlp": {"gate_proj": (L, intermediate, hidden), "up_proj": ..., "down_proj": (L, hidden, intermediate)},
        "post_feedforward_norm": (L, hidden),
      },
    }
"""
from __future__ import annotations

import pathlib as pl

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from tokenizers import Tokenizer

from sa3jax.config import ConditionerConfig
from sa3jax.dtypes import Dtype

def _rms_norm(x: Array, weight: Array, eps: float, dtype: Dtype) -> Array:
    """Gemma RMSNorm: statistics in fp32, raw `(1 + weight)` scale, back to compute dtype."""
    h = x.astype(dtype.norm)
    h = h * jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps)
    return (h * (1.0 + weight.astype(dtype.norm))).astype(dtype.text)


def _linear(x: Array, weight: Array, dtype: Dtype) -> Array:
    """`x @ weight.T` with fp32 accumulation — the semantics of a bf16 cuBLAS GEMM."""
    return jnp.einsum("...i,oi->...o", x, weight, preferred_element_type=dtype.norm).astype(dtype.text)


def _rope_tables(seq_len: int, head_dim: int, theta: float, dtype: Dtype) -> tuple[Array, Array]:
    """Half-half RoPE tables. Frequencies are built in fp32; the reference casts cos/sin down to
    the compute dtype before applying them (`modeling_t5gemma.py:162`), so we do the same."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=dtype.rope) / head_dim))
    freqs = jnp.outer(jnp.arange(seq_len, dtype=dtype.rope), inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype.text), jnp.sin(emb).astype(dtype.text)


def _rotate_half(x: Array) -> Array:
    """`[-x2, x1]` — the half-half (GPT-NeoX/Llama) rotation, not the interleaved one."""
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _attention(
    x: Array,
    params: dict[str, Array],
    cos: Array,
    sin: Array,
    key_bias: Array,
    row_has_key: Array,
    cfg: ConditionerConfig,
    dtype: Dtype,
    softcap: float | None,
) -> Array:
    """Self-attention: q/k/v/o projections, RoPE on q and k, key-side padding mask.

    A row with no valid key at all — an empty prompt — would otherwise softmax over all `-inf` and
    return NaN. `key_bias` is pre-neutralised for such rows (built once in `encode`, not per layer)
    and their probabilities are zeroed here, which is what torch's `_masked_softmax` does. A large
    finite negative is *not* a substitute for `-inf`: it yields a uniform distribution over the pad
    keys rather than zero.
    """
    batch, seq, _ = x.shape
    heads, head_dim = cfg.t5_heads, cfg.t5_head_dim
    shape = (batch, seq, heads, head_dim)
    q = _linear(x, params["q_proj"], dtype).reshape(shape).transpose(0, 2, 1, 3)
    k = _linear(x, params["k_proj"], dtype).reshape(shape).transpose(0, 2, 1, 3)
    v = _linear(x, params["v_proj"], dtype).reshape(shape).transpose(0, 2, 1, 3)
    # Half-half RoPE over the full head_dim (inv_freq is head_dim/2 wide, `emb` is duplicated to
    # head_dim) — not the DiT's partial rotary. The reference casts cos/sin to the compute dtype
    # before multiplying (`modeling_t5gemma.py:162`); applying them in fp32 is a wash (+0.2 dB on
    # p030, -0.25 dB on cond_expanded), so the faithful bf16 form is kept.
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=dtype.norm)
    # `query_pre_attn_scalar ** -0.5`, not `head_dim ** -0.5`; they coincide at 64 here.
    logits = (logits * cfg.t5_query_pre_attn_scalar**-0.5).astype(dtype.text)
    if softcap is not None:
        logits = jnp.tanh(logits / softcap) * softcap
    probs = jax.nn.softmax((logits + key_bias).astype(dtype.norm), axis=-1).astype(dtype.text)
    probs = jnp.where(row_has_key, probs, 0.0)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v, preferred_element_type=dtype.norm)
    out = out.astype(dtype.text).transpose(0, 2, 1, 3).reshape(batch, seq, heads * head_dim)
    return _linear(out, params["o_proj"], dtype)


def _mlp(x: Array, params: dict[str, Array], dtype: Dtype) -> Array:
    """GeGLU: `down(gelu_pytorch_tanh(gate(x)) * up(x))`.

    torch's tanh-GELU CUDA kernel computes in `opmath_type` (fp32 for bf16) and rounds once, so
    the activation is evaluated in fp32 and cast back; doing it in bf16 costs ~1.8 dB of parity.
    """
    gate = jax.nn.gelu(_linear(x, params["gate_proj"], dtype).astype(dtype.norm), approximate=True)
    return _linear(gate.astype(dtype.text) * _linear(x, params["up_proj"], dtype), params["down_proj"], dtype)


def _layer(
    x: Array,
    params: dict[str, object],
    cos: Array,
    sin: Array,
    key_bias: Array,
    row_has_key: Array,
    cfg: ConditionerConfig,
    dtype: Dtype,
    softcap: float | None,
) -> Array:
    """One encoder layer: both sub-blocks are pre-norm, post-normed, then residual-added."""
    h = _rms_norm(x, params["pre_self_attn_norm"], cfg.t5_rms_eps, dtype)
    h = _attention(h, params["self_attn"], cos, sin, key_bias, row_has_key, cfg, dtype, softcap)
    h = _rms_norm(h, params["post_self_attn_norm"], cfg.t5_rms_eps, dtype)
    x = x + h
    h = _rms_norm(x, params["pre_feedforward_norm"], cfg.t5_rms_eps, dtype)
    h = _mlp(h, params["mlp"], dtype)
    h = _rms_norm(h, params["post_feedforward_norm"], cfg.t5_rms_eps, dtype)
    return x + h


def encode(
    params: dict[str, object],
    token_ids: Array,
    attention_mask: Array,
    cfg: ConditionerConfig,
    dtype: Dtype,
    softcap: float | None = None,
) -> Array:
    """`(B, S)` token ids -> `(B, S, t5_hidden)` last_hidden_state. Pure, jit-traceable.

    `softcap` defaults to `None` (off) because that is what the reference actually computes:
    the config sets `attn_logit_softcapping=50` and `T5GemmaSelfAttention` forwards it, but the
    resolved backend is `sdpa` and `sdpa_attention_forward` has no `softcap` parameter — it is
    swallowed by `**kwargs`. Only `eager`/`flex_attention` apply `tanh(x/50)*50`. Reproducing
    the fixture on victoria settles it: `sdpa` is bit-identical (maxabs 0.0) to
    `p030_fp32/cond/t5_last_hidden`, `eager` is 34.2 dB / 32.2 dB away. Pass
    `softcap=cfg.t5_softcap` to get the eager semantics.
    """
    x = params["embed_tokens"][token_ids]
    # Reference builds the normaliser in the compute dtype, so bf16 rounds sqrt(768) first.
    x = x * jnp.asarray(np.sqrt(cfg.t5_hidden), dtype=dtype.text)
    cos, sin = _rope_tables(token_ids.shape[1], cfg.t5_head_dim, cfg.t5_rope_theta, dtype)
    # Padding masks keys only (transformers `padding_mask_function` masks the kv axis), so a
    # padded query still attends to the real tokens. A sliding band would be added here.
    # Built once, outside the layer scan: a row with no valid key at all (an empty prompt) gets a
    # zero bias so its softmax stays finite, and is zeroed after it.
    keep = attention_mask.astype(bool)[:, None, None, :]
    row_has_key = jnp.any(keep, axis=-1, keepdims=True)
    key_bias = jnp.where(row_has_key, jnp.where(keep, 0.0, -jnp.inf), 0.0).astype(dtype.text)
    x, _ = jax.lax.scan(
        lambda h, lp: (_layer(h, lp, cos, sin, key_bias, row_has_key, cfg, dtype, softcap), None),
        x,
        params["layers"],
    )
    return _rms_norm(x, params["norm"], cfg.t5_rms_eps, dtype)


def load_tokenizer(tokenizer_path: pl.Path, cfg: ConditionerConfig) -> Tokenizer:
    """Rust tokenizer configured to reproduce `AutoTokenizer` with `truncation=True,
    max_length=cfg.max_length, padding="max_length"` (right padding, pad id 0)."""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=cfg.max_length)
    tokenizer.enable_padding(direction="right", pad_id=0, pad_token="<pad>", length=cfg.max_length)
    return tokenizer


def tokenize(prompt: str, tokenizer_path: pl.Path, cfg: ConditionerConfig) -> tuple[Array, Array]:
    """Prompt text -> `(1, max_length)` int32 ids and bool mask. Host-side: keep it out of jit."""
    encoded = load_tokenizer(tokenizer_path, cfg).encode(prompt)
    return (
        jnp.asarray(encoded.ids, dtype=jnp.int32)[None],
        jnp.asarray(encoded.attention_mask, dtype=bool)[None],
    )
