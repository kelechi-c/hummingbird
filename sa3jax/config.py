"""Static configuration for Stable Audio 3 checkpoints.

Every value is transcribed from a checkpoint's `model_config.json` and is immutable, so the
whole object is hashable and safe to close over inside a `jax.jit`. Nothing here is
runtime-tunable: a different checkpoint means a different preset, not a mutated config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ConditionerConfig:
    """`models/conditioners.py`. Prompt is T5Gemma; seconds is a Fourier number embedder."""

    cond_dim: int = 768
    max_length: int = 256
    t5_hidden: int = 768
    t5_layers: int = 12
    t5_heads: int = 12
    t5_head_dim: int = 64
    t5_intermediate: int = 2048
    t5_vocab: int = 256000
    t5_rms_eps: float = 1e-6
    t5_rope_theta: float = 10000.0
    t5_softcap: float = 50.0
    t5_query_pre_attn_scalar: int = 64
    seconds_min: float = 0.0
    seconds_max: float = 384.0
    fourier_dim: int = 256
    fourier_min_freq: float = 0.5
    fourier_max_freq: float = 10000.0


@dataclass(frozen=True)
class DiTConfig:
    """`models/dit.py` + `models/transformer.py`, `adaLN` global conditioning, no prepend."""

    io_channels: int = 256
    embed_dim: int = 1024
    depth: int = 20
    num_heads: int = 16
    cond_token_dim: int = 768
    global_cond_dim: int = 768
    local_add_cond_dim: int = 257
    num_memory_tokens: int = 64
    ff_mult: float = 4.0
    differential: bool = False  # `medium` sets True: to_qkv widens to 5D and SDPA is differenced
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    fourier_dim: int = 256
    # Self-attention implementation: the XLA `_sdpa` chain (any backend) or, on TPU, the fused Pallas
    # kernel (`dense_attention`: norm -> RoPE -> V-mask -> both differential softmaxes -> subtract in
    # one VMEM pass; scores kept fp32 where XLA rounds them to bf16). "fused" falls back off TPU.
    attention: Literal["xla", "fused"] = "xla"
    # "fused" only engages at or above this many self-attention tokens (64 memory + T latents): below
    # it there is too little work per kernel program to amortise its fixed cost (v6e-1: XLA wins at 184
    # and 452 tokens, fused at 1422). A static shape, so the choice is free at run time.
    fused_min_tokens: int = 1024

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def rope_dim(self) -> int:
        """`RotaryEmbedding(max(dim_heads // 2, 32))` then `cat(freqs, freqs)`: partial rotary."""
        return max(self.head_dim // 2, 32)

    @property
    def ff_inner(self) -> int:
        return int(self.embed_dim * self.ff_mult)

    @property
    def adaln_dim(self) -> int:
        return 6 * self.embed_dim


@dataclass(frozen=True)
class SameConfig:
    """`models/autoencoders.py` `TransformerResamplingBlock`. Covers SAME-S and SAME-L.

    `dim = channels * c_mults[0]`, `ff_inner = dim * ff_mult`.
    `chunk_size`/`chunk_midpoint_shift`/`conv_mapping` drive the no-sliding-window path
    (SAME-S); `sliding_window`/`sinusoidal_blocks` drive the sliding-window path (SAME-L).
    """

    latent_dim: int = 256
    out_channels: int = 512
    channels: int = 128
    c_mults: tuple[int, ...] = (6,)
    stride: int = 16
    transformer_depth: int = 6
    dim_heads: int = 64
    # Both shipped checkpoints set `variable_stride: true` (verified in their model_config.json),
    # which only decides the *stored* shape of `new_tokens` — `(1, 1, dim)` rather than
    # `(1, stride, dim)`. The forward broadcast is identical either way.
    variable_stride: bool = True
    ff_mult: int = 3
    differential: bool = True
    dyt: bool = True
    qk_norm_eps: float = 1e-3
    chunk_size: int = 32
    chunk_midpoint_shift: bool = True
    conv_mapping: bool = True
    sliding_window: tuple[int, int] | None = None
    sinusoidal_blocks: int = 0
    mask_noise: float = 0.01
    mapping_bias: bool = True
    # SAME-L attention implementation: the XLA band (`same_l._sdpa`, any backend) or the fused Pallas
    # TPU kernel (`band_attention`, one HBM pass instead of ~150 ms of window/rope/transpose traffic).
    # "pallas" falls back to XLA off TPU. Needs libtpu matching jax (0.0.23 for jax 0.7.2).
    attention: Literal["xla", "pallas"] = "xla"

    @property
    def dim(self) -> int:
        return self.channels * self.c_mults[0]

    @property
    def num_heads(self) -> int:
        return self.dim // self.dim_heads

    @property
    def ff_inner(self) -> int:
        return self.dim * self.ff_mult

    @property
    def sub_chunk_size(self) -> int:
        """One latent plus `stride` new-token positions."""
        return self.stride + 1

    @property
    def new_tokens_per_latent(self) -> int:
        return self.stride

    @property
    def effective_chunk_size(self) -> int:
        """`chunk_size + chunk_size // stride` tokens, the unit the no-SWA path folds into."""
        return self.chunk_size + self.chunk_size // self.stride


@dataclass(frozen=True)
class AudioConfig:
    """`models/pretransforms.py` patched pretransform + `model.py` output handling."""

    sample_rate: int = 44100
    audio_channels: int = 2
    patch_size: int = 256
    downsampling_ratio: int = 4096
    sample_size: int = 5292032
    headroom_seconds: float = 6.0
    align: int = 8192


@dataclass(frozen=True)
class ModelConfig:
    conditioner: ConditionerConfig = ConditionerConfig()
    dit: DiTConfig = DiTConfig()
    decoder: SameConfig = SameConfig()
    audio: AudioConfig = AudioConfig()
    steps: int = 8
    sigma_max: float = 1.0


SMALL_MUSIC = ModelConfig()

# SAME-L / `medium`. Only the DiT and the decoder differ; the conditioner is byte-identical
# (verified: same sha256 for the bundled t5gemma-b-b-ul2 weights in both checkpoints).
MEDIUM = ModelConfig(
    dit=DiTConfig(embed_dim=1536, depth=24, num_heads=24, differential=True,
                  attention="fused"),  # D4: gate pass, closer to the reference than XLA (fp32 scores)
    decoder=SameConfig(
        channels=256,
        transformer_depth=12,
        sliding_window=(1, 1),
        sinusoidal_blocks=8,
        mask_noise=0.1,
        conv_mapping=False,
        chunk_midpoint_shift=False,
        attention="pallas",  # E1: fused band attention on TPU (gate A pass, 2.58x decode); XLA elsewhere
    ),
    audio=AudioConfig(sample_size=16777216),
)
