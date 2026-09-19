"""Stable Audio 3 inference in JAX.

Pure functions over explicit parameter pytrees. No hidden state, no Flax: the port is
inference-only and every entry point is designed to sit inside a single `jax.jit`.
"""
from __future__ import annotations

from sa3jax.config import MEDIUM, SMALL_MUSIC, ModelConfig
from sa3jax.dtypes import BF16, FP16, FP32_PARITY, Dtype

__all__ = ["BF16", "FP16", "FP32_PARITY", "MEDIUM", "SMALL_MUSIC", "Dtype", "ModelConfig"]
