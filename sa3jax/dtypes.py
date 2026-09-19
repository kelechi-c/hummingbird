"""Per-role precision map.

Each field is the dtype for one *role*, not one tensor: `compute` is the bulk matmul/attention
precision, `norm`/`rope`/`t`/`audio` are fp32 islands kept at full precision for reasons the
module docstring below records, `weight` is storage, and `text` is T5Gemma's own dtype (bf16 in
the checkpoint, and `model_half` never touches it).

Default is bf16 compute with fp32 islands, matching the TPU target. `FP32_PARITY` runs the
same code path in fp32 so the port can be checked against the fp32 reference fixture without
a second implementation.

The islands are not stylistic. `[R]` from the TensorRT port notes: bf16 RoPE destroys 9/16
frequency pairs at long sequence lengths, and the SAME decoder's differential attention
cancels catastrophically when its subtraction is done in reduced precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Dtype:
    """Storage and compute dtypes for each part of the pipeline."""

    weight: jnp.dtype = jnp.bfloat16
    compute: jnp.dtype = jnp.bfloat16
    norm: jnp.dtype = jnp.float32
    rope: jnp.dtype = jnp.float32
    t: jnp.dtype = jnp.float32
    audio: jnp.dtype = jnp.float32
    text: jnp.dtype = jnp.bfloat16

    @property
    def name(self) -> str:
        return jnp.dtype(self.compute).name


BF16 = Dtype()
FP32_PARITY = Dtype(
    weight=jnp.float32,
    compute=jnp.float32,
    norm=jnp.float32,
    rope=jnp.float32,
    t=jnp.float32,
    audio=jnp.float32,
    text=jnp.float32,
)
FP16 = Dtype(
    weight=jnp.float16,
    compute=jnp.float16,
    norm=jnp.float32,
    rope=jnp.float32,
    t=jnp.float32,
    audio=jnp.float32,
    text=jnp.float16,
)
