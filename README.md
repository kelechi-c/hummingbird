# hummingbird

**Stable Audio 3 inference in JAX, with fused Pallas kernels for TPU v6e**

A 2-minute Stable Audio 3 **medium** clip renders in **163 ms** on one TPU v6e chip (batch 1, 8 steps) — 2.46× faster
than the straight port, and faster than Stability's own TensorRT fp8 engine on an RTX PRO 6000 (205 ms). A full 380 s
song takes 674 ms. Every optimisation is quality-gated against the official PyTorch output.

| clip length | 5 s | 30 s | 120 s | 380 s |
|---|---|---|---|---|
| sa3jax, TPU v6e-1 | 27.8 ms | 53.0 ms | **163.4 ms** | **673.5 ms** |
| baseline port, TPU v6e-1 | 43.7 ms | 101.5 ms | 401.4 ms | 1766 ms |
| TensorRT fp8, RTX PRO 6000 | 49.1 ms | 80.6 ms | 205.2 ms | 692.5 ms |
| official PyTorch, RTX PRO 6000 | 301.4 ms | 327.2 ms | 474.7 ms | 1305 ms |

Warm device latency, bf16, one clip at a time; TensorRT renders ~5 % fewer latents per clip.

## model

Stable Audio 3 generates audio by denoising a latent with a diffusion transformer (DiT) and decoding it with a
SAME-family autoencoder. `sa3jax` reimplements each stage as a pure function over an explicit parameter pytree:

```
prompt ──► T5Gemma encoder ──► conditioning ──┐
                                              ▼
              initial latent ──► pingpong denoiser (DiT) ──► latents ──► SAME decoder ──► audio
```

## what makes it fast (TPU)

- `sa3jax/band_attention.py` — one Pallas kernel for the SAME-L decoder's attention: DyT → RoPE → both branches of the
  differential sliding-window attention → subtract, in on-chip memory. Replaces ~190 ms of XLA data movement per
  2-minute clip; decode 230 → 61 ms.
- `sa3jax/dense_attention.py` — one Pallas kernel for the DiT's self-attention: RMSNorm → RoPE → value mask → both
  softmaxes → subtract, scores never leaving VMEM; keys and values prepared once per head pair. DiT 160 → 100 ms.
- `sa3jax/dit.py` — cross-attention keys/values computed once per clip instead of once per block per step.

Both kernels engage only on TPU (`SameConfig.attention`, `DiTConfig.attention`); other backends use the XLA path.
They need a libtpu that matches jax (jax 0.7.2 → `libtpu==0.0.23`).

## usage

```python
from sa3jax.config import MEDIUM
from sa3jax.dtypes import BF16
from sa3jax.pipeline import generate

audio = generate("slow atmospheric film score, low strings, organ swells", seconds=120.0, steps=8,
                 dtype=BF16, cfg=MEDIUM, root="/path/to/stable-audio-3-medium")
# audio: (1, 2, 5292000) fp32 in [-1, 1]
```

`root` holds the checkpoint in the layout `weights.load_all` reads (`model.safetensors`, `t5gemma.safetensors`,
`t5gemma-b-b-ul2/tokenizer.json`). The per-step re-noise and the decoder's noise default to zeros; pass `noise=`,
`x0=`, `dither=` and `mask_noise=` (standard normals) to sample like the reference. `pipeline.render` and
`pipeline.postprocess` expose the stages separately.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Stable Audio 3 weights are under Stability AI's own license.
