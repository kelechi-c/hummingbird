# hummingbird

**Stable Audio 3 inference, in JAX**

## model info

Stable Audio 3 (`stabilityai/stable-audio-3-*`) generates audio by denoising a latent
with a diffusion transformer (DiT) and decoding that latent with a SAME-family
autoencoder. `sa3jax` reimplements each stage as a function over an explicit parameter
pytree:

```
prompt ──► T5Gemma encoder ──► conditioning ──┐
                                              ▼
              initial latent ──► pingpong denoiser (DiT) ──► latents ──► SAME decoder ──► audio
```

## usage

```python
from sa3jax.config import SMALL_MUSIC
from sa3jax.dtypes import BF16
from sa3jax.pipeline import generate

audio = generate("your best impression of Mozart's Lacrimosa", seconds=10.0, steps=8, dtype=BF16)
# audio: (1, 2, 441000) fp32 in [-1, 1]
```

`generate` is host-side: it tokenises, runs the jitted conditioner, renders, then masks,
clamps and truncates. The three stages are also usable separately — `pipeline.render`
returns `(latents, raw_audio)` and `pipeline.postprocess` does the output handling — which
is how the parity harness injects the reference fixture's own conditioning and noise
instead of a tokenizer round trip.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
