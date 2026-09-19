# hummingbird

A pure-functional **JAX port of Stable Audio 3 inference** — text-to-audio generation
without PyTorch at runtime, written to run inside a single `jax.jit` and to scale onto
TPU.

`sa3jax/` is the whole package. It is inference-only, has no Flax/NNX dependency, keeps
no module state, and reproduces the reference PyTorch sampler's numerical decisions
deliberately rather than approximately.

## What it is

Stable Audio 3 (`stabilityai/stable-audio-3-*`) generates audio by denoising a latent
with a diffusion transformer (DiT) and decoding that latent with a SAME-family
autoencoder. `sa3jax` reimplements each stage as a function over an explicit parameter
pytree:

```
prompt ──► T5Gemma encoder ──► conditioning ──┐
                                              ▼
              initial latent ──► pingpong denoiser (DiT) ──► latents ──► SAME decoder ──► audio
```

Every entry point takes `params` as an **argument**. Closing over weights lets XLA
constant-fold the whole model: that was measured at 754 s of compile time against 3.4 s
when `params` is an argument, so it is a hard constraint rather than a style preference.

## Layout

| module | role |
| --- | --- |
| `config.py` | frozen dataclass configs, transcribed from each checkpoint's `model_config.json`; `SMALL_MUSIC` and `MEDIUM` presets |
| `dtypes.py` | per-role precision map (`BF16`, `FP16`, `FP32_PARITY`) — compute vs the fp32 "islands" |
| `weights.py` | safetensors → JAX pytrees; asserts the checkpoint key set equals `consumed ∪ dropped` |
| `text_encoder.py` | T5Gemma encoder (`google/t5gemma-b-b-ul2`) + tokenizer |
| `conditioner.py` | number conditioner (expo Fourier features), padding, conditioning assembly |
| `dit.py` | the diffusion transformer; `prepare_static` + `dit_forward` split so per-step work is hoisted |
| `sampling.py` | LogSNR-shifted timestep schedule + the pingpong denoiser sampler |
| `same.py` | SAME-S decoder (`small-music`), unchunked |
| `same_l.py` | SAME-L decoder (`medium`) — sliding-window banded attention |
| `pipeline.py` | end-to-end `generate()` / `render()` / `postprocess()` |
| `emit.py` | parity emitter — records activations in the reference fixture's key layout |

## Design decisions

These are load-bearing; the module docstrings carry the full rationale and the upstream
line references.

**Purity over module state.** No Flax, no NNX, no hidden buffers. Noise is never drawn
inside the model: the initial latent comes from a `seed`, but the per-step renoise and
both decoder noises are explicit arguments. Torch's Philox stream and JAX's Threefry are
different generators, so consuming the reference's verbatim draws is the only way to
compare against it.

**One `jit` boundary.** The denoising loop runs under `jax.lax.scan` with fixed shapes
and no branching on traced values. `T_lat` is static by construction (it is an array
shape), so the executable cache holds one entry per `(steps, T_lat, cfg, dtype)` — bucket
durations rather than recompiling per second.

**fp32 islands are measured, not stylistic.** bf16 RoPE destroys 9/16 frequency pairs at
long sequence lengths; the SAME decoders' differential attention cancels catastrophically
when its subtraction is done in reduced precision; dropping the fp32 QK^T/PV accumulation
costs the bf16 tier 36 dB. The fp32 tier is bit-identical with and without them.

**The reference's own quirks are reproduced.** The timestep schedule is per-element (the
shift receives `effective_seq_len` as a tensor, so the per-element branch is taken even at
batch size 1), `t` is cast to the latent dtype *before* the DiT sees it, and the noise
draw on the final step is consumed even though `t_next = 0` multiplies it away.

**Explicitly out of scope:** CFG and APG (`cfg_scale != 1.0` raises rather than rendering
silently-wrong audio), the inpaint path, init-audio/variation, LoRA, `euler`/`rk4`/`dpmpp`,
chunked decode, and `soft_clip`.

## Install

Requires Python 3.11+ and a JAX install for your accelerator — CPU works for reading the
code, but generation wants a GPU or TPU.

```bash
uv venv
uv pip install -e .
```

For CPU-only work, install `jax` first (`uv pip install jax`); the extras in
`pyproject.toml` pull the CUDA build.

## Weights

Weights are **not** in this repo. Fetch the gated checkpoints from Hugging Face into a
local directory laid out as:

```
<root>/
  model.safetensors      # DiT + decoder + conditioner
  t5gemma.safetensors    # text encoder
  t5gemma-b-b-ul2/
    tokenizer.json
```

Both upstream repos are gated and need an authenticated token. `weights.load_all` takes
the two safetensors paths directly, so any layout works as long as you pass the paths —
`pipeline.ROOT` is only the default.

| preset | checkpoint | DiT | decoder |
| --- | --- | ---: | --- |
| `SMALL_MUSIC` | `stabilityai/stable-audio-3-small-music` | 433 M | SAME-Small |
| `MEDIUM` | `stabilityai/stable-audio-3-medium` | 1.4 B | SAME-Large |

## Usage

```python
from sa3jax.config import SMALL_MUSIC
from sa3jax.dtypes import BF16
from sa3jax.pipeline import generate

audio = generate("a warm lo-fi hip hop loop", seconds=10.0, steps=8, dtype=BF16)
# audio: (1, 2, 441000) fp32 in [-1, 1]
```

`generate` is host-side: it tokenises, runs the jitted conditioner, renders, then masks,
clamps and truncates. The three stages are also usable separately — `pipeline.render`
returns `(latents, raw_audio)` and `pipeline.postprocess` does the output handling — which
is how the parity harness injects the reference fixture's own conditioning and noise
instead of a tokenizer round trip.

`pipeline.compile_count()` reports the two executable-cache sizes, so cache growth is
observable rather than inferred.

Parameter loading can be checked standalone:

```bash
python -m sa3jax.weights <root>    # per-group parameter counts + storage dtypes
```

## Precision tiers

`dtypes.Dtype` names a role for every tensor rather than one global precision:

- `BF16` — bf16 compute with fp32 islands. The TPU target, and the default.
- `FP16` — fp16 compute; the cheapest tier, and the one where the fp32 islands matter most.
- `FP32_PARITY` — the same code path in fp32, so the port can be validated against an fp32
  reference fixture without maintaining a second implementation.

## Status

Working and validated against reference fixtures for `small-music`. `MEDIUM` (SAME-L) is
ported and numerically checked. The package is the JAX/TPU arm of a wider Stable Audio 3
inference-optimization effort; the Triton/GPU side and the measurement harnesses live
outside this repo.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
