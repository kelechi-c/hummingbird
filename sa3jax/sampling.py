"""Timestep schedule + pingpong denoiser sampler — `sample/*` and the DiT trajectory (rung 10).

Ground truth: `inference/sampling.py:9-64` (`build_schedule`), `:308-351`
(`sample_flow_pingpong`), `:433-434` (the sampler choice) and `:455-464` (init-audio);
`inference/distribution_shift.py:134-198` (`LogSNRShift`). Shipped path only:
`rf_denoiser` + `pingpong`, text-to-audio, `cfg_scale=1.0`.

Three reference behaviours are reproduced deliberately rather than simplified away:

1. **The schedule is per-element.** `effective_seq_len` reaches `LogSNRShift.shift` as a
   *tensor*, so `t` is broadcast to `(B, steps+1)` and `sample_flow_pingpong` always takes its
   per-element branch (`distribution_shift.py:186-188`, `sampling.py:325-330`) — at `B=1` too.
   `build_schedule` returns the `(steps+1,)` vector the interface documents; `pingpong`
   broadcasts it to `(B, steps+1)`, so the branch and every downstream shape are the
   reference's. With `rate=0` all rows are equal, which is why the schedule does not depend on
   sequence length (5/30/120 s give identical values) — `sample/effective_seq_len` is evidence
   only, not an input.
2. **`t` is cast to the latent dtype before the DiT sees it** (`sampling.py:326-327`). At fp16
   the schedule becomes `[1.0, 0.994141, ...]`, at bf16 `[1.0, 0.99609375, ...]`; the DiT then
   re-upcasts to fp32 internally, so the rounding is baked in upstream. It is done here, once,
   and never silently dropped.
3. **The noise draw on the last step is consumed.** `torch.randn_like(x)` runs on every step
   including the one where `t_next = 0` multiplies it away. `noise` is injected by the caller
   (D2: pure, no RNG) as `(steps, B, 256, T)`, so the inert last draw is still an input.

Out of scope, one line each: `euler`/`rk4`/`dpmpp` (`sampling.py:494-499`); init-audio /
variation noise mixing `init*(1-σ) + noise*σ` (`:463-464`) — `sigma_max` is carried through the
schedule so re-enabling it is a caller-side mix; CFG/APG/norm-threshold/rescale
(`cfg_scale=1.0` on the shipped checkpoint); decode (`:509-521`) stays outside, after the scan.

`lax.scan` constraints, all of them consequences of tracing with static shapes:

- `steps` is a Python int at trace time (from `schedule.shape[-1] - 1`): one compile per step
  count. A new `steps` is a new executable, not a new iteration count.
- `T` is static too: one compile per duration bucket (120/388/775/1292 latents). Pad to the
  bucket — passing a new length recompiles instead of running.
- No Python `if`/`for` on traced values and no `.item()`/`bool()`/`int()` on a tracer. The trip
  count is Python; every per-step quantity is an array slice.
- The schedule and the padding mask are traced arrays, not closures over Python floats. The
  mask belongs to `static` and is consumed by `dit_forward`.
- The carry is `x` alone; the schedule slices and the noise ride in as `xs`, so the body never
  closes over a step index and the same traced body serves every step.
- `y` is the per-step velocity `v`, or `None` when `trajectory=False`, so XLA never
  materialises a `(steps, B, 256, T)` array that the caller does not want.

The scan is safe here because nothing about the loop is data-dependent: fixed trip count,
fixed shapes, no early exit, no state beyond `x`. That is also why it is bit-exact against the
unrolled form — see `scripts/check_sampling.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from sa3jax.config import ModelConfig
from sa3jax.dit import dit_forward
from sa3jax.dtypes import Dtype

DitParams = dict[str, object]
Static = dict[str, object]


def build_schedule(steps: int, sigma_max: float, anchor_logsnr: float, logsnr_end: float) -> Array:
    """`LogSNRShift(rate=0)` on a linear schedule, fp32 `(steps+1,)`.

    `logsnr = logsnr_end - t*(logsnr_end - anchor_logsnr)`, `t_out = sigmoid(-logsnr)`; the
    endpoints are pinned with `torch.where` in the shift and `t[0]` is re-anchored to
    `sigma_max` afterwards (`sampling.py:53-62`) so the schedule stays consistent with the
    `sigma_max`-mixed initialisation. `rate=0` collapses `get_logsnr_start` to the anchor, so
    neither `anchor_length` nor the sequence length appears.
    """
    t = jnp.linspace(sigma_max, 0.0, steps + 1, dtype=jnp.float32)
    t_out = jax.nn.sigmoid(-(logsnr_end - t * (logsnr_end - anchor_logsnr)))
    t_out = jnp.where(t <= 0.0, 0.0, t_out)
    t_out = jnp.where(t >= 1.0, 1.0, t_out)
    return t_out.at[0].set(sigma_max)


def pingpong(
    params: DitParams,
    x0: Array,
    schedule: Array,
    noise: Array,
    static: Static,
    cfg: ModelConfig,
    dtype: Dtype,
    *,
    trajectory: bool = True,
) -> tuple[Array, Array | None]:
    """`sample_flow_pingpong` as one `lax.scan`: `(x_final, velocity_trajectory)`.

    `denoised = x - t_curr*v`; `x = (1 - t_next)*denoised + t_next*noise`. `t_next = 0` on the
    last step, so the final `x` is exactly `denoised` and no noise survives.

    `x0`/`noise` are cast to `dtype.compute` — the reference's `noise.type(model_dtype)`
    (`model.py:312-313`) — which is also what makes the bf16 parity group replayable: its
    fixture tensors are bf16 upcast to fp32. `noise` is `(steps, B, 256, T)` and `schedule` is
    `(steps+1,)` or `(B, steps+1)`; the trajectory is `(steps, B, 256, T)` of per-step `v`.
    """
    latent = dtype.compute
    t = jnp.asarray(schedule)
    steps = t.shape[-1] - 1
    if t.ndim == 1:
        t = jnp.broadcast_to(t, (x0.shape[0], steps + 1))
    t = t.astype(latent)

    def step(x: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, Array | None]:
        t_curr, t_next, eps = xs
        v = dit_forward(params, x, t_curr, static, cfg.dit, dtype)
        denoised = x - t_curr[:, None, None] * v
        x_next = (1 - t_next[:, None, None]) * denoised + t_next[:, None, None] * eps
        return x_next, v if trajectory else None

    # `(steps, B)` so scan iterates over steps; `(steps, B, 256, T)` noise rides alongside
    xs = (t[:, :-1].T, t[:, 1:].T, jnp.asarray(noise, dtype=latent))
    return jax.lax.scan(step, jnp.asarray(x0, dtype=latent), xs)
    