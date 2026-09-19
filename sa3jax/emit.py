"""Parity emitter: records JAX activations under the reference fixture's key layout.

The contract is `docs/parity-fixture.md`. Keys are `<group>/<stage>/<name>` and must match the
reference exactly — `scripts/compare_features.py` treats a key the reference has and the
candidate lacks as a FAIL. Noise is *loaded from* the fixture and injected, never regenerated:
torch's Philox stream and JAX's Threefry are different generators, and the noise carries no
information worth reproducing.
"""
from __future__ import annotations

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


class Recorder:
    """Collects named arrays for one capture group and writes them as a safetensors file."""

    def __init__(self, group: str) -> None:
        self.group = group
        self.tensors: dict[str, np.ndarray] = {}

    def add(self, key: str, value: np.ndarray | None) -> None:
        """Store `value` under `<group>/<key>`. `None` is a no-op so callers can chain."""
        if value is None:
            return
        self.tensors[f"{self.group}/{key}"] = np.asarray(value)

    def add_int(self, key: str, value: int) -> None:
        self.add(key, np.asarray(value, dtype=np.int64))

    def save(self, path: str) -> int:
        save_file(self.tensors, path)
        return len(self.tensors)


def load_noise(ref_path: str, group: str) -> dict[str, np.ndarray]:
    """Read one group's captured noise draws, keyed by role (`sampler`, `ae_mask_noise`, ...).

    Returns `{role: (n_draws, ...)}`-style entries as a flat dict of `noise/<role>_<i>`.
    """
    out: dict[str, np.ndarray] = {}
    with safe_open(ref_path, framework="numpy") as handle:
        for key in handle.keys():
            if key.startswith(f"{group}/noise/"):
                out[key.split("/", 2)[2]] = handle.get_tensor(key)
    return out


def noise_draws(noise: dict[str, np.ndarray], role: str) -> list[np.ndarray]:
    """All draws for one role, in call order (`sampler_00`, `sampler_01`, ...)."""
    items = [(k, v) for k, v in noise.items() if k.startswith(f"{role}_")]
    return [v for _k, v in sorted(items)]
