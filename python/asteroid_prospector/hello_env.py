"""M0 hello environment.

This is a contract-only stub that preserves the frozen interface:
- observation shape `(260,)` float32
- action space size 69 (`0..68`)
- Gymnasium-style step return tuple
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import INVALID_ACTION_PENALTY, N_ACTIONS, OBS_DIM


@dataclass(frozen=True)
class DiscreteActionSpace:
    """Minimal Discrete(n) stand-in for M0 without extra dependencies."""

    n: int

    def contains(self, action: int) -> bool:
        return isinstance(action, (int, np.integer)) and 0 <= int(action) < self.n


@dataclass(frozen=True)
class ObservationSpace:
    """Minimal observation-space descriptor for contract introspection."""

    shape: tuple[int, ...]
    dtype: Any


class HelloProspectorEnv:
    """Contract-only environment stub for Milestone M0."""

    def __init__(self, seed: int | None = None) -> None:
        self.action_space = DiscreteActionSpace(N_ACTIONS)
        self.observation_space = ObservationSpace(shape=(OBS_DIM,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._obs = np.zeros((OBS_DIM,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options  # Reserved for future Gymnasium compatibility.
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._obs.fill(0.0)
        return self._obs.copy(), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        invalid_action = not self.action_space.contains(action)
        reward = INVALID_ACTION_PENALTY if invalid_action else 0.0
        info = {
            "invalid_action": invalid_action,
            "action_received": int(action),
        }
        return self._obs.copy(), float(reward), False, False, info
