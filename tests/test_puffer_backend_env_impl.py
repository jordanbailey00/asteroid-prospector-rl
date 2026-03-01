import sys
import types

import asteroid_prospector
import numpy as np
import pytest

from training.puffer_backend import (
    PpoConfig,
    _ProspectorNativeGymEnv,
    _resolve_env_impl,
    _validate_config,
)


def _base_cfg(**overrides: object) -> PpoConfig:
    payload = {
        "total_env_steps": 128,
        "seed": 7,
        "env_time_max": 1000.0,
        "num_envs": 4,
        "num_workers": 2,
        "rollout_steps": 8,
        "num_minibatches": 2,
        "update_epochs": 1,
        "learning_rate": 3.0e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "vector_backend": "multiprocessing",
        "env_impl": "auto",
    }
    payload.update(overrides)
    return PpoConfig(**payload)


def test_validate_config_rejects_invalid_env_impl() -> None:
    cfg = _base_cfg(env_impl="unsupported")
    with pytest.raises(ValueError, match="ppo_env_impl"):
        _validate_config(cfg)


def test_resolve_env_impl_auto_falls_back_to_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "training.puffer_backend._probe_native_core_availability",
        lambda: (False, "native-missing"),
    )

    selected, native_available, detail = _resolve_env_impl("auto")

    assert selected == "reference"
    assert native_available is False
    assert detail == "native-missing"


def test_resolve_env_impl_auto_prefers_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "training.puffer_backend._probe_native_core_availability",
        lambda: (True, "native-ok"),
    )

    selected, native_available, detail = _resolve_env_impl("auto")

    assert selected == "native"
    assert native_available is True
    assert detail == "native-ok"


def test_resolve_env_impl_native_requires_available_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "training.puffer_backend._probe_native_core_availability",
        lambda: (False, "native-missing"),
    )

    with pytest.raises(RuntimeError, match="ppo_env_impl='native'"):
        _resolve_env_impl("native")


class _FakeDiscrete:
    def __init__(self, n: int) -> None:
        self.n = int(n)


class _FakeBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: object) -> None:
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype


def test_native_wrapper_contract_with_fake_core(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gym = types.SimpleNamespace(
        spaces=types.SimpleNamespace(Discrete=_FakeDiscrete, Box=_FakeBox)
    )
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)

    state: dict[str, object] = {}

    class FakeCore:
        def __init__(self, seed: int, *, config: object) -> None:
            state["seed"] = int(seed)
            state["config"] = config
            state["instance"] = self
            self.reset_calls: list[int] = []
            self.step_calls: list[int] = []
            self.closed = False

        def reset(self, seed: int) -> np.ndarray:
            self.reset_calls.append(int(seed))
            return np.full((asteroid_prospector.OBS_DIM,), 0.25, dtype=np.float32)

        def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            self.step_calls.append(int(action))
            obs = np.full((asteroid_prospector.OBS_DIM,), 0.5, dtype=np.float32)
            return obs, 1.25, True, False, {"action": int(action)}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(asteroid_prospector, "NativeProspectorCore", FakeCore)

    env = _ProspectorNativeGymEnv(time_max=321.0, seed=7)

    assert env.action_space.n == asteroid_prospector.N_ACTIONS
    assert env.observation_space.shape == (asteroid_prospector.OBS_DIM,)

    obs, info = env.reset(seed=11)
    assert obs.shape == (asteroid_prospector.OBS_DIM,)
    assert obs.dtype == np.float32
    assert info == {}

    instance = state["instance"]
    assert isinstance(instance, FakeCore)
    assert instance.reset_calls[0] == 11

    step_obs, reward, terminated, truncated, step_info = env.step(3)
    assert step_obs.shape == (asteroid_prospector.OBS_DIM,)
    assert step_obs.dtype == np.float32
    assert reward == pytest.approx(1.25)
    assert terminated is True
    assert truncated is False
    assert step_info["action"] == 3

    _, _ = env.reset()
    assert isinstance(instance.reset_calls[-1], int)

    env.close()
    assert instance.closed is True
