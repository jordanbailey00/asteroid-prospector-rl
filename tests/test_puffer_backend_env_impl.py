import sys
import types
from pathlib import Path

import asteroid_prospector
import numpy as np
import pytest

from training.puffer_backend import (
    PpoConfig,
    _dispatch_step_callbacks,
    _NativeBatchVectorEnv,
    _probe_native_core_availability,
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


def test_dispatch_step_callbacks_prefers_batch_callback() -> None:
    calls = {"step": 0, "batch": 0}

    def on_step(_reward: float, _info: dict[str, object], _term: bool, _trunc: bool) -> bool:
        calls["step"] += 1
        return False

    def on_step_batch(
        rewards: np.ndarray,
        infos: object,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> bool:
        calls["batch"] += 1
        assert rewards.shape == (3,)
        assert terminated.shape == (3,)
        assert truncated.shape == (3,)
        assert isinstance(infos, dict)
        return False

    stop = _dispatch_step_callbacks(
        on_step=on_step,
        on_step_batch=on_step_batch,
        rewards=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        infos={"dt": np.array([1, 1, 1], dtype=np.int32)},
        terminated=np.array([False, False, False], dtype=bool),
        truncated=np.array([False, False, False], dtype=bool),
    )

    assert stop is False
    assert calls["batch"] == 1
    assert calls["step"] == 0


def test_dispatch_step_callbacks_falls_back_to_per_env_step_callback() -> None:
    calls = {"step": 0}

    def on_step(_reward: float, info: dict[str, object], _term: bool, _trunc: bool) -> bool:
        calls["step"] += 1
        assert "dt" in info
        return calls["step"] == 2

    stop = _dispatch_step_callbacks(
        on_step=on_step,
        on_step_batch=None,
        rewards=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        infos={"dt": np.array([1, 2, 3], dtype=np.int32)},
        terminated=np.array([False, False, False], dtype=bool),
        truncated=np.array([False, False, False], dtype=bool),
    )

    assert stop is True
    assert calls["step"] == 2


def test_native_batch_vector_env_uses_batch_bridge_and_autoresets_done_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gym = types.SimpleNamespace(
        spaces=types.SimpleNamespace(Discrete=_FakeDiscrete, Box=_FakeBox)
    )
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)

    class FakeCore:
        instances: list["FakeCore"] = []
        reset_many_calls: list[dict[str, object]] = []
        step_many_calls: list[dict[str, object]] = []

        def __init__(self, seed: int, *, config: object) -> None:
            self.seed = int(seed)
            self.config = config
            self.closed = False
            FakeCore.instances.append(self)

        @staticmethod
        def reset_many(cores: list["FakeCore"], seeds: list[int]) -> np.ndarray:
            FakeCore.reset_many_calls.append(
                {
                    "count": len(cores),
                    "seeds": [int(seed) for seed in seeds],
                }
            )
            obs = np.full((len(cores), asteroid_prospector.OBS_DIM), 7.0, dtype=np.float32)
            for i in range(len(cores)):
                obs[i, 0] = np.float32(7.0 + i)
            return obs

        @staticmethod
        def step_many(
            cores: list["FakeCore"],
            actions: list[int],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
            count = len(cores)
            FakeCore.step_many_calls.append(
                {
                    "count": count,
                    "actions": [int(action) for action in actions],
                }
            )
            obs = np.full((count, asteroid_prospector.OBS_DIM), 0.5, dtype=np.float32)
            rewards = np.arange(1, count + 1, dtype=np.float32)
            terminated = np.zeros((count,), dtype=bool)
            if count > 1:
                terminated[1] = True
            truncated = np.zeros((count,), dtype=bool)
            infos = {
                "dt": np.ones((count,), dtype=np.int32),
                "action_received": np.asarray(actions, dtype=np.int64),
            }
            return obs, rewards, terminated, truncated, infos

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(asteroid_prospector, "NativeProspectorCore", FakeCore)

    env = _NativeBatchVectorEnv(time_max=321.0, seed=7, num_envs=3)

    assert env.single_action_space.n == asteroid_prospector.N_ACTIONS
    assert env.single_observation_space.shape == (asteroid_prospector.OBS_DIM,)

    reset_obs, reset_info = env.reset(seed=11)
    assert reset_obs.shape == (3, asteroid_prospector.OBS_DIM)
    assert reset_obs.dtype == np.float32
    assert reset_info == {}
    assert len(FakeCore.reset_many_calls) == 1
    assert FakeCore.reset_many_calls[0]["count"] == 3

    obs, rewards, terminated, truncated, infos = env.step(np.array([4, 5, 6], dtype=np.int64))

    assert len(FakeCore.step_many_calls) == 1
    assert FakeCore.step_many_calls[0]["actions"] == [4, 5, 6]
    assert rewards.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert terminated.tolist() == [False, True, False]
    assert truncated.tolist() == [False, False, False]
    assert isinstance(infos, dict)

    assert obs[0, 0] == pytest.approx(0.5)
    assert obs[2, 0] == pytest.approx(0.5)
    assert obs[1, 0] == pytest.approx(7.0)

    assert len(FakeCore.reset_many_calls) == 2
    assert FakeCore.reset_many_calls[1]["count"] == 1

    env.close()
    assert all(core.closed for core in FakeCore.instances)


def test_native_batch_vector_env_rejects_wrong_action_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gym = types.SimpleNamespace(
        spaces=types.SimpleNamespace(Discrete=_FakeDiscrete, Box=_FakeBox)
    )
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)

    class FakeCore:
        def __init__(self, seed: int, *, config: object) -> None:
            self.seed = int(seed)
            self.config = config

        @staticmethod
        def reset_many(cores: list["FakeCore"], seeds: list[int]) -> np.ndarray:
            del seeds
            return np.zeros((len(cores), asteroid_prospector.OBS_DIM), dtype=np.float32)

        @staticmethod
        def step_many(
            cores: list["FakeCore"],
            actions: list[int],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
            del actions
            count = len(cores)
            return (
                np.zeros((count, asteroid_prospector.OBS_DIM), dtype=np.float32),
                np.zeros((count,), dtype=np.float32),
                np.zeros((count,), dtype=bool),
                np.zeros((count,), dtype=bool),
                {},
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(asteroid_prospector, "NativeProspectorCore", FakeCore)

    env = _NativeBatchVectorEnv(time_max=100.0, seed=5, num_envs=2)
    with pytest.raises(ValueError, match="Expected 2 actions"):
        env.step(np.array([1], dtype=np.int64))
    env.close()


def test_probe_native_core_availability_handles_load_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_library = tmp_path / "abp_core.dll"
    fake_library.write_bytes(b"not a real library")

    class _FailingCore:
        def __init__(self, seed: int, *, config: object) -> None:
            del seed, config
            raise OSError("invalid ELF header")

    monkeypatch.setattr(asteroid_prospector, "NativeProspectorCore", _FailingCore)
    import asteroid_prospector.native_core as native_core_module

    monkeypatch.setattr(
        native_core_module,
        "default_native_library_path",
        lambda: fake_library,
    )

    available, detail = _probe_native_core_availability()

    assert available is False
    assert detail is not None
    assert "load_error:OSError" in detail
