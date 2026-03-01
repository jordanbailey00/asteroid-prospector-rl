import ctypes
import sys

import numpy as np
import pytest
from asteroid_prospector.constants import N_ACTIONS, OBS_DIM
from asteroid_prospector.native_core import (
    NativeProspectorCore,
    _AbpCoreState,
    default_native_library_path,
)


def test_default_native_library_path_points_to_platform_location() -> None:
    path = default_native_library_path()
    assert path.parent.name == "build"

    if sys.platform.startswith("win"):
        assert path.name == "abp_core.dll"
    elif sys.platform == "darwin":
        assert path.name in {"abp_core.dylib", "abp_core.so", "abp_core.dll"}
    else:
        assert path.name in {"abp_core.so", "abp_core.dylib", "abp_core.dll"}


def test_native_core_raises_for_missing_library(tmp_path) -> None:
    missing = tmp_path / "missing_core.dll"
    with pytest.raises(FileNotFoundError):
        NativeProspectorCore(seed=0, library_path=missing)


def test_reset_many_fallback_uses_scalar_resets() -> None:
    class FakeCore:
        def __init__(self, offset: float) -> None:
            self.offset = float(offset)
            self.seeds: list[int] = []

        def reset(self, seed: int) -> np.ndarray:
            self.seeds.append(int(seed))
            return np.full((OBS_DIM,), float(seed) + self.offset, dtype=np.float32)

    cores = [FakeCore(0.0), FakeCore(1.0)]
    obs = NativeProspectorCore.reset_many(cores, [3, 5])

    assert obs.shape == (2, OBS_DIM)
    assert np.allclose(obs[0], 3.0)
    assert np.allclose(obs[1], 6.0)
    assert cores[0].seeds == [3]
    assert cores[1].seeds == [5]


def test_step_many_fallback_uses_scalar_steps() -> None:
    class FakeCore:
        def __init__(self, offset: float) -> None:
            self.offset = float(offset)
            self.actions: list[int] = []

        def step(self, action: int):
            action_i = int(action)
            self.actions.append(action_i)
            obs = np.full((OBS_DIM,), self.offset + float(action_i), dtype=np.float32)
            reward = float(action_i) * 0.5
            terminated = bool(action_i % 2 == 0)
            truncated = False
            info = {
                "action": action_i - 1,
                "dt": 2,
                "invalid_action": False,
                "credits": 1.5,
                "terminated": terminated,
                "truncated": truncated,
            }
            return obs, reward, terminated, truncated, info

    cores = [FakeCore(0.0), FakeCore(1.0)]
    obs, rewards, terminated, truncated, infos = NativeProspectorCore.step_many(cores, [2, 3])

    assert obs.shape == (2, OBS_DIM)
    assert rewards.shape == (2,)
    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, False]
    assert infos["action_received"].tolist() == [2, 3]
    assert infos["action"].tolist() == [1, 2]
    assert infos["dt"].tolist() == [2, 2]
    assert infos["credits"].tolist() == [1.5, 1.5]
    assert cores[0].actions == [2]
    assert cores[1].actions == [3]


def test_reset_many_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        NativeProspectorCore.reset_many([], [1])


def test_step_many_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        NativeProspectorCore.step_many([], [1])


def test_batch_many_path_uses_native_many_symbols() -> None:
    class FakeLib:
        def __init__(self) -> None:
            self.reset_many_calls = 0
            self.step_many_calls = 0

        def abp_core_reset_many(self, states, seeds, count, obs_out) -> None:
            del states
            self.reset_many_calls += 1
            n = int(count.value) if hasattr(count, "value") else int(count)
            seed_vals = np.ctypeslib.as_array(seeds, shape=(n,))
            obs_vals = np.ctypeslib.as_array(obs_out, shape=(n * OBS_DIM,))
            for i in range(n):
                obs_vals[i * OBS_DIM : (i + 1) * OBS_DIM] = float(seed_vals[i])

        def abp_core_step_many(self, states, actions, count, out_results) -> None:
            del states
            self.step_many_calls += 1
            n = int(count.value) if hasattr(count, "value") else int(count)
            action_vals = np.ctypeslib.as_array(actions, shape=(n,))
            for i in range(n):
                result = out_results[i]
                for j in range(OBS_DIM):
                    result.obs[j] = float(i)
                result.reward = float(action_vals[i]) * 0.1
                result.terminated = 1 if i == 0 else 0
                result.truncated = 1 if i == 1 else 0
                result.invalid_action = 1 if int(action_vals[i]) >= N_ACTIONS else 0
                result.dt = i + 1
                result.action = int(action_vals[i])
                result.credits = 10.0 + float(i)
                result.net_profit = 20.0 + float(i)
                result.profit_per_tick = 2.0 + float(i)
                result.survival = 1.0 - 0.5 * float(i)
                result.overheat_ticks = float(i)
                result.pirate_encounters = float(i + 2)
                result.value_lost_to_pirates = float(i + 3)
                result.fuel_used = float(i + 4)
                result.hull_damage = float(i + 5)
                result.tool_wear = float(i + 6)
                result.scan_count = float(i + 7)
                result.mining_ticks = float(i + 8)
                result.cargo_utilization_avg = 0.25 + 0.25 * float(i)
                result.time_remaining = 100.0 - float(i)

    def make_core(lib: FakeLib) -> NativeProspectorCore:
        core = object.__new__(NativeProspectorCore)
        core._lib = lib
        core._has_batch_apis = True
        core._state = ctypes.POINTER(_AbpCoreState)()
        return core

    lib = FakeLib()
    cores = [make_core(lib), make_core(lib)]

    reset_obs = NativeProspectorCore.reset_many(cores, [7, 9])
    assert lib.reset_many_calls == 1
    assert reset_obs.shape == (2, OBS_DIM)
    assert np.allclose(reset_obs[0], 7.0)
    assert np.allclose(reset_obs[1], 9.0)

    obs, rewards, terminated, truncated, infos = NativeProspectorCore.step_many(cores, [5, 999])

    assert lib.step_many_calls == 1
    assert obs.shape == (2, OBS_DIM)
    assert np.allclose(obs[0], 0.0)
    assert np.allclose(obs[1], 1.0)
    assert rewards.tolist() == pytest.approx([0.5, float(N_ACTIONS) * 0.1])
    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, True]
    assert infos["action_received"].tolist() == [5, 999]
    assert infos["action"].tolist() == [5, N_ACTIONS]
    assert infos["invalid_action"].tolist() == [False, True]
