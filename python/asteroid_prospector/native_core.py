from __future__ import annotations

import ctypes
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import N_ACTIONS, OBS_DIM


@dataclass(frozen=True)
class NativeCoreConfig:
    time_max: float = 20000.0
    invalid_action_penalty: float = 0.01


class _AbpCoreConfig(ctypes.Structure):
    _fields_ = [
        ("time_max", ctypes.c_float),
        ("invalid_action_penalty", ctypes.c_float),
    ]


class _AbpCoreStepResult(ctypes.Structure):
    _fields_ = [
        ("obs", ctypes.c_float * OBS_DIM),
        ("reward", ctypes.c_float),
        ("terminated", ctypes.c_uint8),
        ("truncated", ctypes.c_uint8),
        ("invalid_action", ctypes.c_uint8),
        ("dt", ctypes.c_uint16),
        ("action", ctypes.c_int16),
        ("credits", ctypes.c_float),
        ("net_profit", ctypes.c_float),
        ("profit_per_tick", ctypes.c_float),
        ("survival", ctypes.c_float),
        ("overheat_ticks", ctypes.c_float),
        ("pirate_encounters", ctypes.c_float),
        ("value_lost_to_pirates", ctypes.c_float),
        ("fuel_used", ctypes.c_float),
        ("hull_damage", ctypes.c_float),
        ("tool_wear", ctypes.c_float),
        ("scan_count", ctypes.c_float),
        ("mining_ticks", ctypes.c_float),
        ("cargo_utilization_avg", ctypes.c_float),
        ("time_remaining", ctypes.c_float),
    ]


class _AbpCoreState(ctypes.Structure):
    pass


def default_native_library_path() -> Path:
    build_dir = Path(__file__).resolve().parents[2] / "engine_core" / "build"

    if sys.platform.startswith("win"):
        candidate_names = ("abp_core.dll",)
    elif sys.platform == "darwin":
        candidate_names = ("abp_core.dylib", "abp_core.so", "abp_core.dll")
    else:
        candidate_names = ("abp_core.so", "abp_core.dylib", "abp_core.dll")

    for name in candidate_names:
        candidate = build_dir / name
        if candidate.exists():
            return candidate

    return build_dir / candidate_names[0]


class NativeProspectorCore:
    def __init__(
        self,
        seed: int = 0,
        *,
        config: NativeCoreConfig | None = None,
        library_path: str | Path | None = None,
    ) -> None:
        self._cfg_ref: _AbpCoreConfig | None = None
        self._has_batch_apis = False
        library_file = (
            Path(library_path) if library_path is not None else default_native_library_path()
        )

        if not library_file.exists():
            raise FileNotFoundError(
                f"Native core library not found at '{library_file}'. "
                "Run .\\tools\\build_native_core.ps1 first."
            )

        self._lib = ctypes.CDLL(str(library_file))
        self._configure_signatures()

        if config is None:
            self._state = self._lib.abp_core_create(None, ctypes.c_uint64(int(seed)))
        else:
            self._cfg_ref = _AbpCoreConfig(
                time_max=float(config.time_max),
                invalid_action_penalty=float(config.invalid_action_penalty),
            )
            self._state = self._lib.abp_core_create(
                ctypes.byref(self._cfg_ref), ctypes.c_uint64(int(seed))
            )

        if not self._state:
            raise RuntimeError("Failed to create native core state")

    def _configure_signatures(self) -> None:
        self._lib.abp_core_create.argtypes = [ctypes.POINTER(_AbpCoreConfig), ctypes.c_uint64]
        self._lib.abp_core_create.restype = ctypes.POINTER(_AbpCoreState)

        self._lib.abp_core_destroy.argtypes = [ctypes.POINTER(_AbpCoreState)]
        self._lib.abp_core_destroy.restype = None

        self._lib.abp_core_reset.argtypes = [
            ctypes.POINTER(_AbpCoreState),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.abp_core_reset.restype = None

        self._lib.abp_core_step.argtypes = [
            ctypes.POINTER(_AbpCoreState),
            ctypes.c_uint8,
            ctypes.POINTER(_AbpCoreStepResult),
        ]
        self._lib.abp_core_step.restype = None

        if hasattr(self._lib, "abp_core_reset_many") and hasattr(self._lib, "abp_core_step_many"):
            self._lib.abp_core_reset_many.argtypes = [
                ctypes.POINTER(ctypes.POINTER(_AbpCoreState)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_float),
            ]
            self._lib.abp_core_reset_many.restype = None

            self._lib.abp_core_step_many.argtypes = [
                ctypes.POINTER(ctypes.POINTER(_AbpCoreState)),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_uint32,
                ctypes.POINTER(_AbpCoreStepResult),
            ]
            self._lib.abp_core_step_many.restype = None
            self._has_batch_apis = True

    @staticmethod
    def _coerce_action_u8(action: int) -> int:
        action_value = int(action)
        return action_value if 0 <= action_value < 256 else N_ACTIONS

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed):
            return float(default)
        return parsed

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        try:
            return bool(value)
        except (TypeError, ValueError):
            return bool(default)

    @staticmethod
    def _info_from_result(
        result: _AbpCoreStepResult,
        *,
        action_received: int,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "action": int(result.action),
            "action_received": int(action_received),
            "dt": int(result.dt),
            "invalid_action": bool(result.invalid_action),
            "credits": float(result.credits),
            "net_profit": float(result.net_profit),
            "profit_per_tick": float(result.profit_per_tick),
            "survival": float(result.survival),
            "overheat_ticks": float(result.overheat_ticks),
            "pirate_encounters": float(result.pirate_encounters),
            "value_lost_to_pirates": float(result.value_lost_to_pirates),
            "fuel_used": float(result.fuel_used),
            "hull_damage": float(result.hull_damage),
            "tool_wear": float(result.tool_wear),
            "scan_count": float(result.scan_count),
            "mining_ticks": float(result.mining_ticks),
            "cargo_utilization_avg": float(result.cargo_utilization_avg),
            "time_remaining": float(result.time_remaining),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    @staticmethod
    def _allocate_info_arrays(
        count: int, *, action_received: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        received = np.zeros((count,), dtype=np.int64)
        if action_received is not None:
            received = np.asarray(action_received, dtype=np.int64).copy()

        return {
            "action": np.zeros((count,), dtype=np.int16),
            "action_received": received,
            "dt": np.zeros((count,), dtype=np.int32),
            "invalid_action": np.zeros((count,), dtype=bool),
            "credits": np.zeros((count,), dtype=np.float32),
            "net_profit": np.zeros((count,), dtype=np.float32),
            "profit_per_tick": np.zeros((count,), dtype=np.float32),
            "survival": np.zeros((count,), dtype=np.float32),
            "overheat_ticks": np.zeros((count,), dtype=np.float32),
            "pirate_encounters": np.zeros((count,), dtype=np.float32),
            "value_lost_to_pirates": np.zeros((count,), dtype=np.float32),
            "fuel_used": np.zeros((count,), dtype=np.float32),
            "hull_damage": np.zeros((count,), dtype=np.float32),
            "tool_wear": np.zeros((count,), dtype=np.float32),
            "scan_count": np.zeros((count,), dtype=np.float32),
            "mining_ticks": np.zeros((count,), dtype=np.float32),
            "cargo_utilization_avg": np.zeros((count,), dtype=np.float32),
            "time_remaining": np.zeros((count,), dtype=np.float32),
            "terminated": np.zeros((count,), dtype=bool),
            "truncated": np.zeros((count,), dtype=bool),
        }

    @staticmethod
    def _infos_rows_to_arrays(
        rows: list[dict[str, Any]], action_received: np.ndarray
    ) -> dict[str, np.ndarray]:
        count = len(rows)
        infos = NativeProspectorCore._allocate_info_arrays(count, action_received=action_received)

        for i, row in enumerate(rows):
            infos["action"][i] = np.int16(NativeProspectorCore._safe_int(row.get("action", 0)))
            infos["dt"][i] = np.int32(NativeProspectorCore._safe_int(row.get("dt", 1)))
            infos["invalid_action"][i] = NativeProspectorCore._safe_bool(
                row.get("invalid_action", False)
            )
            infos["credits"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("credits", 0.0))
            )
            infos["net_profit"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("net_profit", 0.0))
            )
            infos["profit_per_tick"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("profit_per_tick", 0.0))
            )
            infos["survival"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("survival", 0.0))
            )
            infos["overheat_ticks"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("overheat_ticks", 0.0))
            )
            infos["pirate_encounters"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("pirate_encounters", 0.0))
            )
            infos["value_lost_to_pirates"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("value_lost_to_pirates", 0.0))
            )
            infos["fuel_used"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("fuel_used", 0.0))
            )
            infos["hull_damage"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("hull_damage", 0.0))
            )
            infos["tool_wear"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("tool_wear", 0.0))
            )
            infos["scan_count"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("scan_count", 0.0))
            )
            infos["mining_ticks"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("mining_ticks", 0.0))
            )
            infos["cargo_utilization_avg"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("cargo_utilization_avg", 0.0))
            )
            infos["time_remaining"][i] = np.float32(
                NativeProspectorCore._safe_float(row.get("time_remaining", 0.0))
            )
            infos["terminated"][i] = NativeProspectorCore._safe_bool(row.get("terminated", False))
            infos["truncated"][i] = NativeProspectorCore._safe_bool(row.get("truncated", False))

        return infos

    @staticmethod
    def _can_use_batch_ops(cores: Sequence[Any]) -> bool:
        if len(cores) == 0:
            return False

        first = cores[0]
        if not isinstance(first, NativeProspectorCore):
            return False
        if not bool(getattr(first, "_has_batch_apis", False)):
            return False

        lib = getattr(first, "_lib", None)
        if lib is None:
            return False

        for core in cores:
            if not isinstance(core, NativeProspectorCore):
                return False
            if not bool(getattr(core, "_has_batch_apis", False)):
                return False
            if getattr(core, "_lib", None) is not lib:
                return False
            if getattr(core, "_state", None) is None:
                return False
        return True

    @staticmethod
    def reset_many(cores: Sequence[Any], seeds: Sequence[int]) -> np.ndarray:
        count = len(cores)
        if count != len(seeds):
            raise ValueError("reset_many requires equal lengths for cores and seeds")
        if count == 0:
            return np.zeros((0, OBS_DIM), dtype=np.float32)

        if NativeProspectorCore._can_use_batch_ops(cores):
            first: NativeProspectorCore = cores[0]
            state_array_type = ctypes.POINTER(_AbpCoreState) * count
            state_ptrs = state_array_type(*(core._state for core in cores))
            seed_array = (ctypes.c_uint64 * count)(*(int(seed) for seed in seeds))
            obs = np.empty((count, OBS_DIM), dtype=np.float32)
            obs_ptr = obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            first._lib.abp_core_reset_many(
                state_ptrs,
                seed_array,
                ctypes.c_uint32(count),
                obs_ptr,
            )
            return obs

        obs = np.empty((count, OBS_DIM), dtype=np.float32)
        for i, core in enumerate(cores):
            obs[i] = np.asarray(core.reset(int(seeds[i])), dtype=np.float32)
        return obs

    @staticmethod
    def step_many(
        cores: Sequence[Any],
        actions: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        count = len(cores)
        if count != len(actions):
            raise ValueError("step_many requires equal lengths for cores and actions")

        if count == 0:
            empty_obs = np.zeros((0, OBS_DIM), dtype=np.float32)
            empty = np.zeros((0,), dtype=np.float32)
            empty_bool = np.zeros((0,), dtype=bool)
            infos = NativeProspectorCore._allocate_info_arrays(0)
            return empty_obs, empty, empty_bool, empty_bool, infos

        action_received = np.asarray(actions, dtype=np.int64).reshape(-1)

        if NativeProspectorCore._can_use_batch_ops(cores):
            first: NativeProspectorCore = cores[0]
            state_array_type = ctypes.POINTER(_AbpCoreState) * count
            state_ptrs = state_array_type(*(core._state for core in cores))

            action_u8 = np.empty((count,), dtype=np.uint8)
            for i in range(count):
                action_u8[i] = np.uint8(
                    NativeProspectorCore._coerce_action_u8(int(action_received[i]))
                )
            action_array = (ctypes.c_uint8 * count)(*(int(v) for v in action_u8))

            result_array_type = _AbpCoreStepResult * count
            results = result_array_type()
            first._lib.abp_core_step_many(
                state_ptrs,
                action_array,
                ctypes.c_uint32(count),
                results,
            )

            obs = np.empty((count, OBS_DIM), dtype=np.float32)
            rewards = np.empty((count,), dtype=np.float32)
            terminated = np.empty((count,), dtype=bool)
            truncated = np.empty((count,), dtype=bool)
            infos = NativeProspectorCore._allocate_info_arrays(
                count, action_received=action_received
            )

            for i in range(count):
                result = results[i]
                obs[i] = np.ctypeslib.as_array(result.obs).astype(np.float32, copy=True)
                rewards[i] = np.float32(result.reward)
                terminated[i] = bool(result.terminated)
                truncated[i] = bool(result.truncated)

                infos["action"][i] = np.int16(result.action)
                infos["dt"][i] = np.int32(result.dt)
                infos["invalid_action"][i] = bool(result.invalid_action)
                infos["credits"][i] = np.float32(result.credits)
                infos["net_profit"][i] = np.float32(result.net_profit)
                infos["profit_per_tick"][i] = np.float32(result.profit_per_tick)
                infos["survival"][i] = np.float32(result.survival)
                infos["overheat_ticks"][i] = np.float32(result.overheat_ticks)
                infos["pirate_encounters"][i] = np.float32(result.pirate_encounters)
                infos["value_lost_to_pirates"][i] = np.float32(result.value_lost_to_pirates)
                infos["fuel_used"][i] = np.float32(result.fuel_used)
                infos["hull_damage"][i] = np.float32(result.hull_damage)
                infos["tool_wear"][i] = np.float32(result.tool_wear)
                infos["scan_count"][i] = np.float32(result.scan_count)
                infos["mining_ticks"][i] = np.float32(result.mining_ticks)
                infos["cargo_utilization_avg"][i] = np.float32(result.cargo_utilization_avg)
                infos["time_remaining"][i] = np.float32(result.time_remaining)
                infos["terminated"][i] = bool(result.terminated)
                infos["truncated"][i] = bool(result.truncated)

            return obs, rewards, terminated, truncated, infos

        obs = np.empty((count, OBS_DIM), dtype=np.float32)
        rewards = np.empty((count,), dtype=np.float32)
        terminated = np.empty((count,), dtype=bool)
        truncated = np.empty((count,), dtype=bool)
        info_rows: list[dict[str, Any]] = []

        for i, core in enumerate(cores):
            obs_i, reward_i, term_i, trunc_i, info_i = core.step(int(action_received[i]))
            obs[i] = np.asarray(obs_i, dtype=np.float32)
            rewards[i] = np.float32(reward_i)
            terminated[i] = bool(term_i)
            truncated[i] = bool(trunc_i)

            row = dict(info_i)
            row.setdefault("action_received", int(action_received[i]))
            row.setdefault("terminated", bool(term_i))
            row.setdefault("truncated", bool(trunc_i))
            info_rows.append(row)

        infos = NativeProspectorCore._infos_rows_to_arrays(info_rows, action_received)
        return obs, rewards, terminated, truncated, infos

    def reset(self, seed: int) -> np.ndarray:
        obs_buffer = (ctypes.c_float * OBS_DIM)()
        self._lib.abp_core_reset(self._state, ctypes.c_uint64(int(seed)), obs_buffer)
        return np.ctypeslib.as_array(obs_buffer).copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_value = int(action)
        action_u8 = self._coerce_action_u8(action_value)

        result = _AbpCoreStepResult()
        self._lib.abp_core_step(self._state, ctypes.c_uint8(action_u8), ctypes.byref(result))

        obs = np.ctypeslib.as_array(result.obs).copy()
        reward = float(result.reward)
        terminated = bool(result.terminated)
        truncated = bool(result.truncated)
        info = self._info_from_result(
            result,
            action_received=action_value,
            terminated=terminated,
            truncated=truncated,
        )
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if hasattr(self, "_state") and self._state:
            self._lib.abp_core_destroy(self._state)
            self._state = None

    def __enter__(self) -> NativeProspectorCore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.close()

    def __del__(self) -> None:
        self.close()
