from __future__ import annotations

import ctypes
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
    ]


class _AbpCoreState(ctypes.Structure):
    pass


def default_native_library_path() -> Path:
    return Path(__file__).resolve().parents[2] / "engine_core" / "build" / "abp_core.dll"


class NativeProspectorCore:
    def __init__(
        self,
        seed: int = 0,
        *,
        config: NativeCoreConfig | None = None,
        library_path: str | Path | None = None,
    ) -> None:
        self._cfg_ref: _AbpCoreConfig | None = None
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

    def reset(self, seed: int) -> np.ndarray:
        obs_buffer = (ctypes.c_float * OBS_DIM)()
        self._lib.abp_core_reset(self._state, ctypes.c_uint64(int(seed)), obs_buffer)
        return np.ctypeslib.as_array(obs_buffer).copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_value = int(action)
        action_u8 = action_value if 0 <= action_value < 256 else N_ACTIONS

        result = _AbpCoreStepResult()
        self._lib.abp_core_step(self._state, ctypes.c_uint8(action_u8), ctypes.byref(result))

        obs = np.ctypeslib.as_array(result.obs).copy()
        reward = float(result.reward)
        terminated = bool(result.terminated)
        truncated = bool(result.truncated)
        info = {
            "invalid_action": bool(result.invalid_action),
            "dt": int(result.dt),
            "action_received": action_value,
        }
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
