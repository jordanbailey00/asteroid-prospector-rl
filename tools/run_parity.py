from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from asteroid_prospector import (
    NativeCoreConfig,
    NativeProspectorCore,
    ProspectorReferenceEnv,
    ReferenceEnvConfig,
)

NUMERIC_INFO_KEYS = (
    "credits",
    "net_profit",
    "profit_per_tick",
    "survival",
    "overheat_ticks",
    "pirate_encounters",
    "value_lost_to_pirates",
    "fuel_used",
    "hull_damage",
    "tool_wear",
    "scan_count",
    "mining_ticks",
    "cargo_utilization_avg",
)


@dataclass(frozen=True)
class TraceData:
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dt: np.ndarray
    obs: np.ndarray
    info: dict[str, np.ndarray]


class ParityMismatch(RuntimeError):
    pass


def _generate_actions(suite: str, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if suite == "A":
        return rng.integers(-5, 75, size=steps, dtype=np.int32)

    if suite == "B":
        pattern = np.array(
            [
                43,
                61,
                67,
                28,
                29,
                30,
                10,
                9,
                7,
                33,
                200,
                -3,
                255,
                6,
                42,
                68,
            ],
            dtype=np.int32,
        )
    elif suite == "C":
        pattern = np.array(
            [
                8,
                11,
                12,
                29,
                33,
                6,
                0,
                42,
                45,
                32,
                35,
                7,
                6,
                68,
            ],
            dtype=np.int32,
        )
    else:
        raise ValueError(f"Unsupported suite: {suite}")

    tiled = np.tile(pattern, int(np.ceil(steps / pattern.size)))
    return tiled[:steps].astype(np.int32, copy=False)


def _empty_trace(steps: int) -> TraceData:
    return TraceData(
        reward=np.zeros((steps,), dtype=np.float32),
        terminated=np.zeros((steps,), dtype=np.bool_),
        truncated=np.zeros((steps,), dtype=np.bool_),
        dt=np.zeros((steps,), dtype=np.int32),
        obs=np.zeros((steps, 260), dtype=np.float32),
        info={k: np.zeros((steps,), dtype=np.float32) for k in NUMERIC_INFO_KEYS},
    )


def _run_python_trace(seed: int, time_max: float, actions: np.ndarray) -> TraceData:
    env = ProspectorReferenceEnv(config=ReferenceEnvConfig(time_max=time_max), seed=seed)
    env.reset(seed=seed)

    trace = _empty_trace(actions.size)
    episode_seed = int(seed)

    for t, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(int(action))
        trace.reward[t] = float(reward)
        trace.terminated[t] = bool(terminated)
        trace.truncated[t] = bool(truncated)
        trace.dt[t] = int(info["dt"])
        trace.obs[t] = obs
        for key in NUMERIC_INFO_KEYS:
            trace.info[key][t] = float(info[key])

        if terminated or truncated:
            episode_seed += 1
            env.reset(seed=episode_seed)

    return trace


def _run_native_trace(
    seed: int,
    time_max: float,
    actions: np.ndarray,
    library_path: Path | None,
) -> TraceData:
    cfg = NativeCoreConfig(time_max=time_max, invalid_action_penalty=0.01)

    with NativeProspectorCore(seed=seed, config=cfg, library_path=library_path) as core:
        core.reset(seed)
        trace = _empty_trace(actions.size)
        episode_seed = int(seed)

        for t, action in enumerate(actions):
            obs, reward, terminated, truncated, info = core.step(int(action))
            trace.reward[t] = float(reward)
            trace.terminated[t] = bool(terminated)
            trace.truncated[t] = bool(truncated)
            trace.dt[t] = int(info["dt"])
            trace.obs[t] = obs
            for key in NUMERIC_INFO_KEYS:
                trace.info[key][t] = float(info[key])

            if terminated or truncated:
                episode_seed += 1
                core.reset(episode_seed)

    return trace


def _isclose_with_tol(a: float, b: float, atol: float, rtol: float) -> bool:
    return bool(abs(a - b) <= (atol + rtol * max(abs(a), abs(b))))


def _compare_traces(
    actions: np.ndarray,
    py_trace: TraceData,
    native_trace: TraceData,
    *,
    obs_atol: float,
    obs_rtol: float,
    reward_atol: float,
    reward_rtol: float,
) -> dict[str, object] | None:
    for t in range(actions.size):
        if py_trace.terminated[t] != native_trace.terminated[t]:
            return {
                "step": int(t),
                "field": "terminated",
                "python": bool(py_trace.terminated[t]),
                "native": bool(native_trace.terminated[t]),
            }

        if py_trace.truncated[t] != native_trace.truncated[t]:
            return {
                "step": int(t),
                "field": "truncated",
                "python": bool(py_trace.truncated[t]),
                "native": bool(native_trace.truncated[t]),
            }

        if py_trace.dt[t] != native_trace.dt[t]:
            return {
                "step": int(t),
                "field": "dt",
                "python": int(py_trace.dt[t]),
                "native": int(native_trace.dt[t]),
            }

        py_reward = float(py_trace.reward[t])
        native_reward = float(native_trace.reward[t])
        if not _isclose_with_tol(py_reward, native_reward, reward_atol, reward_rtol):
            return {
                "step": int(t),
                "field": "reward",
                "python": py_reward,
                "native": native_reward,
                "abs_diff": float(abs(py_reward - native_reward)),
            }

        obs_diff = np.abs(py_trace.obs[t] - native_trace.obs[t])
        tol = obs_atol + obs_rtol * np.maximum(np.abs(py_trace.obs[t]), np.abs(native_trace.obs[t]))
        bad = np.where(obs_diff > tol)[0]
        if bad.size > 0:
            idx = int(bad[0])
            return {
                "step": int(t),
                "field": "obs",
                "obs_index": idx,
                "python": float(py_trace.obs[t, idx]),
                "native": float(native_trace.obs[t, idx]),
                "abs_diff": float(obs_diff[idx]),
            }

        for key in NUMERIC_INFO_KEYS:
            py_val = float(py_trace.info[key][t])
            native_val = float(native_trace.info[key][t])
            info_tol = 1.0e-4 * max(1.0, abs(py_val), abs(native_val))
            if abs(py_val - native_val) > info_tol:
                return {
                    "step": int(t),
                    "field": f"info.{key}",
                    "python": py_val,
                    "native": native_val,
                    "abs_diff": float(abs(py_val - native_val)),
                    "tol": float(info_tol),
                }

    return None


def _write_mismatch_bundle(
    bundle_dir: Path,
    suite: str,
    seed: int,
    time_max: float,
    actions: np.ndarray,
    py_trace: TraceData,
    native_trace: TraceData,
    mismatch: dict[str, object],
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    case_dir = bundle_dir / f"{timestamp}_suite-{suite}_seed-{seed}_time-{int(time_max)}"
    case_dir.mkdir(parents=True, exist_ok=True)

    np.save(case_dir / "actions.npy", actions)
    np.savez_compressed(
        case_dir / "python_trace.npz",
        reward=py_trace.reward,
        terminated=py_trace.terminated,
        truncated=py_trace.truncated,
        dt=py_trace.dt,
        obs=py_trace.obs,
        **{f"info_{k}": v for k, v in py_trace.info.items()},
    )
    np.savez_compressed(
        case_dir / "native_trace.npz",
        reward=native_trace.reward,
        terminated=native_trace.terminated,
        truncated=native_trace.truncated,
        dt=native_trace.dt,
        obs=native_trace.obs,
        **{f"info_{k}": v for k, v in native_trace.info.items()},
    )

    payload = {
        "suite": suite,
        "seed": int(seed),
        "time_max": float(time_max),
        "mismatch": mismatch,
    }
    (case_dir / "mismatch.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return case_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Python-vs-native parity harness.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Initial seed.")
    parser.add_argument("--steps", type=int, default=2000, help="Steps per case.")
    parser.add_argument(
        "--suite",
        action="append",
        choices=("A", "B", "C"),
        help="Action suites to run (repeat flag for multiple). Defaults to A,B,C.",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        nargs="+",
        default=[2000.0, 8000.0],
        help="One or more time_max configs to test.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/parity/mismatch_bundles"),
        help="Mismatch bundle directory.",
    )
    parser.add_argument("--obs-atol", type=float, default=1e-6)
    parser.add_argument("--obs-rtol", type=float, default=1e-5)
    parser.add_argument("--reward-atol", type=float, default=1e-6)
    parser.add_argument("--reward-rtol", type=float, default=1e-5)
    parser.add_argument("--stop-on-first", action="store_true")
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument(
        "--native-library",
        type=Path,
        default=None,
        help="Optional explicit path to abp_core shared library.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    suites = args.suite if args.suite else ["A", "B", "C"]
    seed_values = [args.seed_start + i for i in range(args.seeds)]

    total_cases = 0
    failed_cases = 0

    for time_max in args.time_max:
        for suite in suites:
            for seed in seed_values:
                total_cases += 1
                actions = _generate_actions(suite, args.steps, seed + 1000)

                py_trace = _run_python_trace(seed, time_max, actions)
                native_trace = _run_native_trace(seed, time_max, actions, args.native_library)

                mismatch = _compare_traces(
                    actions,
                    py_trace,
                    native_trace,
                    obs_atol=args.obs_atol,
                    obs_rtol=args.obs_rtol,
                    reward_atol=args.reward_atol,
                    reward_rtol=args.reward_rtol,
                )

                if mismatch is None:
                    print(
                        f"PASS suite={suite} seed={seed} time_max={time_max:g} steps={args.steps}"
                    )
                    continue

                failed_cases += 1
                bundle_path = _write_mismatch_bundle(
                    args.bundle_dir,
                    suite,
                    seed,
                    time_max,
                    actions,
                    py_trace,
                    native_trace,
                    mismatch,
                )
                print(
                    "FAIL "
                    f"suite={suite} seed={seed} time_max={time_max:g} "
                    f"step={mismatch.get('step')} field={mismatch.get('field')} "
                    f"bundle={bundle_path}"
                )

                if args.stop_on_first:
                    summary = f"Stopped after first mismatch ({failed_cases}/{total_cases} failed)."
                    print(summary)
                    return 0 if args.allow_mismatch else 1

    print(f"Completed {total_cases} parity cases. Failed: {failed_cases}.")

    if failed_cases > 0 and not args.allow_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
