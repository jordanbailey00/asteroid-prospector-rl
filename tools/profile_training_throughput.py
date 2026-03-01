from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from training import TrainConfig, run_training

PROFILE_MODE_ENV_ONLY = "env_only"
PROFILE_MODE_TRAINER = "trainer"
PROFILE_MODE_TRAINER_EVAL = "trainer_eval"
SUPPORTED_PROFILE_MODES = (
    PROFILE_MODE_ENV_ONLY,
    PROFILE_MODE_TRAINER,
    PROFILE_MODE_TRAINER_EVAL,
)

ENV_IMPL_AUTO = "auto"
ENV_IMPL_REFERENCE = "reference"
ENV_IMPL_NATIVE = "native"
SUPPORTED_ENV_IMPLS = (ENV_IMPL_AUTO, ENV_IMPL_REFERENCE, ENV_IMPL_NATIVE)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))

    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _parse_modes_csv(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if text == "":
        return tuple()

    modes: list[str] = []
    for item in text.split(","):
        mode = item.strip().lower()
        if mode == "":
            continue
        if mode not in SUPPORTED_PROFILE_MODES:
            raise ValueError(
                f"Unsupported profile mode: {mode!r}. "
                f"Supported: {', '.join(SUPPORTED_PROFILE_MODES)}"
            )
        modes.append(mode)

    deduped = tuple(dict.fromkeys(modes))
    return deduped


def _to_relative_posix(path: Path, *, start: Path) -> str:
    try:
        return path.relative_to(start).as_posix()
    except ValueError:
        return path.as_posix()


def _measure_call(callable_fn: Any) -> tuple[Any, dict[str, float]]:
    gc.collect()
    tracemalloc.start()
    start_current, _ = tracemalloc.get_traced_memory()
    start_wall = perf_counter()
    start_cpu = process_time()

    result = callable_fn()

    end_wall = perf_counter()
    end_cpu = process_time()
    end_current, end_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_seconds = end_wall - start_wall
    cpu_seconds = end_cpu - start_cpu
    cpu_utilization_pct = 0.0
    if wall_seconds > 0.0:
        cpu_utilization_pct = 100.0 * cpu_seconds / wall_seconds

    mib = 1024.0 * 1024.0
    return result, {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_utilization_pct": cpu_utilization_pct,
        "py_heap_current_growth_mb": float(end_current - start_current) / mib,
        "py_heap_peak_growth_mb": float(end_peak - start_current) / mib,
    }


@dataclass(frozen=True)
class ThroughputProfileConfig:
    run_root: Path = Path("artifacts/throughput/runs")
    output_path: Path | None = None
    run_id: str | None = None
    run_id_prefix: str = "throughput"

    seed: int = 17
    env_time_max: float = 20000.0

    modes: tuple[str, ...] = (PROFILE_MODE_ENV_ONLY, PROFILE_MODE_TRAINER)
    env_impl: str = ENV_IMPL_AUTO
    env_duration_seconds: float = 2.0
    env_repeats: int = 3

    trainer_backend: str = "random"
    trainer_total_env_steps: int = 3000
    trainer_window_env_steps: int = 1000
    trainer_eval_replays_per_window: int = 1
    trainer_eval_max_steps_per_episode: int = 256
    trainer_repeats: int = 3

    target_steps_per_sec: float = 100000.0
    enforce_target: bool = False


def _validate_config(cfg: ThroughputProfileConfig) -> None:
    if not cfg.modes:
        raise ValueError("At least one mode must be selected.")
    if cfg.env_impl not in SUPPORTED_ENV_IMPLS:
        raise ValueError(
            f"Unsupported env_impl: {cfg.env_impl!r}. "
            f"Supported: {', '.join(SUPPORTED_ENV_IMPLS)}"
        )
    if cfg.env_duration_seconds <= 0.0:
        raise ValueError("env_duration_seconds must be positive.")
    if cfg.env_repeats <= 0:
        raise ValueError("env_repeats must be positive.")
    if cfg.trainer_total_env_steps <= 0:
        raise ValueError("trainer_total_env_steps must be positive.")
    if cfg.trainer_window_env_steps <= 0:
        raise ValueError("trainer_window_env_steps must be positive.")
    if cfg.trainer_repeats <= 0:
        raise ValueError("trainer_repeats must be positive.")
    if cfg.target_steps_per_sec <= 0.0:
        raise ValueError("target_steps_per_sec must be positive.")
    if cfg.trainer_eval_replays_per_window < 0:
        raise ValueError("trainer_eval_replays_per_window must be non-negative.")
    if cfg.trainer_eval_max_steps_per_episode <= 0:
        raise ValueError("trainer_eval_max_steps_per_episode must be positive.")


def _profile_env_only(cfg: ThroughputProfileConfig) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    python_src = repo_root / "python"
    if str(python_src) not in sys.path:
        sys.path.insert(0, str(python_src))

    from asteroid_prospector import (
        N_ACTIONS,
        NativeCoreConfig,
        NativeProspectorCore,
        ProspectorReferenceEnv,
        ReferenceEnvConfig,
    )

    env_impl_selected = cfg.env_impl
    if env_impl_selected == ENV_IMPL_AUTO:
        # Native first when available; reference fallback if DLL is not present.
        try:
            maybe_native = NativeProspectorCore(
                seed=cfg.seed,
                config=NativeCoreConfig(time_max=cfg.env_time_max),
            )
            maybe_native.close()
            env_impl_selected = ENV_IMPL_NATIVE
        except FileNotFoundError:
            env_impl_selected = ENV_IMPL_REFERENCE

    def env_ctor(seed: int) -> Any:
        if env_impl_selected == ENV_IMPL_NATIVE:
            return NativeProspectorCore(
                seed=seed,
                config=NativeCoreConfig(time_max=cfg.env_time_max),
            )
        return ProspectorReferenceEnv(
            config=ReferenceEnvConfig(time_max=cfg.env_time_max),
            seed=seed,
        )

    samples_steps_per_sec: list[float] = []
    samples_wall_seconds: list[float] = []
    samples_cpu_seconds: list[float] = []
    samples_cpu_utilization_pct: list[float] = []
    samples_heap_peak_mb: list[float] = []

    for repeat_idx in range(cfg.env_repeats):
        sample_seed = int(cfg.seed + repeat_idx * 1009)
        rng = np.random.default_rng(sample_seed + 33)

        def _run_sample(*, _sample_seed: int = sample_seed, _rng: Any = rng) -> dict[str, Any]:
            env = env_ctor(_sample_seed)
            obs, _info = env.reset(seed=_sample_seed)
            del obs

            steps = 0
            episode_seed = _sample_seed
            sample_start = perf_counter()
            try:
                while True:
                    elapsed = perf_counter() - sample_start
                    if elapsed >= cfg.env_duration_seconds:
                        break

                    action = int(_rng.integers(0, N_ACTIONS))
                    _obs, _reward, terminated, truncated, _info = env.step(action)
                    steps += 1
                    if terminated or truncated:
                        episode_seed += 1
                        env.reset(seed=episode_seed)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

            return {"steps": int(steps)}

        sample_result, perf = _measure_call(lambda: _run_sample())
        wall_seconds = float(perf["wall_seconds"])
        steps = int(sample_result["steps"])
        steps_per_sec = float(steps) / wall_seconds if wall_seconds > 0.0 else 0.0

        samples_steps_per_sec.append(steps_per_sec)
        samples_wall_seconds.append(wall_seconds)
        samples_cpu_seconds.append(float(perf["cpu_seconds"]))
        samples_cpu_utilization_pct.append(float(perf["cpu_utilization_pct"]))
        samples_heap_peak_mb.append(float(perf["py_heap_peak_growth_mb"]))

    return {
        "mode": PROFILE_MODE_ENV_ONLY,
        "env_impl_requested": cfg.env_impl,
        "env_impl_selected": env_impl_selected,
        "duration_seconds": float(cfg.env_duration_seconds),
        "samples": cfg.env_repeats,
        "steps_per_sec_samples": samples_steps_per_sec,
        "steps_per_sec_stats": _stats(samples_steps_per_sec),
        "wall_seconds_stats": _stats(samples_wall_seconds),
        "cpu_seconds_stats": _stats(samples_cpu_seconds),
        "cpu_utilization_pct_stats": _stats(samples_cpu_utilization_pct),
        "py_heap_peak_growth_mb_stats": _stats(samples_heap_peak_mb),
    }


def _profile_trainer_mode(
    cfg: ThroughputProfileConfig, *, mode: str, run_id: str
) -> dict[str, Any]:
    run_root = cfg.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    samples_steps_per_sec: list[float] = []
    samples_wall_seconds: list[float] = []
    samples_cpu_seconds: list[float] = []
    samples_cpu_utilization_pct: list[float] = []
    samples_heap_peak_mb: list[float] = []
    run_records: list[dict[str, Any]] = []

    for repeat_idx in range(cfg.trainer_repeats):
        sample_seed = int(cfg.seed + repeat_idx * 2027)
        sample_run_id = f"{run_id}-{mode}-r{repeat_idx + 1:02d}"
        eval_replays_per_window = 0
        if mode == PROFILE_MODE_TRAINER_EVAL:
            eval_replays_per_window = cfg.trainer_eval_replays_per_window

        train_cfg = TrainConfig(
            run_root=run_root,
            run_id=sample_run_id,
            total_env_steps=int(cfg.trainer_total_env_steps),
            window_env_steps=int(cfg.trainer_window_env_steps),
            checkpoint_every_windows=1,
            seed=sample_seed,
            env_time_max=float(cfg.env_time_max),
            trainer_backend=cfg.trainer_backend,
            wandb_mode="disabled",
            eval_replays_per_window=int(eval_replays_per_window),
            eval_max_steps_per_episode=int(cfg.trainer_eval_max_steps_per_episode),
            eval_include_info=False,
        )

        def _run_sample_training(*, _train_cfg: TrainConfig = train_cfg) -> dict[str, Any]:
            return run_training(_train_cfg)

        summary, perf = _measure_call(_run_sample_training)
        if not isinstance(summary, dict):
            raise RuntimeError("run_training returned unexpected payload.")

        env_steps = int(summary.get("env_steps_total", 0))
        wall_seconds = float(perf["wall_seconds"])
        steps_per_sec = float(env_steps) / wall_seconds if wall_seconds > 0.0 else 0.0

        run_dir = run_root / sample_run_id
        run_records.append(
            {
                "run_id": sample_run_id,
                "run_dir": _to_relative_posix(run_dir, start=run_root),
                "env_steps_total": env_steps,
                "steps_per_sec": steps_per_sec,
                "wall_seconds": wall_seconds,
            }
        )

        samples_steps_per_sec.append(steps_per_sec)
        samples_wall_seconds.append(wall_seconds)
        samples_cpu_seconds.append(float(perf["cpu_seconds"]))
        samples_cpu_utilization_pct.append(float(perf["cpu_utilization_pct"]))
        samples_heap_peak_mb.append(float(perf["py_heap_peak_growth_mb"]))

    return {
        "mode": mode,
        "trainer_backend": cfg.trainer_backend,
        "trainer_total_env_steps": int(cfg.trainer_total_env_steps),
        "trainer_window_env_steps": int(cfg.trainer_window_env_steps),
        "eval_replays_per_window": int(
            cfg.trainer_eval_replays_per_window if mode == PROFILE_MODE_TRAINER_EVAL else 0
        ),
        "eval_max_steps_per_episode": int(cfg.trainer_eval_max_steps_per_episode),
        "samples": cfg.trainer_repeats,
        "steps_per_sec_samples": samples_steps_per_sec,
        "steps_per_sec_stats": _stats(samples_steps_per_sec),
        "wall_seconds_stats": _stats(samples_wall_seconds),
        "cpu_seconds_stats": _stats(samples_cpu_seconds),
        "cpu_utilization_pct_stats": _stats(samples_cpu_utilization_pct),
        "py_heap_peak_growth_mb_stats": _stats(samples_heap_peak_mb),
        "runs": run_records,
    }


def _serialize_config(cfg: ThroughputProfileConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["run_root"] = cfg.run_root.as_posix()
    payload["output_path"] = cfg.output_path.as_posix() if cfg.output_path is not None else None
    payload["modes"] = list(cfg.modes)
    return payload


def run_training_throughput_profile(cfg: ThroughputProfileConfig) -> dict[str, Any]:
    _validate_config(cfg)

    run_id = cfg.run_id or _default_run_id(cfg.run_id_prefix)
    mode_reports: list[dict[str, Any]] = []

    for mode in cfg.modes:
        if mode == PROFILE_MODE_ENV_ONLY:
            mode_reports.append(_profile_env_only(cfg))
            continue
        if mode in {PROFILE_MODE_TRAINER, PROFILE_MODE_TRAINER_EVAL}:
            mode_reports.append(_profile_trainer_mode(cfg, mode=mode, run_id=run_id))
            continue
        raise ValueError(f"Unsupported mode: {mode!r}")

    threshold_failures: list[str] = []
    if cfg.enforce_target:
        target = float(cfg.target_steps_per_sec)
        for mode_report in mode_reports:
            mode_name = str(mode_report.get("mode", "unknown"))
            stats = mode_report.get("steps_per_sec_stats", {})
            mean_steps = float(stats.get("mean", 0.0))
            if mean_steps < target:
                threshold_failures.append(
                    f"{mode_name} mean steps/sec {mean_steps:.3f} below target {target:.3f}"
                )

    output_path = cfg.output_path
    if output_path is None:
        output_path = Path("artifacts/throughput") / f"{run_id}.json"

    report = {
        "generated_at": now_iso(),
        "run_id": run_id,
        "config": _serialize_config(cfg),
        "summary": {
            "pass": len(threshold_failures) == 0,
            "threshold_failures": threshold_failures,
            "target_steps_per_sec": float(cfg.target_steps_per_sec),
            "enforce_target": bool(cfg.enforce_target),
        },
        "system": {
            "python_version": sys.version.split()[0],
            "cpu_count_logical": int(os.cpu_count() or 0),
            "platform": sys.platform,
        },
        "modes": mode_reports,
        "artifacts": {
            "run_root": cfg.run_root.as_posix(),
            "report_path": output_path.as_posix(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> ThroughputProfileConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Profile training throughput across env-only and trainer flows. "
            "Supports optional threshold enforcement against target steps/sec."
        )
    )

    parser.add_argument("--run-root", type=Path, default=Path("artifacts/throughput/runs"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-id-prefix", type=str, default="throughput")

    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--env-time-max", type=float, default=20000.0)
    parser.add_argument("--modes", type=str, default="env_only,trainer")
    parser.add_argument("--env-impl", choices=list(SUPPORTED_ENV_IMPLS), default=ENV_IMPL_AUTO)
    parser.add_argument("--env-duration-seconds", type=float, default=2.0)
    parser.add_argument("--env-repeats", type=int, default=3)

    parser.add_argument("--trainer-backend", choices=["random", "puffer_ppo"], default="random")
    parser.add_argument("--trainer-total-env-steps", type=int, default=3000)
    parser.add_argument("--trainer-window-env-steps", type=int, default=1000)
    parser.add_argument("--trainer-eval-replays-per-window", type=int, default=1)
    parser.add_argument("--trainer-eval-max-steps-per-episode", type=int, default=256)
    parser.add_argument("--trainer-repeats", type=int, default=3)

    parser.add_argument("--target-steps-per-sec", type=float, default=100000.0)
    parser.add_argument("--enforce-target", action="store_true")

    args = parser.parse_args()
    modes = _parse_modes_csv(args.modes)

    return ThroughputProfileConfig(
        run_root=args.run_root,
        output_path=args.output_path,
        run_id=args.run_id,
        run_id_prefix=args.run_id_prefix,
        seed=args.seed,
        env_time_max=args.env_time_max,
        modes=modes,
        env_impl=args.env_impl,
        env_duration_seconds=args.env_duration_seconds,
        env_repeats=args.env_repeats,
        trainer_backend=args.trainer_backend,
        trainer_total_env_steps=args.trainer_total_env_steps,
        trainer_window_env_steps=args.trainer_window_env_steps,
        trainer_eval_replays_per_window=args.trainer_eval_replays_per_window,
        trainer_eval_max_steps_per_episode=args.trainer_eval_max_steps_per_episode,
        trainer_repeats=args.trainer_repeats,
        target_steps_per_sec=args.target_steps_per_sec,
        enforce_target=args.enforce_target,
    )


def main() -> int:
    cfg = _parse_args()
    report = run_training_throughput_profile(cfg)
    print(json.dumps(report, indent=2))
    summary = report.get("summary", {})
    return 0 if bool(summary.get("pass", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
