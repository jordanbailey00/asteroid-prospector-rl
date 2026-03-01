from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from tools.profile_training_throughput import (
    ENV_IMPL_AUTO,
    ENV_IMPL_NATIVE,
    ENV_IMPL_REFERENCE,
    PROFILE_MODE_ENV_ONLY,
    PROFILE_MODE_TRAINER,
    PROFILE_MODE_TRAINER_EVAL,
    SUPPORTED_PROFILE_MODES,
    ThroughputProfileConfig,
    run_training_throughput_profile,
)

SUPPORTED_TRAINER_BACKENDS = ("random", "puffer_ppo")
SUPPORTED_PPO_VECTOR_BACKENDS = ("serial", "multiprocessing")
SUPPORTED_PPO_ENV_IMPLS = ("reference", "native", "auto")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _parse_csv(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if text == "":
        return tuple()
    parts = [item.strip() for item in text.split(",") if item.strip() != ""]
    return tuple(dict.fromkeys(parts))


def _parse_int_csv(raw: str, *, field_name: str) -> tuple[int, ...]:
    parts = _parse_csv(raw)
    values: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {field_name}: {part!r}") from exc
        values.append(value)
    return tuple(values)


def _validate_choices(*, name: str, values: tuple[str, ...], supported: tuple[str, ...]) -> None:
    unsupported = [value for value in values if value not in supported]
    if unsupported:
        raise ValueError(
            f"Unsupported {name}: {', '.join(repr(value) for value in unsupported)}. "
            f"Supported: {', '.join(supported)}"
        )


def _as_posix(path: Path) -> str:
    return path.as_posix()


@dataclass(frozen=True)
class ThroughputMatrixConfig:
    run_root: Path = Path("artifacts/throughput/runs")
    output_path: Path | None = None
    run_id: str | None = None
    run_id_prefix: str = "throughput-matrix"

    seed: int = 17
    env_time_max: float = 20000.0

    modes: tuple[str, ...] = (PROFILE_MODE_ENV_ONLY, PROFILE_MODE_TRAINER)
    env_impls: tuple[str, ...] = (ENV_IMPL_NATIVE, ENV_IMPL_REFERENCE)
    trainer_backends: tuple[str, ...] = ("random",)

    env_duration_seconds: float = 2.0
    env_repeats: int = 3

    trainer_total_env_steps: int = 3000
    trainer_window_env_steps: int = 1000
    trainer_eval_replays_per_window: int = 1
    trainer_eval_max_steps_per_episode: int = 256
    trainer_repeats: int = 3

    ppo_num_envs_values: tuple[int, ...] = (8,)
    ppo_num_workers_values: tuple[int, ...] = (4,)
    ppo_rollout_steps_values: tuple[int, ...] = (128,)
    ppo_num_minibatches_values: tuple[int, ...] = (4,)
    ppo_update_epochs_values: tuple[int, ...] = (4,)
    ppo_vector_backends: tuple[str, ...] = ("multiprocessing",)
    ppo_env_impls: tuple[str, ...] = ("auto",)

    target_steps_per_sec: float = 100000.0
    enforce_target: bool = False

    floor_safety_factor: float = 0.90
    fail_on_candidate_error: bool = False


@dataclass(frozen=True)
class MatrixCandidate:
    candidate_id: str
    mode: str
    env_impl: str
    trainer_backend: str
    ppo_num_envs: int
    ppo_num_workers: int
    ppo_rollout_steps: int
    ppo_num_minibatches: int
    ppo_update_epochs: int
    ppo_vector_backend: str
    ppo_env_impl: str
    skip_reason: str | None = None


def _validate_config(cfg: ThroughputMatrixConfig) -> None:
    if not cfg.modes:
        raise ValueError("At least one mode is required.")
    _validate_choices(name="modes", values=cfg.modes, supported=SUPPORTED_PROFILE_MODES)
    _validate_choices(
        name="env_impls",
        values=cfg.env_impls,
        supported=(ENV_IMPL_AUTO, ENV_IMPL_REFERENCE, ENV_IMPL_NATIVE),
    )
    _validate_choices(
        name="trainer_backends",
        values=cfg.trainer_backends,
        supported=SUPPORTED_TRAINER_BACKENDS,
    )
    _validate_choices(
        name="ppo_vector_backends",
        values=cfg.ppo_vector_backends,
        supported=SUPPORTED_PPO_VECTOR_BACKENDS,
    )
    _validate_choices(
        name="ppo_env_impls",
        values=cfg.ppo_env_impls,
        supported=SUPPORTED_PPO_ENV_IMPLS,
    )

    if PROFILE_MODE_ENV_ONLY in cfg.modes and not cfg.env_impls:
        raise ValueError("env_impls must be non-empty when env_only mode is selected.")
    if (
        PROFILE_MODE_TRAINER in cfg.modes or PROFILE_MODE_TRAINER_EVAL in cfg.modes
    ) and not cfg.trainer_backends:
        raise ValueError("trainer_backends must be non-empty when trainer modes are selected.")

    if cfg.env_duration_seconds <= 0.0:
        raise ValueError("env_duration_seconds must be positive.")
    if cfg.env_repeats <= 0:
        raise ValueError("env_repeats must be positive.")
    if cfg.trainer_total_env_steps <= 0:
        raise ValueError("trainer_total_env_steps must be positive.")
    if cfg.trainer_window_env_steps <= 0:
        raise ValueError("trainer_window_env_steps must be positive.")
    if cfg.trainer_eval_replays_per_window < 0:
        raise ValueError("trainer_eval_replays_per_window must be non-negative.")
    if cfg.trainer_eval_max_steps_per_episode <= 0:
        raise ValueError("trainer_eval_max_steps_per_episode must be positive.")
    if cfg.trainer_repeats <= 0:
        raise ValueError("trainer_repeats must be positive.")

    int_fields: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("ppo_num_envs_values", cfg.ppo_num_envs_values),
        ("ppo_num_workers_values", cfg.ppo_num_workers_values),
        ("ppo_rollout_steps_values", cfg.ppo_rollout_steps_values),
        ("ppo_num_minibatches_values", cfg.ppo_num_minibatches_values),
        ("ppo_update_epochs_values", cfg.ppo_update_epochs_values),
    )
    for field_name, values in int_fields:
        if not values:
            raise ValueError(f"{field_name} must be non-empty.")
        if any(value <= 0 for value in values):
            raise ValueError(f"{field_name} must contain only positive integers.")

    if cfg.target_steps_per_sec <= 0.0:
        raise ValueError("target_steps_per_sec must be positive.")
    if not (0.0 < cfg.floor_safety_factor <= 1.0):
        raise ValueError("floor_safety_factor must be in the interval (0.0, 1.0].")


def _candidate_id(
    *,
    mode: str,
    env_impl: str,
    trainer_backend: str,
    ppo_num_envs: int,
    ppo_num_workers: int,
    ppo_rollout_steps: int,
    ppo_num_minibatches: int,
    ppo_update_epochs: int,
    ppo_vector_backend: str,
    ppo_env_impl: str,
) -> str:
    if mode == PROFILE_MODE_ENV_ONLY:
        return f"{mode}-env_{env_impl}"

    if trainer_backend != "puffer_ppo":
        return f"{mode}-backend_{trainer_backend}"

    return (
        f"{mode}-backend_{trainer_backend}"
        f"-ne{ppo_num_envs}"
        f"-nw{ppo_num_workers}"
        f"-rs{ppo_rollout_steps}"
        f"-mb{ppo_num_minibatches}"
        f"-ue{ppo_update_epochs}"
        f"-vb_{ppo_vector_backend}"
        f"-ei_{ppo_env_impl}"
    )


def _build_candidates(cfg: ThroughputMatrixConfig) -> list[MatrixCandidate]:
    candidates: list[MatrixCandidate] = []

    for mode in cfg.modes:
        if mode == PROFILE_MODE_ENV_ONLY:
            for env_impl in cfg.env_impls:
                candidates.append(
                    MatrixCandidate(
                        candidate_id=_candidate_id(
                            mode=mode,
                            env_impl=env_impl,
                            trainer_backend="random",
                            ppo_num_envs=cfg.ppo_num_envs_values[0],
                            ppo_num_workers=cfg.ppo_num_workers_values[0],
                            ppo_rollout_steps=cfg.ppo_rollout_steps_values[0],
                            ppo_num_minibatches=cfg.ppo_num_minibatches_values[0],
                            ppo_update_epochs=cfg.ppo_update_epochs_values[0],
                            ppo_vector_backend=cfg.ppo_vector_backends[0],
                            ppo_env_impl=cfg.ppo_env_impls[0],
                        ),
                        mode=mode,
                        env_impl=env_impl,
                        trainer_backend="random",
                        ppo_num_envs=cfg.ppo_num_envs_values[0],
                        ppo_num_workers=cfg.ppo_num_workers_values[0],
                        ppo_rollout_steps=cfg.ppo_rollout_steps_values[0],
                        ppo_num_minibatches=cfg.ppo_num_minibatches_values[0],
                        ppo_update_epochs=cfg.ppo_update_epochs_values[0],
                        ppo_vector_backend=cfg.ppo_vector_backends[0],
                        ppo_env_impl=cfg.ppo_env_impls[0],
                    )
                )
            continue

        if mode not in {PROFILE_MODE_TRAINER, PROFILE_MODE_TRAINER_EVAL}:
            continue

        for trainer_backend in cfg.trainer_backends:
            if trainer_backend != "puffer_ppo":
                candidates.append(
                    MatrixCandidate(
                        candidate_id=_candidate_id(
                            mode=mode,
                            env_impl=ENV_IMPL_AUTO,
                            trainer_backend=trainer_backend,
                            ppo_num_envs=cfg.ppo_num_envs_values[0],
                            ppo_num_workers=cfg.ppo_num_workers_values[0],
                            ppo_rollout_steps=cfg.ppo_rollout_steps_values[0],
                            ppo_num_minibatches=cfg.ppo_num_minibatches_values[0],
                            ppo_update_epochs=cfg.ppo_update_epochs_values[0],
                            ppo_vector_backend=cfg.ppo_vector_backends[0],
                            ppo_env_impl=cfg.ppo_env_impls[0],
                        ),
                        mode=mode,
                        env_impl=ENV_IMPL_AUTO,
                        trainer_backend=trainer_backend,
                        ppo_num_envs=cfg.ppo_num_envs_values[0],
                        ppo_num_workers=cfg.ppo_num_workers_values[0],
                        ppo_rollout_steps=cfg.ppo_rollout_steps_values[0],
                        ppo_num_minibatches=cfg.ppo_num_minibatches_values[0],
                        ppo_update_epochs=cfg.ppo_update_epochs_values[0],
                        ppo_vector_backend=cfg.ppo_vector_backends[0],
                        ppo_env_impl=cfg.ppo_env_impls[0],
                    )
                )
                continue

            for (
                ppo_num_envs,
                ppo_num_workers,
                ppo_rollout_steps,
                ppo_num_minibatches,
                ppo_update_epochs,
                ppo_vector_backend,
                ppo_env_impl,
            ) in itertools.product(
                cfg.ppo_num_envs_values,
                cfg.ppo_num_workers_values,
                cfg.ppo_rollout_steps_values,
                cfg.ppo_num_minibatches_values,
                cfg.ppo_update_epochs_values,
                cfg.ppo_vector_backends,
                cfg.ppo_env_impls,
            ):
                skip_reason: str | None = None
                if ppo_vector_backend == "multiprocessing" and ppo_num_envs % ppo_num_workers != 0:
                    skip_reason = (
                        "invalid puffer_ppo config: ppo_num_envs must be divisible by "
                        "ppo_num_workers for multiprocessing vector backend"
                    )
                elif sys.platform.startswith("win"):
                    skip_reason = (
                        "puffer_ppo trainer backend requires Linux runtime (Docker/WSL), "
                        "native Windows execution skipped"
                    )

                candidates.append(
                    MatrixCandidate(
                        candidate_id=_candidate_id(
                            mode=mode,
                            env_impl=ENV_IMPL_AUTO,
                            trainer_backend=trainer_backend,
                            ppo_num_envs=int(ppo_num_envs),
                            ppo_num_workers=int(ppo_num_workers),
                            ppo_rollout_steps=int(ppo_rollout_steps),
                            ppo_num_minibatches=int(ppo_num_minibatches),
                            ppo_update_epochs=int(ppo_update_epochs),
                            ppo_vector_backend=str(ppo_vector_backend),
                            ppo_env_impl=str(ppo_env_impl),
                        ),
                        mode=mode,
                        env_impl=ENV_IMPL_AUTO,
                        trainer_backend=trainer_backend,
                        ppo_num_envs=int(ppo_num_envs),
                        ppo_num_workers=int(ppo_num_workers),
                        ppo_rollout_steps=int(ppo_rollout_steps),
                        ppo_num_minibatches=int(ppo_num_minibatches),
                        ppo_update_epochs=int(ppo_update_epochs),
                        ppo_vector_backend=str(ppo_vector_backend),
                        ppo_env_impl=str(ppo_env_impl),
                        skip_reason=skip_reason,
                    )
                )

    deduped: list[MatrixCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate.candidate_id)
    return deduped


def _serialize_config(cfg: ThroughputMatrixConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["run_root"] = _as_posix(cfg.run_root)
    payload["output_path"] = _as_posix(cfg.output_path) if cfg.output_path is not None else None

    for key in (
        "modes",
        "env_impls",
        "trainer_backends",
        "ppo_num_envs_values",
        "ppo_num_workers_values",
        "ppo_rollout_steps_values",
        "ppo_num_minibatches_values",
        "ppo_update_epochs_values",
        "ppo_vector_backends",
        "ppo_env_impls",
    ):
        payload[key] = list(payload[key])
    return payload


def _build_mode_summary(
    *,
    mode: str,
    successful_candidates: list[dict[str, Any]],
    target_steps_per_sec: float,
    floor_safety_factor: float,
) -> dict[str, Any]:
    if not successful_candidates:
        return {
            "mode": mode,
            "successful_candidates": 0,
            "best_candidate_id": None,
            "best_mean_steps_per_sec": 0.0,
            "best_min_steps_per_sec": 0.0,
            "recommended_floor_steps_per_sec": 0.0,
            "floor_safety_factor": float(floor_safety_factor),
            "target_steps_per_sec": float(target_steps_per_sec),
            "delta_to_target_steps_per_sec": float(target_steps_per_sec),
            "target_attained": False,
        }

    ranked = sorted(
        successful_candidates,
        key=lambda item: float(item.get("steps_per_sec_mean", 0.0)),
        reverse=True,
    )
    best = ranked[0]
    best_mean = float(best.get("steps_per_sec_mean", 0.0))
    best_min = float(best.get("steps_per_sec_min", 0.0))
    recommended_floor = max(0.0, best_min * float(floor_safety_factor))

    return {
        "mode": mode,
        "successful_candidates": len(successful_candidates),
        "best_candidate_id": best.get("candidate_id"),
        "best_mean_steps_per_sec": best_mean,
        "best_min_steps_per_sec": best_min,
        "recommended_floor_steps_per_sec": recommended_floor,
        "floor_safety_factor": float(floor_safety_factor),
        "target_steps_per_sec": float(target_steps_per_sec),
        "delta_to_target_steps_per_sec": float(target_steps_per_sec) - best_mean,
        "target_attained": best_mean >= float(target_steps_per_sec),
        "top_candidates": [
            {
                "candidate_id": item.get("candidate_id"),
                "steps_per_sec_mean": float(item.get("steps_per_sec_mean", 0.0)),
                "steps_per_sec_min": float(item.get("steps_per_sec_min", 0.0)),
            }
            for item in ranked[:3]
        ],
    }


def run_throughput_matrix(cfg: ThroughputMatrixConfig) -> dict[str, Any]:
    _validate_config(cfg)

    run_id = cfg.run_id or _default_run_id(cfg.run_id_prefix)
    output_path = cfg.output_path
    if output_path is None:
        output_path = Path("artifacts/throughput") / f"{run_id}.json"

    if output_path.exists():
        raise ValueError(f"Matrix output path already exists: {output_path.as_posix()}")

    matrix_root = Path("artifacts/throughput/matrix") / run_id
    candidate_run_root = cfg.run_root / run_id
    candidates = _build_candidates(cfg)

    candidate_reports: list[dict[str, Any]] = []
    target_failures: list[str] = []

    for idx, candidate in enumerate(candidates):
        candidate_record: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "mode": candidate.mode,
            "env_impl": candidate.env_impl,
            "trainer_backend": candidate.trainer_backend,
            "ppo_num_envs": candidate.ppo_num_envs,
            "ppo_num_workers": candidate.ppo_num_workers,
            "ppo_rollout_steps": candidate.ppo_rollout_steps,
            "ppo_num_minibatches": candidate.ppo_num_minibatches,
            "ppo_update_epochs": candidate.ppo_update_epochs,
            "ppo_vector_backend": candidate.ppo_vector_backend,
            "ppo_env_impl": candidate.ppo_env_impl,
        }

        if candidate.skip_reason is not None:
            candidate_record.update({"status": "skipped", "skip_reason": candidate.skip_reason})
            candidate_reports.append(candidate_record)
            continue

        candidate_profile_output = matrix_root / f"{candidate.candidate_id}.json"
        profile_cfg = ThroughputProfileConfig(
            run_root=candidate_run_root,
            output_path=candidate_profile_output,
            run_id=f"{run_id}-{idx + 1:03d}-{candidate.candidate_id}",
            run_id_prefix="throughput",
            seed=int(cfg.seed),
            env_time_max=float(cfg.env_time_max),
            modes=(candidate.mode,),
            env_impl=candidate.env_impl,
            env_duration_seconds=float(cfg.env_duration_seconds),
            env_repeats=int(cfg.env_repeats),
            trainer_backend=candidate.trainer_backend,
            trainer_total_env_steps=int(cfg.trainer_total_env_steps),
            trainer_window_env_steps=int(cfg.trainer_window_env_steps),
            trainer_eval_replays_per_window=int(cfg.trainer_eval_replays_per_window),
            trainer_eval_max_steps_per_episode=int(cfg.trainer_eval_max_steps_per_episode),
            trainer_repeats=int(cfg.trainer_repeats),
            ppo_num_envs=int(candidate.ppo_num_envs),
            ppo_num_workers=int(candidate.ppo_num_workers),
            ppo_rollout_steps=int(candidate.ppo_rollout_steps),
            ppo_num_minibatches=int(candidate.ppo_num_minibatches),
            ppo_update_epochs=int(candidate.ppo_update_epochs),
            ppo_vector_backend=candidate.ppo_vector_backend,
            ppo_env_impl=candidate.ppo_env_impl,
            target_steps_per_sec=float(cfg.target_steps_per_sec),
            enforce_target=bool(cfg.enforce_target),
        )

        try:
            profile_report = run_training_throughput_profile(profile_cfg)
        except Exception as exc:
            candidate_record.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "profile_output_path": _as_posix(candidate_profile_output),
                }
            )
            candidate_reports.append(candidate_record)
            if cfg.fail_on_candidate_error:
                raise
            continue

        mode_reports = profile_report.get("modes", [])
        mode_report = mode_reports[0] if isinstance(mode_reports, list) and mode_reports else {}
        stats = mode_report.get("steps_per_sec_stats", {})
        summary = profile_report.get("summary", {})

        threshold_failures_raw = summary.get("threshold_failures", [])
        threshold_failures = (
            [str(item) for item in threshold_failures_raw]
            if isinstance(threshold_failures_raw, list)
            else []
        )
        if cfg.enforce_target:
            target_failures.extend(
                [f"{candidate.candidate_id}: {message}" for message in threshold_failures]
            )

        candidate_record.update(
            {
                "status": "ok",
                "profile_output_path": _as_posix(candidate_profile_output),
                "profile_pass": bool(summary.get("pass", True)),
                "threshold_failures": threshold_failures,
                "steps_per_sec_mean": float(stats.get("mean", 0.0)),
                "steps_per_sec_min": float(stats.get("min", 0.0)),
                "steps_per_sec_p50": float(stats.get("p50", 0.0)),
                "steps_per_sec_p95": float(stats.get("p95", 0.0)),
                "steps_per_sec_p99": float(stats.get("p99", 0.0)),
            }
        )
        candidate_reports.append(candidate_record)

    mode_summaries: dict[str, Any] = {}
    for mode in cfg.modes:
        successful = [
            item
            for item in candidate_reports
            if item.get("mode") == mode and item.get("status") == "ok"
        ]
        mode_summaries[mode] = _build_mode_summary(
            mode=mode,
            successful_candidates=successful,
            target_steps_per_sec=cfg.target_steps_per_sec,
            floor_safety_factor=cfg.floor_safety_factor,
        )

    total_candidates = len(candidate_reports)
    success_count = sum(1 for item in candidate_reports if item.get("status") == "ok")
    skipped_count = sum(1 for item in candidate_reports if item.get("status") == "skipped")
    error_count = sum(1 for item in candidate_reports if item.get("status") == "error")

    mode_target_failures: list[str] = []
    for mode, summary in mode_summaries.items():
        if cfg.enforce_target and not bool(summary.get("target_attained", False)):
            mode_target_failures.append(
                f"{mode} best mean {float(summary.get('best_mean_steps_per_sec', 0.0)):.3f} "
                f"below target {float(cfg.target_steps_per_sec):.3f}"
            )

    report = {
        "generated_at": now_iso(),
        "run_id": run_id,
        "config": _serialize_config(cfg),
        "summary": {
            "pass": error_count == 0
            and len(target_failures) == 0
            and len(mode_target_failures) == 0,
            "total_candidates": total_candidates,
            "successful_candidates": success_count,
            "skipped_candidates": skipped_count,
            "error_candidates": error_count,
            "target_failures": target_failures,
            "mode_target_failures": mode_target_failures,
        },
        "mode_summaries": mode_summaries,
        "candidates": candidate_reports,
        "artifacts": {
            "candidate_reports_root": _as_posix(matrix_root),
            "run_root": _as_posix(candidate_run_root),
            "report_path": _as_posix(output_path),
        },
        "system": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "cpu_count_logical": int(os.cpu_count() or 0),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> ThroughputMatrixConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled throughput matrix sweeps over env/trainer candidates and "
            "publish calibrated floor recommendations per mode."
        )
    )
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/throughput/runs"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-id-prefix", type=str, default="throughput-matrix")

    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--env-time-max", type=float, default=20000.0)

    parser.add_argument("--modes", type=str, default="env_only,trainer")
    parser.add_argument("--env-impls", type=str, default="native,reference")
    parser.add_argument("--trainer-backends", type=str, default="random")

    parser.add_argument("--env-duration-seconds", type=float, default=2.0)
    parser.add_argument("--env-repeats", type=int, default=3)

    parser.add_argument("--trainer-total-env-steps", type=int, default=3000)
    parser.add_argument("--trainer-window-env-steps", type=int, default=1000)
    parser.add_argument("--trainer-eval-replays-per-window", type=int, default=1)
    parser.add_argument("--trainer-eval-max-steps-per-episode", type=int, default=256)
    parser.add_argument("--trainer-repeats", type=int, default=3)

    parser.add_argument("--ppo-num-envs-values", type=str, default="8")
    parser.add_argument("--ppo-num-workers-values", type=str, default="4")
    parser.add_argument("--ppo-rollout-steps-values", type=str, default="128")
    parser.add_argument("--ppo-num-minibatches-values", type=str, default="4")
    parser.add_argument("--ppo-update-epochs-values", type=str, default="4")
    parser.add_argument("--ppo-vector-backends", type=str, default="multiprocessing")
    parser.add_argument("--ppo-env-impls", type=str, default="auto")

    parser.add_argument("--target-steps-per-sec", type=float, default=100000.0)
    parser.add_argument("--enforce-target", action="store_true")

    parser.add_argument("--floor-safety-factor", type=float, default=0.90)
    parser.add_argument("--fail-on-candidate-error", action="store_true")

    args = parser.parse_args()

    return ThroughputMatrixConfig(
        run_root=args.run_root,
        output_path=args.output_path,
        run_id=args.run_id,
        run_id_prefix=args.run_id_prefix,
        seed=args.seed,
        env_time_max=args.env_time_max,
        modes=_parse_csv(args.modes),
        env_impls=_parse_csv(args.env_impls),
        trainer_backends=_parse_csv(args.trainer_backends),
        env_duration_seconds=args.env_duration_seconds,
        env_repeats=args.env_repeats,
        trainer_total_env_steps=args.trainer_total_env_steps,
        trainer_window_env_steps=args.trainer_window_env_steps,
        trainer_eval_replays_per_window=args.trainer_eval_replays_per_window,
        trainer_eval_max_steps_per_episode=args.trainer_eval_max_steps_per_episode,
        trainer_repeats=args.trainer_repeats,
        ppo_num_envs_values=_parse_int_csv(
            args.ppo_num_envs_values, field_name="ppo_num_envs_values"
        ),
        ppo_num_workers_values=_parse_int_csv(
            args.ppo_num_workers_values,
            field_name="ppo_num_workers_values",
        ),
        ppo_rollout_steps_values=_parse_int_csv(
            args.ppo_rollout_steps_values,
            field_name="ppo_rollout_steps_values",
        ),
        ppo_num_minibatches_values=_parse_int_csv(
            args.ppo_num_minibatches_values,
            field_name="ppo_num_minibatches_values",
        ),
        ppo_update_epochs_values=_parse_int_csv(
            args.ppo_update_epochs_values,
            field_name="ppo_update_epochs_values",
        ),
        ppo_vector_backends=_parse_csv(args.ppo_vector_backends),
        ppo_env_impls=_parse_csv(args.ppo_env_impls),
        target_steps_per_sec=args.target_steps_per_sec,
        enforce_target=args.enforce_target,
        floor_safety_factor=args.floor_safety_factor,
        fail_on_candidate_error=args.fail_on_candidate_error,
    )


def main() -> int:
    cfg = _parse_args()
    report = run_throughput_matrix(cfg)
    print(json.dumps(report, indent=2))
    summary = report.get("summary", {})
    return 0 if bool(summary.get("pass", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
