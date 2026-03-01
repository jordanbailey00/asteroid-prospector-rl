from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from tools.profile_training_throughput import (  # noqa: E402
    PROFILE_MODE_ENV_ONLY,
    PROFILE_MODE_TRAINER,
    PROFILE_MODE_TRAINER_EVAL,
    ThroughputProfileConfig,
    run_training_throughput_profile,
)

SUPPORTED_MODES = (PROFILE_MODE_ENV_ONLY, PROFILE_MODE_TRAINER, PROFILE_MODE_TRAINER_EVAL)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _parse_modes_csv(raw: str) -> tuple[str, ...]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip() != ""]
    deduped = tuple(dict.fromkeys(values))
    for value in deduped:
        if value not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode in --modes: {value!r}. Supported: {', '.join(SUPPORTED_MODES)}"
            )
    if not deduped:
        raise ValueError("At least one mode must be supplied.")
    return deduped


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object payload.")
    return payload


@dataclass(frozen=True)
class ThroughputFloorGateConfig:
    matrix_report_path: Path
    run_root: Path = Path("artifacts/throughput/runs")
    output_path: Path | None = None
    run_id: str | None = None
    run_id_prefix: str = "throughput-floor-gate"

    modes: tuple[str, ...] = (PROFILE_MODE_ENV_ONLY, PROFILE_MODE_TRAINER)

    seed: int = 17
    env_time_max: float = 20000.0

    env_duration_seconds: float = 2.0
    env_repeats: int = 3

    trainer_total_env_steps: int = 3000
    trainer_window_env_steps: int = 1000
    trainer_eval_replays_per_window: int = 1
    trainer_eval_max_steps_per_episode: int = 256
    trainer_repeats: int = 3


def _validate_config(cfg: ThroughputFloorGateConfig) -> None:
    if not cfg.matrix_report_path.exists():
        raise ValueError(f"matrix_report_path does not exist: {cfg.matrix_report_path.as_posix()}")
    if not cfg.modes:
        raise ValueError("modes must be non-empty.")
    for mode in cfg.modes:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode!r}")
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


def _find_candidate(matrix_report: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates = matrix_report.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("matrix report candidates payload is invalid.")

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("candidate_id", "")) != candidate_id:
            continue
        if str(candidate.get("status", "")) != "ok":
            raise ValueError(
                f"Best candidate {candidate_id!r} exists but is not successful in matrix report."
            )
        return candidate

    raise ValueError(f"Candidate {candidate_id!r} not found in matrix report.")


def _mode_target(
    *,
    matrix_report: dict[str, Any],
    mode: str,
) -> tuple[float, str, dict[str, Any]]:
    mode_summaries = matrix_report.get("mode_summaries", {})
    if not isinstance(mode_summaries, dict):
        raise ValueError("matrix report mode_summaries payload is invalid.")

    mode_summary = mode_summaries.get(mode)
    if not isinstance(mode_summary, dict):
        raise ValueError(f"Mode summary missing for {mode!r} in matrix report.")

    target = float(mode_summary.get("recommended_floor_steps_per_sec", 0.0))
    if target <= 0.0:
        raise ValueError(
            f"Mode {mode!r} has non-positive recommended floor in matrix report: {target}"
        )

    best_candidate_id = str(mode_summary.get("best_candidate_id", "")).strip()
    if best_candidate_id == "":
        raise ValueError(f"Mode {mode!r} has no best_candidate_id in matrix report.")

    best_candidate = _find_candidate(matrix_report, best_candidate_id)
    return target, best_candidate_id, best_candidate


def _serialize_config(cfg: ThroughputFloorGateConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["matrix_report_path"] = cfg.matrix_report_path.as_posix()
    payload["run_root"] = cfg.run_root.as_posix()
    payload["output_path"] = cfg.output_path.as_posix() if cfg.output_path is not None else None
    payload["modes"] = list(cfg.modes)
    return payload


def run_throughput_floor_gate(cfg: ThroughputFloorGateConfig) -> dict[str, Any]:
    _validate_config(cfg)

    matrix_report = _load_json(cfg.matrix_report_path)
    run_id = cfg.run_id or _default_run_id(cfg.run_id_prefix)

    output_path = cfg.output_path
    if output_path is None:
        output_path = Path("artifacts/throughput") / f"{run_id}.json"

    if output_path.exists():
        raise ValueError(f"Output already exists: {output_path.as_posix()}")

    run_root = cfg.run_root / run_id
    profiler_outputs_root = Path("artifacts/throughput/floor_gate") / run_id

    mode_results: list[dict[str, Any]] = []
    all_pass = True

    for mode in cfg.modes:
        target, best_candidate_id, best_candidate = _mode_target(
            matrix_report=matrix_report, mode=mode
        )

        candidate_mode = str(best_candidate.get("mode", "")).strip()
        if candidate_mode != mode:
            raise ValueError(
                "Best candidate mode mismatch for "
                f"{mode!r}: {best_candidate_id!r} has mode {candidate_mode!r}"
            )

        profile_output_path = profiler_outputs_root / f"{mode}.json"
        profile_cfg = ThroughputProfileConfig(
            run_root=run_root,
            output_path=profile_output_path,
            run_id=f"{run_id}-{mode}",
            run_id_prefix="throughput",
            seed=int(cfg.seed),
            env_time_max=float(cfg.env_time_max),
            modes=(mode,),
            env_impl=str(best_candidate.get("env_impl", "auto")),
            env_duration_seconds=float(cfg.env_duration_seconds),
            env_repeats=int(cfg.env_repeats),
            trainer_backend=str(best_candidate.get("trainer_backend", "random")),
            trainer_total_env_steps=int(cfg.trainer_total_env_steps),
            trainer_window_env_steps=int(cfg.trainer_window_env_steps),
            trainer_eval_replays_per_window=int(cfg.trainer_eval_replays_per_window),
            trainer_eval_max_steps_per_episode=int(cfg.trainer_eval_max_steps_per_episode),
            trainer_repeats=int(cfg.trainer_repeats),
            ppo_num_envs=int(best_candidate.get("ppo_num_envs", 8)),
            ppo_num_workers=int(best_candidate.get("ppo_num_workers", 4)),
            ppo_rollout_steps=int(best_candidate.get("ppo_rollout_steps", 128)),
            ppo_num_minibatches=int(best_candidate.get("ppo_num_minibatches", 4)),
            ppo_update_epochs=int(best_candidate.get("ppo_update_epochs", 4)),
            ppo_vector_backend=str(best_candidate.get("ppo_vector_backend", "multiprocessing")),
            ppo_env_impl=str(best_candidate.get("ppo_env_impl", "auto")),
            target_steps_per_sec=target,
            enforce_target=True,
        )

        profile_report = run_training_throughput_profile(profile_cfg)
        profile_summary = profile_report.get("summary", {})
        profile_modes = profile_report.get("modes", [])
        profile_mode_report = (
            profile_modes[0] if isinstance(profile_modes, list) and profile_modes else {}
        )
        stats = profile_mode_report.get("steps_per_sec_stats", {})

        mode_pass = bool(profile_summary.get("pass", False))
        all_pass = all_pass and mode_pass

        threshold_failures_raw = profile_summary.get("threshold_failures", [])
        threshold_failures = (
            [str(item) for item in threshold_failures_raw]
            if isinstance(threshold_failures_raw, list)
            else []
        )

        mode_results.append(
            {
                "mode": mode,
                "status": "pass" if mode_pass else "fail",
                "target_steps_per_sec": target,
                "best_candidate_id": best_candidate_id,
                "best_candidate": best_candidate,
                "measured_steps_per_sec_mean": float(stats.get("mean", 0.0)),
                "measured_steps_per_sec_min": float(stats.get("min", 0.0)),
                "measured_steps_per_sec_p50": float(stats.get("p50", 0.0)),
                "measured_steps_per_sec_p95": float(stats.get("p95", 0.0)),
                "threshold_failures": threshold_failures,
                "profile_output_path": profile_output_path.as_posix(),
            }
        )

    aspirational_target = float(
        matrix_report.get("config", {}).get("target_steps_per_sec", 100000.0)
    )
    aspirational_deltas = {
        result["mode"]: aspirational_target - float(result.get("measured_steps_per_sec_mean", 0.0))
        for result in mode_results
    }

    report = {
        "generated_at": now_iso(),
        "run_id": run_id,
        "config": _serialize_config(cfg),
        "matrix_source": {
            "path": cfg.matrix_report_path.as_posix(),
            "run_id": matrix_report.get("run_id"),
            "generated_at": matrix_report.get("generated_at"),
        },
        "summary": {
            "pass": all_pass,
            "aspirational_target_steps_per_sec": aspirational_target,
            "modes_evaluated": list(cfg.modes),
            "mode_status": {result["mode"]: result["status"] for result in mode_results},
            "aspirational_delta_steps_per_sec": aspirational_deltas,
        },
        "modes": mode_results,
        "artifacts": {
            "report_path": output_path.as_posix(),
            "profile_reports_root": profiler_outputs_root.as_posix(),
            "run_root": run_root.as_posix(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> ThroughputFloorGateConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce per-mode throughput floors using a previously generated throughput "
            "matrix artifact."
        )
    )
    parser.add_argument(
        "--matrix-report-path",
        type=Path,
        default=Path("artifacts/throughput/throughput-matrix-20260301-p6.json"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/throughput/runs"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-id-prefix", type=str, default="throughput-floor-gate")

    parser.add_argument("--modes", type=str, default="env_only,trainer")

    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--env-time-max", type=float, default=20000.0)

    parser.add_argument("--env-duration-seconds", type=float, default=2.0)
    parser.add_argument("--env-repeats", type=int, default=3)

    parser.add_argument("--trainer-total-env-steps", type=int, default=3000)
    parser.add_argument("--trainer-window-env-steps", type=int, default=1000)
    parser.add_argument("--trainer-eval-replays-per-window", type=int, default=1)
    parser.add_argument("--trainer-eval-max-steps-per-episode", type=int, default=256)
    parser.add_argument("--trainer-repeats", type=int, default=3)

    args = parser.parse_args()
    return ThroughputFloorGateConfig(
        matrix_report_path=args.matrix_report_path,
        run_root=args.run_root,
        output_path=args.output_path,
        run_id=args.run_id,
        run_id_prefix=args.run_id_prefix,
        modes=_parse_modes_csv(args.modes),
        seed=args.seed,
        env_time_max=args.env_time_max,
        env_duration_seconds=args.env_duration_seconds,
        env_repeats=args.env_repeats,
        trainer_total_env_steps=args.trainer_total_env_steps,
        trainer_window_env_steps=args.trainer_window_env_steps,
        trainer_eval_replays_per_window=args.trainer_eval_replays_per_window,
        trainer_eval_max_steps_per_episode=args.trainer_eval_max_steps_per_episode,
        trainer_repeats=args.trainer_repeats,
    )


def main() -> int:
    cfg = _parse_args()
    report = run_throughput_floor_gate(cfg)
    print(json.dumps(report, indent=2))
    summary = report.get("summary", {})
    return 0 if bool(summary.get("pass", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
