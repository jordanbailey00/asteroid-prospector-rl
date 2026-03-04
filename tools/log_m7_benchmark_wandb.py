from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.logging import WandbBenchmarkLogger

SUPPORTED_WANDB_MODES = ("disabled", "offline", "online")
COMPARISON_METRICS = (
    "net_profit_mean",
    "survival_rate",
    "profit_per_tick_mean",
    "overheat_ticks_mean",
    "pirate_encounters_mean",
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_tags_csv(raw: str) -> tuple[str, ...]:
    parts = [item.strip() for item in raw.split(",") if item.strip() != ""]
    return tuple(dict.fromkeys(parts))


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark report payload must be a JSON object.")
    return payload


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _flatten_benchmark_metrics(report: dict[str, Any]) -> dict[str, float]:
    summary = report.get("summary", {})
    comparison = report.get("comparison", {})
    contenders_raw = report.get("contenders", [])

    summary_dict = summary if isinstance(summary, dict) else {}
    comparison_dict = comparison if isinstance(comparison, dict) else {}
    contenders = contenders_raw if isinstance(contenders_raw, list) else []

    metrics: dict[str, float] = {
        "benchmark/pass": 1.0 if bool(summary_dict.get("pass", False)) else 0.0,
        "benchmark/seed_count": _as_float(summary_dict.get("seed_count", 0)),
        "benchmark/episodes_per_seed": _as_float(summary_dict.get("episodes_per_seed", 0)),
        "benchmark/episodes_per_contender": _as_float(
            summary_dict.get("episodes_per_contender", 0)
        ),
        "benchmark/expectations_pass": (
            1.0 if bool(summary_dict.get("expectations_pass", False)) else 0.0
        ),
    }

    reference_policy = str(comparison_dict.get("reference_policy", "")).strip()
    contender_map: dict[str, dict[str, Any]] = {}
    for row in contenders:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        aggregate = row.get("aggregate", {})
        if name == "" or not isinstance(aggregate, dict):
            continue
        contender_map[name] = aggregate

    if reference_policy in contender_map:
        aggregate = contender_map[reference_policy]
        for metric in COMPARISON_METRICS:
            metrics[f"benchmark/reference/{metric}"] = _as_float(aggregate.get(metric, 0.0))

    comparison_rows_raw = comparison_dict.get("rows", [])
    comparison_rows = comparison_rows_raw if isinstance(comparison_rows_raw, list) else []
    for row in comparison_rows:
        if not isinstance(row, dict):
            continue
        contender = str(row.get("contender", "")).strip()
        if contender == "":
            continue
        metric_rows_raw = row.get("metrics", [])
        metric_rows = metric_rows_raw if isinstance(metric_rows_raw, list) else []
        for metric_row in metric_rows:
            if not isinstance(metric_row, dict):
                continue
            metric_name = str(metric_row.get("metric", "")).strip()
            if metric_name == "":
                continue
            delta = _as_float(metric_row.get("delta_reference_minus_candidate", 0.0))
            better = bool(metric_row.get("reference_better_or_equal", False))
            metrics[f"benchmark/delta/{contender}/{metric_name}"] = delta
            metrics[f"benchmark/better_or_equal/{contender}/{metric_name}"] = 1.0 if better else 0.0

    expectations_raw = comparison_dict.get("expectations", [])
    expectations = expectations_raw if isinstance(expectations_raw, list) else []
    for expectation in expectations:
        if not isinstance(expectation, dict):
            continue
        name = str(expectation.get("name", "")).strip()
        if name == "":
            continue
        metrics[f"benchmark/expectation/{name}/pass"] = (
            1.0 if bool(expectation.get("pass", False)) else 0.0
        )

    return metrics


def _resolve_lineage_paths(report: dict[str, Any], *, report_path: Path) -> list[Path]:
    config = report.get("config", {}) if isinstance(report, dict) else {}
    config_dict = config if isinstance(config, dict) else {}

    run_root_raw = str(config_dict.get("run_root", "")).strip()
    run_root = Path(run_root_raw) if run_root_raw != "" else report_path.parent / "runs"

    training_runs_raw = report.get("training_runs", []) if isinstance(report, dict) else []
    training_runs = training_runs_raw if isinstance(training_runs_raw, list) else []

    paths: list[Path] = []
    for row in training_runs:
        if not isinstance(row, dict):
            continue
        run_id = str(row.get("run_id", "")).strip()
        checkpoint_rel = str(row.get("latest_checkpoint_path", "")).strip()
        if run_id == "" or checkpoint_rel == "":
            continue
        candidate = run_root / run_id / checkpoint_rel
        if candidate.exists():
            paths.append(candidate)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.resolve().as_posix() if path.exists() else path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


@dataclass(frozen=True)
class M7BenchmarkWandbLogConfig:
    report_path: Path
    output_path: Path | None = None
    wandb_mode: str = "online"
    wandb_project: str = "asteroid-prospector"
    wandb_entity: str | None = None
    wandb_job_type: str = "eval"
    wandb_tags: tuple[str, ...] = ("m7", "benchmark", "eval")
    wandb_run_name: str | None = None


def _serialize_config(cfg: M7BenchmarkWandbLogConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["report_path"] = cfg.report_path.as_posix()
    payload["output_path"] = cfg.output_path.as_posix() if cfg.output_path is not None else None
    payload["wandb_tags"] = list(cfg.wandb_tags)
    return payload


def _validate_config(cfg: M7BenchmarkWandbLogConfig) -> None:
    if not cfg.report_path.exists():
        raise ValueError(f"report_path does not exist: {cfg.report_path.as_posix()}")
    if cfg.wandb_mode not in SUPPORTED_WANDB_MODES:
        raise ValueError(
            f"Unsupported wandb_mode {cfg.wandb_mode!r}; "
            f"supported: {', '.join(SUPPORTED_WANDB_MODES)}."
        )
    if cfg.wandb_project.strip() == "":
        raise ValueError("wandb_project must be non-empty.")
    if cfg.wandb_job_type.strip() == "":
        raise ValueError("wandb_job_type must be non-empty.")


def log_m7_benchmark_to_wandb(cfg: M7BenchmarkWandbLogConfig) -> dict[str, Any]:
    _validate_config(cfg)

    output_path = cfg.output_path or cfg.report_path
    report = _load_report(cfg.report_path)
    report_run_id = str(report.get("run_id", "")).strip()
    if report_run_id == "":
        raise ValueError("Benchmark report is missing run_id.")

    if cfg.wandb_mode == "disabled":
        report["wandb_benchmark"] = {
            "enabled": False,
            "mode": cfg.wandb_mode,
            "project": cfg.wandb_project,
            "entity": cfg.wandb_entity,
            "job_type": cfg.wandb_job_type,
            "logged_at": now_iso(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    logger = WandbBenchmarkLogger.create(
        run_id=report_run_id,
        project=cfg.wandb_project,
        config={
            "source_report_path": cfg.report_path.as_posix(),
            "benchmark_config": report.get("config", {}),
            "logger_config": _serialize_config(cfg),
        },
        mode=cfg.wandb_mode,
        job_type=cfg.wandb_job_type,
        tags=list(cfg.wandb_tags),
        entity=cfg.wandb_entity,
        run_name=cfg.wandb_run_name,
    )
    if logger is None:
        raise RuntimeError("Expected WandbBenchmarkLogger instance but got None.")

    metrics_payload = _flatten_benchmark_metrics(report)
    logger.log_metrics(metrics_payload, step=0)

    lineage_paths = _resolve_lineage_paths(report, report_path=cfg.report_path)
    artifact_info = logger.log_benchmark_report(
        report_path=cfg.report_path,
        run_id=report_run_id,
        report=report,
        lineage_paths=lineage_paths,
    )

    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    summary_dict = summary if isinstance(summary, dict) else {}
    logger.finish(
        {
            "benchmark_pass": bool(summary_dict.get("pass", False)),
            "seed_count": int(summary_dict.get("seed_count", 0)),
            "episodes_per_seed": int(summary_dict.get("episodes_per_seed", 0)),
            "source_report_path": cfg.report_path.as_posix(),
        }
    )

    report["wandb_benchmark"] = {
        "enabled": True,
        "mode": cfg.wandb_mode,
        "project": cfg.wandb_project,
        "entity": cfg.wandb_entity,
        "job_type": cfg.wandb_job_type,
        "run_url": logger.run_url,
        "tags": list(cfg.wandb_tags),
        "logged_at": now_iso(),
        **artifact_info,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Log an M7 benchmark protocol report to Weights & Biases as an "
            "eval/benchmark run with artifact lineage."
        )
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=list(SUPPORTED_WANDB_MODES),
        default="online",
    )
    parser.add_argument("--wandb-project", type=str, default="asteroid-prospector")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default="eval")
    parser.add_argument("--wandb-tags", type=str, default="m7,benchmark,eval")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = M7BenchmarkWandbLogConfig(
        report_path=args.report_path,
        output_path=args.output_path,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_job_type=args.wandb_job_type,
        wandb_tags=_parse_tags_csv(args.wandb_tags),
        wandb_run_name=args.wandb_run_name,
    )

    report = log_m7_benchmark_to_wandb(cfg)
    wandb_payload = report.get("wandb_benchmark", {}) if isinstance(report, dict) else {}
    print(
        "[m7-wandb] run_id={run_id} enabled={enabled} mode={mode} project={project}".format(
            run_id=report.get("run_id", "unknown"),
            enabled=bool(wandb_payload.get("enabled", False)),
            mode=str(wandb_payload.get("mode", cfg.wandb_mode)),
            project=str(wandb_payload.get("project", cfg.wandb_project)),
        )
    )
    if bool(wandb_payload.get("enabled", False)):
        print(
            "[m7-wandb] run_url={url} artifact={artifact} lineage_files={count}".format(
                url=str(wandb_payload.get("run_url", "")),
                artifact=str(wandb_payload.get("artifact_name", "")),
                count=int(wandb_payload.get("lineage_file_count", 0)),
            )
        )

    print(f"[m7-wandb] wrote report to { (cfg.output_path or cfg.report_path).as_posix() }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
