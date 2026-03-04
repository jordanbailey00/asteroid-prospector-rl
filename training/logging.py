from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALIAS_ALLOWED = re.compile(r"[^a-zA-Z0-9_.-]+")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")))
        handle.write("\n")


def _artifact_alias(raw: str) -> str:
    alias = _ALIAS_ALLOWED.sub("-", raw.strip())
    return alias.strip("-") or "latest"


def _run_url(run: Any) -> str | None:
    url = getattr(run, "url", None)
    if url:
        return str(url)
    get_url = getattr(run, "get_url", None)
    if callable(get_url):
        return str(get_url())
    return None


def _benchmark_metadata(report: dict[str, Any]) -> dict[str, Any]:
    comparison = report.get("comparison") if isinstance(report, dict) else {}
    summary = report.get("summary") if isinstance(report, dict) else {}
    artifacts = report.get("artifacts") if isinstance(report, dict) else {}

    comparison_dict = comparison if isinstance(comparison, dict) else {}
    summary_dict = summary if isinstance(summary, dict) else {}
    artifacts_dict = artifacts if isinstance(artifacts, dict) else {}

    training_ids_raw = artifacts_dict.get("training_run_ids", [])
    training_ids = (
        [str(value) for value in training_ids_raw] if isinstance(training_ids_raw, list) else []
    )

    seed_count = 0
    try:
        seed_count = int(summary_dict.get("seed_count", 0))
    except (TypeError, ValueError):
        seed_count = 0

    return {
        "generated_at": str(report.get("generated_at", "")),
        "reference_policy": str(comparison_dict.get("reference_policy", "")),
        "summary_pass": bool(summary_dict.get("pass", False)),
        "seed_count": seed_count,
        "training_run_ids": training_ids,
    }


class JsonlWindowLogger:
    def __init__(self, *, path: Path) -> None:
        self.path = path

    def log_window(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.path, payload)


class WandbWindowLogger:
    """Thin wrapper over a W&B run object for windowed training logs."""

    def __init__(
        self,
        *,
        run: Any,
        artifact_ctor: Callable[..., Any] | None,
    ) -> None:
        self._run = run
        self._artifact_ctor = artifact_ctor

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        project: str,
        config: dict[str, Any],
        mode: str,
        tags: list[str] | None = None,
    ) -> WandbWindowLogger | None:
        if mode == "disabled":
            return None

        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "wandb logging requested but wandb is not installed. "
                "Install wandb or use --wandb-mode disabled."
            ) from exc

        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": run_id,
            "config": config,
            "tags": tags or [],
        }
        if mode in {"offline", "online"}:
            init_kwargs["mode"] = mode

        run = wandb.init(**init_kwargs)
        return cls(run=run, artifact_ctor=wandb.Artifact)

    @property
    def run_url(self) -> str | None:
        return _run_url(self._run)

    def log_window(self, payload: dict[str, Any], *, step: int) -> None:
        self._run.log(payload, step=int(step))

    def log_checkpoint(self, *, checkpoint_path: Path, run_id: str, window_id: int) -> None:
        if self._artifact_ctor is None:
            return

        artifact = self._artifact_ctor(name=f"model-{run_id}", type="model")
        artifact.add_file(str(checkpoint_path))
        self._run.log_artifact(artifact, aliases=["latest", f"window-{window_id}"])

    def log_replay(
        self,
        *,
        replay_path: Path,
        run_id: str,
        window_id: int,
        replay_id: str,
        tags: list[str],
    ) -> None:
        if self._artifact_ctor is None:
            return

        artifact = self._artifact_ctor(
            name=f"replay-{run_id}-{window_id:06d}-{replay_id}", type="replay"
        )
        artifact.add_file(str(replay_path))

        aliases = ["latest", f"window-{window_id}"]
        aliases.extend(_artifact_alias(tag) for tag in tags)
        deduped_aliases = list(dict.fromkeys(aliases))
        self._run.log_artifact(artifact, aliases=deduped_aliases)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if summary:
            for key, value in summary.items():
                self._run.summary[key] = value
        self._run.finish()


class WandbBenchmarkLogger:
    """Thin wrapper over a W&B run object for benchmark/eval reporting."""

    def __init__(
        self,
        *,
        run: Any,
        artifact_ctor: Callable[..., Any] | None,
        job_type: str,
    ) -> None:
        self._run = run
        self._artifact_ctor = artifact_ctor
        self._job_type = job_type

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        project: str,
        config: dict[str, Any],
        mode: str,
        job_type: str,
        tags: list[str] | None = None,
        entity: str | None = None,
        run_name: str | None = None,
    ) -> WandbBenchmarkLogger | None:
        if mode == "disabled":
            return None

        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "wandb benchmark logging requested but wandb is not installed. "
                "Install wandb or use --benchmark-wandb-mode disabled."
            ) from exc

        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": run_name or f"{run_id}-benchmark",
            "config": config,
            "tags": tags or [],
            "job_type": job_type,
        }
        if entity is not None and entity.strip() != "":
            init_kwargs["entity"] = entity.strip()
        if mode in {"offline", "online"}:
            init_kwargs["mode"] = mode

        run = wandb.init(**init_kwargs)
        return cls(run=run, artifact_ctor=wandb.Artifact, job_type=job_type)

    @property
    def run_url(self) -> str | None:
        return _run_url(self._run)

    def log_metrics(self, payload: dict[str, Any], *, step: int = 0) -> None:
        if not payload:
            return
        self._run.log(payload, step=int(step))

    def log_benchmark_report(
        self,
        *,
        report_path: Path,
        run_id: str,
        report: dict[str, Any],
        lineage_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        if self._artifact_ctor is None:
            return {
                "artifact_name": None,
                "artifact_aliases": [],
                "lineage_file_count": 0,
            }

        artifact = self._artifact_ctor(name=f"benchmark-{run_id}", type="benchmark")

        metadata = _benchmark_metadata(report)
        metadata_attr = getattr(artifact, "metadata", None)
        if isinstance(metadata_attr, dict):
            metadata_attr.update(metadata)
        else:
            try:
                artifact.metadata = metadata
            except Exception:
                pass

        artifact.add_file(str(report_path))

        lineage_count = 0
        for path in lineage_paths or []:
            if not path.exists():
                continue
            lineage_count += 1
            try:
                artifact.add_file(str(path), name=f"lineage/{path.name}")
            except TypeError:
                artifact.add_file(str(path))

        aliases = [
            "latest",
            _artifact_alias(run_id),
            _artifact_alias(f"job-{self._job_type}"),
        ]
        deduped_aliases = list(dict.fromkeys(aliases))
        self._run.log_artifact(artifact, aliases=deduped_aliases)

        return {
            "artifact_name": str(getattr(artifact, "name", f"benchmark-{run_id}")),
            "artifact_aliases": deduped_aliases,
            "lineage_file_count": int(lineage_count),
        }

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if summary:
            for key, value in summary.items():
                self._run.summary[key] = value
        self._run.finish()


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    metrics_windows_path: Path
    config_path: Path
    metadata_path: Path
