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
        url = getattr(self._run, "url", None)
        if url:
            return str(url)
        get_url = getattr(self._run, "get_url", None)
        if callable(get_url):
            return str(get_url())
        return None

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


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    metrics_windows_path: Path
    config_path: Path
    metadata_path: Path
