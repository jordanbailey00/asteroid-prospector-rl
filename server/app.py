from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from replay.index import filter_replay_entries, get_replay_entry_by_id, load_replay_index


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    cleaned = [item.strip() for item in value.split(",") if item.strip()]
    return cleaned or None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_replay_index_path(*, run_dir: Path, metadata: dict[str, Any] | None) -> Path:
    if metadata is not None:
        raw_path = metadata.get("replay_index_path")
        if isinstance(raw_path, str) and raw_path.strip() != "":
            return run_dir / raw_path
    return run_dir / "replay_index.json"


def _load_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid run metadata JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"Invalid run metadata payload: {path}")
    return payload


def _open_replay(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def create_app(*, runs_root: Path = Path("runs")) -> FastAPI:
    app = FastAPI(title="Asteroid Prospector API", version="0.1.0")
    app.state.runs_root = runs_root

    def _resolve_run_dir(run_id: str) -> Path:
        run_dir = runs_root / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")
        return run_dir

    def _load_index_for_run(run_id: str, run_dir: Path) -> tuple[Path, dict[str, Any]]:
        metadata = _load_run_metadata(run_dir)
        index_path = _resolve_replay_index_path(run_dir=run_dir, metadata=metadata)
        index_payload = load_replay_index(path=index_path, run_id=run_id)
        return index_path, index_payload

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        if not runs_root.exists():
            return {"runs": [], "count": 0, "total": 0}

        runs: list[dict[str, Any]] = []
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue

            metadata = _load_run_metadata(run_dir)
            if metadata is None:
                continue

            run_id = str(metadata.get("run_id", run_dir.name))
            replay_index_path = _resolve_replay_index_path(run_dir=run_dir, metadata=metadata)
            replay_count = 0
            if replay_index_path.exists():
                index_payload = load_replay_index(path=replay_index_path, run_id=run_id)
                entries = index_payload.get("entries", [])
                replay_count = len(entries) if isinstance(entries, list) else 0

            runs.append(
                {
                    "run_id": run_id,
                    "status": metadata.get("status"),
                    "trainer_backend": metadata.get("trainer_backend"),
                    "env_steps_total": metadata.get("env_steps_total"),
                    "windows_emitted": metadata.get("windows_emitted"),
                    "checkpoints_written": metadata.get("checkpoints_written"),
                    "latest_checkpoint": metadata.get("latest_checkpoint"),
                    "latest_replay": metadata.get("latest_replay"),
                    "replay_count": replay_count,
                    "updated_at": metadata.get("updated_at"),
                    "started_at": metadata.get("started_at"),
                    "finished_at": metadata.get("finished_at"),
                }
            )

        runs.sort(
            key=lambda row: (
                str(row.get("updated_at") or ""),
                str(row.get("run_id") or ""),
            ),
            reverse=True,
        )

        visible = runs[:limit]
        return {
            "runs": visible,
            "count": len(visible),
            "total": len(runs),
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_id)
        metadata = _load_run_metadata(run_dir)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"run metadata not found: {run_id}")

        index_path = _resolve_replay_index_path(run_dir=run_dir, metadata=metadata)
        replay_count = 0
        if index_path.exists():
            index_payload = load_replay_index(path=index_path, run_id=run_id)
            entries = index_payload.get("entries", [])
            replay_count = len(entries) if isinstance(entries, list) else 0

        return {
            "run_id": run_id,
            "metadata": metadata,
            "replay_count": replay_count,
        }

    @app.get("/api/runs/{run_id}/replays")
    def list_replays(
        run_id: str,
        tag: str | None = Query(default=None),
        tags_any: str | None = Query(default=None),
        tags_all: str | None = Query(default=None),
        window_id: int | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=5000),
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_id)
        index_path, index_payload = _load_index_for_run(run_id, run_dir)
        entries = index_payload.get("entries", [])
        if not isinstance(entries, list):
            raise HTTPException(status_code=500, detail="Invalid replay index entries payload")

        replay_rows = filter_replay_entries(
            entries,
            tag=tag,
            tags_any=_parse_csv_arg(tags_any),
            tags_all=_parse_csv_arg(tags_all),
            window_id=window_id,
            limit=limit,
        )

        return {
            "run_id": run_id,
            "index_path": str(index_path.name),
            "count": len(replay_rows),
            "replays": replay_rows,
        }

    @app.get("/api/runs/{run_id}/replays/{replay_id}")
    def get_replay(run_id: str, replay_id: str) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_id)
        _, index_payload = _load_index_for_run(run_id, run_dir)
        entry = get_replay_entry_by_id(index_payload, replay_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"replay_id not found: {replay_id}")

        return {
            "run_id": run_id,
            "replay": entry,
        }

    @app.get("/api/runs/{run_id}/replays/{replay_id}/frames")
    def get_replay_frames(
        run_id: str,
        replay_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=256, ge=1, le=5000),
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_id)
        _, index_payload = _load_index_for_run(run_id, run_dir)
        entry = get_replay_entry_by_id(index_payload, replay_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"replay_id not found: {replay_id}")

        replay_path_raw = entry.get("replay_path")
        if not isinstance(replay_path_raw, str) or replay_path_raw.strip() == "":
            raise HTTPException(status_code=500, detail="Replay entry missing replay_path")

        replay_path = run_dir / replay_path_raw
        if not replay_path.exists():
            raise HTTPException(status_code=404, detail=f"replay file not found: {replay_path_raw}")

        frames: list[dict[str, Any]] = []
        has_more = False
        with _open_replay(replay_path) as handle:
            for frame_idx, line in enumerate(handle):
                if frame_idx < offset:
                    continue
                if len(frames) >= limit:
                    has_more = True
                    break

                text = line.strip()
                if text == "":
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=500, detail="Invalid replay frame JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise HTTPException(
                        status_code=500,
                        detail="Invalid replay frame payload",
                    )
                frames.append(payload)

        return {
            "run_id": run_id,
            "replay_id": replay_id,
            "offset": offset,
            "next_offset": offset + len(frames),
            "count": len(frames),
            "has_more": has_more,
            "frames": frames,
        }

    return app
