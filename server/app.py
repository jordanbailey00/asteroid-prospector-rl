from __future__ import annotations

import asyncio
import gzip
import json
import random
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from replay.index import filter_replay_entries, get_replay_entry_by_id, load_replay_index

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
DEFAULT_CORS_ORIGIN_REGEX = r"https://.*\.vercel\.app"
N_ACTIONS = 69


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_python_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python_src = repo_root / "python"
    if str(python_src) not in sys.path:
        sys.path.insert(0, str(python_src))


def _as_relative_posix(path: Path, *, start: Path) -> str:
    try:
        return path.relative_to(start).as_posix()
    except ValueError:
        return path.as_posix()


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


def _resolve_metrics_windows_path(*, run_dir: Path, metadata: dict[str, Any] | None) -> Path:
    if metadata is not None:
        raw_path = metadata.get("metrics_windows_path")
        if isinstance(raw_path, str) and raw_path.strip() != "":
            return run_dir / raw_path
    return run_dir / "metrics" / "windows.jsonl"


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


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text == "":
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Invalid JSONL row in {path}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=500, detail=f"Invalid JSONL row payload in {path}")
        rows.append(payload)
    return rows


def _parse_replay_frame_payload(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if text == "":
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Invalid replay frame JSON",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail="Invalid replay frame payload",
        )
    return payload


def _resolve_replay_file_path(*, run_dir: Path, entry: dict[str, Any]) -> tuple[str, Path]:
    replay_path_raw = entry.get("replay_path")
    if not isinstance(replay_path_raw, str) or replay_path_raw.strip() == "":
        raise HTTPException(status_code=500, detail="Replay entry missing replay_path")

    replay_path = run_dir / replay_path_raw
    if not replay_path.exists():
        raise HTTPException(status_code=404, detail=f"replay file not found: {replay_path_raw}")

    return replay_path_raw, replay_path


def _obs_to_list(obs: Any) -> list[float]:
    return [float(value) for value in np.asarray(obs, dtype=np.float32).tolist()]


class CreatePlaySessionRequest(BaseModel):
    seed: int | None = None
    env_time_max: float = Field(default=20000.0, gt=0.0)


class ResetPlaySessionRequest(BaseModel):
    seed: int | None = None


class StepPlaySessionRequest(BaseModel):
    action: int = Field(ge=0, le=N_ACTIONS - 1)


@dataclass
class _PlaySession:
    env: Any
    env_time_max: float
    next_seed: int
    steps: int
    created_at: str
    updated_at: str


class _PlaySessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _PlaySession] = {}

    def _coerce_seed(self, seed: int | None) -> int:
        if seed is not None:
            return int(seed)
        return random.SystemRandom().randint(0, 2**31 - 1)

    def create(
        self, *, env_time_max: float, seed: int | None
    ) -> tuple[str, int, Any, dict[str, Any], _PlaySession]:
        _ensure_python_src_on_path()
        from asteroid_prospector import ProspectorReferenceEnv, ReferenceEnvConfig

        actual_seed = self._coerce_seed(seed)
        env = ProspectorReferenceEnv(
            config=ReferenceEnvConfig(time_max=env_time_max),
            seed=actual_seed,
        )
        obs, info = env.reset(seed=actual_seed)

        created_at = now_iso()
        session = _PlaySession(
            env=env,
            env_time_max=float(env_time_max),
            next_seed=int(actual_seed + 1),
            steps=0,
            created_at=created_at,
            updated_at=created_at,
        )
        session_id = uuid4().hex

        with self._lock:
            self._sessions[session_id] = session

        return session_id, actual_seed, obs, info, session

    def get(self, session_id: str) -> _PlaySession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session_id not found: {session_id}")
        return session

    def reset(
        self, *, session_id: str, seed: int | None
    ) -> tuple[int, Any, dict[str, Any], _PlaySession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail=f"session_id not found: {session_id}")

            actual_seed = self._coerce_seed(seed) if seed is not None else session.next_seed
            obs, info = session.env.reset(seed=actual_seed)
            session.next_seed = int(actual_seed + 1)
            session.steps = 0
            session.updated_at = now_iso()

        return actual_seed, obs, info, session

    def step(
        self,
        *,
        session_id: str,
        action: int,
    ) -> tuple[Any, float, bool, bool, dict[str, Any], _PlaySession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail=f"session_id not found: {session_id}")

            obs, reward, terminated, truncated, info = session.env.step(int(action))
            session.steps += 1
            session.updated_at = now_iso()

        return obs, float(reward), bool(terminated), bool(truncated), info, session

    def delete(self, *, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session_id not found: {session_id}")

        close = getattr(session.env, "close", None)
        if callable(close):
            close()


def create_app(
    *,
    runs_root: Path = Path("runs"),
    cors_allow_origins: list[str] | None = None,
    cors_allow_origin_regex: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Asteroid Prospector API", version="0.2.0")
    app.state.runs_root = runs_root
    app.state.play_sessions = _PlaySessionStore()

    origins = (
        list(cors_allow_origins) if cors_allow_origins is not None else list(DEFAULT_CORS_ORIGINS)
    )
    origin_regex = (
        cors_allow_origin_regex
        if cors_allow_origin_regex is not None
        else DEFAULT_CORS_ORIGIN_REGEX
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @app.get("/api/runs/{run_id}/metrics/windows")
    def get_run_metrics_windows(
        run_id: str,
        limit: int = Query(default=200, ge=1, le=5000),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_id)
        metadata = _load_run_metadata(run_dir)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"run metadata not found: {run_id}")

        metrics_path = _resolve_metrics_windows_path(run_dir=run_dir, metadata=metadata)
        metrics_path_rel = _as_relative_posix(metrics_path, start=run_dir)
        if not metrics_path.exists():
            return {
                "run_id": run_id,
                "metrics_path": metrics_path_rel,
                "count": 0,
                "total": 0,
                "windows": [],
            }

        rows = _load_jsonl_rows(metrics_path)
        rows.sort(
            key=lambda row: (
                int(row.get("window_id", -1)),
                int(row.get("env_steps_total", -1)),
            ),
            reverse=(order == "desc"),
        )
        visible = rows[:limit]
        return {
            "run_id": run_id,
            "metrics_path": metrics_path_rel,
            "count": len(visible),
            "total": len(rows),
            "windows": visible,
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
            "index_path": str(_as_relative_posix(index_path, start=run_dir)),
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

        _, replay_path = _resolve_replay_file_path(run_dir=run_dir, entry=entry)

        frames: list[dict[str, Any]] = []
        has_more = False
        with _open_replay(replay_path) as handle:
            for frame_idx, line in enumerate(handle):
                if frame_idx < offset:
                    continue
                if len(frames) >= limit:
                    has_more = True
                    break

                payload = _parse_replay_frame_payload(line)
                if payload is None:
                    continue
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

    @app.websocket("/ws/runs/{run_id}/replays/{replay_id}/frames")
    async def stream_replay_frames(websocket: WebSocket, run_id: str, replay_id: str) -> None:
        await websocket.accept()
        try:
            try:
                offset = int(websocket.query_params.get("offset", "0"))
                limit = int(websocket.query_params.get("limit", "5000"))
                batch_size = int(websocket.query_params.get("batch_size", "256"))
                max_chunk_bytes = int(websocket.query_params.get("max_chunk_bytes", "262144"))
                yield_every_batches = int(websocket.query_params.get("yield_every_batches", "8"))
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "offset, limit, batch_size, max_chunk_bytes, and "
                        "yield_every_batches must be integers"
                    ),
                ) from exc

            if offset < 0:
                raise HTTPException(status_code=400, detail="offset must be >= 0")
            if limit < 1 or limit > 50000:
                raise HTTPException(status_code=400, detail="limit must be in [1, 50000]")
            if batch_size < 1 or batch_size > 5000:
                raise HTTPException(status_code=400, detail="batch_size must be in [1, 5000]")
            if max_chunk_bytes < 1024 or max_chunk_bytes > 4 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail="max_chunk_bytes must be in [1024, 4194304]",
                )
            if yield_every_batches < 0 or yield_every_batches > 1000:
                raise HTTPException(
                    status_code=400,
                    detail="yield_every_batches must be in [0, 1000]",
                )

            run_dir = _resolve_run_dir(run_id)
            _, index_payload = _load_index_for_run(run_id, run_dir)
            entry = get_replay_entry_by_id(index_payload, replay_id)
            if entry is None:
                raise HTTPException(status_code=404, detail=f"replay_id not found: {replay_id}")

            replay_path_raw, replay_path = _resolve_replay_file_path(run_dir=run_dir, entry=entry)

            sent_count = 0
            has_more = False
            chunk: list[dict[str, Any]] = []
            chunk_bytes = 0
            chunks_sent = 0
            chunk_start = offset

            with _open_replay(replay_path) as handle:
                for frame_idx, line in enumerate(handle):
                    if frame_idx < offset:
                        continue
                    if sent_count >= limit:
                        has_more = True
                        break

                    payload = _parse_replay_frame_payload(line)
                    if payload is None:
                        continue

                    chunk.append(payload)
                    chunk_bytes += len(line.encode("utf-8"))
                    sent_count += 1

                    if len(chunk) >= batch_size or chunk_bytes >= max_chunk_bytes:
                        next_offset = offset + sent_count
                        chunks_sent += 1
                        await websocket.send_json(
                            {
                                "type": "frames",
                                "run_id": run_id,
                                "replay_id": replay_id,
                                "offset": chunk_start,
                                "next_offset": next_offset,
                                "count": len(chunk),
                                "has_more": True,
                                "chunk_index": chunks_sent - 1,
                                "chunk_bytes": chunk_bytes,
                                "max_chunk_bytes": max_chunk_bytes,
                                "frames": chunk,
                            }
                        )
                        chunk = []
                        chunk_bytes = 0
                        chunk_start = next_offset
                        if yield_every_batches > 0 and chunks_sent % yield_every_batches == 0:
                            await asyncio.sleep(0)

            if chunk:
                chunks_sent += 1
                await websocket.send_json(
                    {
                        "type": "frames",
                        "run_id": run_id,
                        "replay_id": replay_id,
                        "offset": chunk_start,
                        "next_offset": offset + sent_count,
                        "count": len(chunk),
                        "has_more": has_more,
                        "chunk_index": chunks_sent - 1,
                        "chunk_bytes": chunk_bytes,
                        "max_chunk_bytes": max_chunk_bytes,
                        "frames": chunk,
                    }
                )

            await websocket.send_json(
                {
                    "type": "complete",
                    "run_id": run_id,
                    "replay_id": replay_id,
                    "offset": offset,
                    "next_offset": offset + sent_count,
                    "count": sent_count,
                    "has_more": has_more,
                    "replay_path": replay_path_raw,
                    "chunks_sent": chunks_sent,
                    "batch_size": batch_size,
                    "max_chunk_bytes": max_chunk_bytes,
                    "yield_every_batches": yield_every_batches,
                }
            )
        except WebSocketDisconnect:
            return
        except HTTPException as exc:
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "status_code": exc.status_code,
                        "detail": str(exc.detail),
                    }
                )
                await websocket.close(code=1008)
            except RuntimeError:
                return

    @app.post("/api/play/session")
    def create_play_session(request: CreatePlaySessionRequest) -> dict[str, Any]:
        (
            session_id,
            seed_used,
            obs,
            info,
            session,
        ) = app.state.play_sessions.create(env_time_max=request.env_time_max, seed=request.seed)

        return {
            "session_id": session_id,
            "seed": seed_used,
            "env_time_max": session.env_time_max,
            "obs": _obs_to_list(obs),
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": info,
            "steps": session.steps,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    @app.post("/api/play/session/{session_id}/reset")
    def reset_play_session(
        session_id: str,
        request: ResetPlaySessionRequest,
    ) -> dict[str, Any]:
        seed_used, obs, info, session = app.state.play_sessions.reset(
            session_id=session_id,
            seed=request.seed,
        )
        return {
            "session_id": session_id,
            "seed": seed_used,
            "env_time_max": session.env_time_max,
            "obs": _obs_to_list(obs),
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": info,
            "steps": session.steps,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    @app.post("/api/play/session/{session_id}/step")
    def step_play_session(
        session_id: str,
        request: StepPlaySessionRequest,
    ) -> dict[str, Any]:
        obs, reward, terminated, truncated, info, session = app.state.play_sessions.step(
            session_id=session_id,
            action=request.action,
        )
        return {
            "session_id": session_id,
            "action": int(request.action),
            "obs": _obs_to_list(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
            "steps": session.steps,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    @app.delete("/api/play/session/{session_id}")
    def delete_play_session(session_id: str) -> dict[str, Any]:
        app.state.play_sessions.delete(session_id=session_id)
        return {
            "session_id": session_id,
            "deleted": True,
        }

    return app
