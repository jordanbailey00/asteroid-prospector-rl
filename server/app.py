from __future__ import annotations

import asyncio
import gzip
import json
import math
import random
import re
import sys
import threading
import time
from collections.abc import Callable
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
DEFAULT_WANDB_CACHE_TTL_SECONDS = 30.0
DEFAULT_WANDB_HISTORY_KEYS = (
    "_step",
    "window_id",
    "env_steps_total",
    "reward_mean",
    "return_mean",
    "profit_mean",
    "survival_rate",
)
ANALYTICS_METADATA_REQUIRED_FIELDS = (
    "run_id",
    "status",
    "trainer_backend",
    "updated_at",
    "metrics_windows_path",
    "replay_index_path",
)
ANALYTICS_WINDOW_REQUIRED_FIELDS = (
    "window_id",
    "env_steps_total",
    "reward_mean",
    "return_mean",
    "profit_mean",
    "survival_rate",
    "overheat_ticks_mean",
    "pirate_encounters_mean",
    "value_lost_to_pirates_mean",
    "mining_ticks_mean",
    "scan_count_mean",
)
ANALYTICS_REPLAY_REQUIRED_FIELDS = (
    "replay_id",
    "window_id",
    "replay_path",
    "checkpoint_path",
    "tags",
    "created_at",
)
ANALYTICS_WANDB_SUMMARY_REQUIRED_FIELDS = (
    "window_id",
    "env_steps_total",
    "reward_mean",
    "return_mean",
    "profit_mean",
    "survival_rate",
)
ANALYTICS_STATUS_PRIORITY = ("error", "missing", "stale", "ok")
WANDB_RUN_URL_RE = re.compile(r"/runs/([^/?#]+)")


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


def _parse_wandb_history_keys(value: str | None) -> list[str] | None:
    keys = _parse_csv_arg(value)
    if keys is None:
        return None
    if len(keys) > 64:
        raise HTTPException(status_code=400, detail="W&B history keys must contain at most 64 keys")
    return keys


def _extract_iteration_kpis(
    *,
    summary: dict[str, Any],
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    latest_row = history_rows[-1] if history_rows else {}
    keys = (
        "window_id",
        "env_steps_total",
        "reward_mean",
        "return_mean",
        "profit_mean",
        "survival_rate",
    )

    kpis: dict[str, Any] = {}
    for key in keys:
        if key in latest_row:
            kpis[key] = latest_row[key]
            continue
        if key in summary:
            kpis[key] = summary[key]

    if "_step" in latest_row:
        kpis["step"] = latest_row["_step"]
    return kpis


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text != "" else None


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = _clean_str(value)
    if text is None:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _path_mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _is_stale(*, updated_at: Any, now_utc: datetime, stale_after_seconds: int) -> bool:
    parsed = _parse_iso_datetime(updated_at)
    if parsed is None:
        return False
    return (now_utc - parsed).total_seconds() > float(stale_after_seconds)


def _coverage_status(
    *, error: str | None, missing_fields: list[str], empty: bool, stale: bool
) -> str:
    if error is not None:
        return "error"
    if empty or len(missing_fields) > 0:
        return "missing"
    if stale:
        return "stale"
    return "ok"


def _extract_wandb_run_id(*, metadata: dict[str, Any], run_id_override: str | None) -> str | None:
    override = _clean_str(run_id_override)
    if override is not None:
        return override

    from_metadata = _clean_str(metadata.get("wandb_run_id"))
    if from_metadata is not None:
        return from_metadata

    wandb_url = _clean_str(metadata.get("wandb_run_url"))
    if wandb_url is None:
        return None

    match = WANDB_RUN_URL_RE.search(wandb_url)
    if match is None:
        return None
    return _clean_str(match.group(1))


def _wandb_ops_notes(
    *,
    cache_ttl_seconds: float | None,
    defaults_configured: bool,
    available: bool,
    cache_payload: dict[str, Any] | None,
    operations_payload: dict[str, Any] | None,
) -> list[str]:
    notes: list[str] = []
    if not defaults_configured:
        notes.append(
            "Set ABP_WANDB_ENTITY and ABP_WANDB_PROJECT "
            "(or pass query overrides) to avoid scope errors."
        )

    if cache_ttl_seconds is not None:
        if cache_ttl_seconds <= 0.0:
            notes.append(
                "W&B proxy cache is disabled "
                "(ABP_WANDB_CACHE_TTL_SECONDS<=0); this increases rate-limit risk."
            )
        elif cache_ttl_seconds < 10.0:
            notes.append(
                "W&B proxy cache TTL is low (<10s); "
                "increase ABP_WANDB_CACHE_TTL_SECONDS if rate limits appear."
            )

    if isinstance(cache_payload, dict):
        hits = _coerce_int(cache_payload.get("hits"))
        misses = _coerce_int(cache_payload.get("misses"))
        if hits is not None and misses is not None and (hits + misses) >= 20:
            total = hits + misses
            hit_ratio = float(hits) / float(total)
            if hit_ratio < 0.2:
                notes.append(
                    "W&B cache hit ratio is low (<20% over >=20 lookups); "
                    "increase ABP_WANDB_CACHE_TTL_SECONDS or reduce dashboard query churn."
                )

    if isinstance(operations_payload, dict):
        failing_ops: list[str] = []
        for op_name in sorted(operations_payload.keys()):
            row = operations_payload.get(op_name)
            if not isinstance(row, dict):
                continue
            calls = _coerce_int(row.get("calls")) or 0
            errors = _coerce_int(row.get("errors")) or 0
            if calls > 0 and errors > 0:
                failing_ops.append(f"{op_name} ({errors}/{calls})")
        if len(failing_ops) > 0:
            notes.append("W&B proxy operations reported errors: " + ", ".join(failing_ops))

    if not available:
        notes.append(
            "W&B proxy is unavailable; verify WANDB_API_KEY and backend outbound connectivity."
        )
    return notes


@dataclass(frozen=True)
class _WandbCacheEntry:
    expires_at: float
    payload: Any


@dataclass(frozen=True)
class _WandbProxyStatus:
    available: bool
    reason: str | None
    cache: dict[str, Any] | None
    operations: dict[str, Any] | None


class _WandbProxyClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        cache_ttl_seconds: float = DEFAULT_WANDB_CACHE_TTL_SECONDS,
    ) -> None:
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._cache_lock = threading.Lock()
        self._cache: dict[str, _WandbCacheEntry] = {}
        self._api: Any | None = None
        self._unavailable_reason: str | None = None
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_expired = 0
        self._cache_sets = 0
        self._op_stats: dict[str, dict[str, float | int]] = {
            "list_runs": {
                "calls": 0,
                "errors": 0,
                "latency_ms_total": 0.0,
                "latency_ms_avg": 0.0,
            },
            "run_summary": {
                "calls": 0,
                "errors": 0,
                "latency_ms_total": 0.0,
                "latency_ms_avg": 0.0,
            },
            "run_history": {
                "calls": 0,
                "errors": 0,
                "latency_ms_total": 0.0,
                "latency_ms_avg": 0.0,
            },
        }
        try:
            import wandb  # type: ignore
        except ImportError:
            self._unavailable_reason = (
                "W&B proxy is unavailable because the 'wandb' package is not installed."
            )
            return

        kwargs: dict[str, Any] = {}
        if isinstance(api_key, str) and api_key.strip() != "":
            kwargs["api_key"] = api_key.strip()

        try:
            self._api = wandb.Api(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on local wandb install/runtime
            self._api = None
            self._unavailable_reason = (
                f"W&B proxy initialization failed: {type(exc).__name__}: {exc}"
            )

    def _api_or_raise(self) -> Any:
        if self._api is None:
            detail = self._unavailable_reason or "W&B proxy is unavailable."
            raise RuntimeError(detail)
        return self._api

    def _cache_get(self, key: str) -> Any | None:
        if self._cache_ttl_seconds <= 0.0:
            return None

        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                self._cache_misses += 1
                return None
            if entry.expires_at <= now:
                self._cache.pop(key, None)
                self._cache_expired += 1
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            return entry.payload

    def _cache_set(self, key: str, payload: Any) -> None:
        if self._cache_ttl_seconds <= 0.0:
            return
        with self._cache_lock:
            self._cache[key] = _WandbCacheEntry(
                expires_at=time.monotonic() + self._cache_ttl_seconds,
                payload=payload,
            )
            self._cache_sets += 1

    def _record_operation(self, *, name: str, elapsed_ms: float, ok: bool) -> None:
        with self._cache_lock:
            row = self._op_stats.setdefault(
                name,
                {"calls": 0, "errors": 0, "latency_ms_total": 0.0, "latency_ms_avg": 0.0},
            )
            calls = int(row.get("calls", 0)) + 1
            errors = int(row.get("errors", 0)) + (0 if ok else 1)
            total = float(row.get("latency_ms_total", 0.0)) + float(elapsed_ms)
            row["calls"] = calls
            row["errors"] = errors
            row["latency_ms_total"] = total
            row["latency_ms_avg"] = total / float(calls) if calls > 0 else 0.0

    def diagnostics(self) -> _WandbProxyStatus:
        with self._cache_lock:
            lookups_total = int(self._cache_hits + self._cache_misses)
            cache_payload = {
                "ttl_seconds": float(self._cache_ttl_seconds),
                "entries": len(self._cache),
                "hits": int(self._cache_hits),
                "misses": int(self._cache_misses),
                "expired": int(self._cache_expired),
                "sets": int(self._cache_sets),
                "hit_rate": (
                    float(self._cache_hits) / float(lookups_total) if lookups_total > 0 else None
                ),
            }
            operations_payload = {
                name: {
                    "calls": int(values.get("calls", 0)),
                    "errors": int(values.get("errors", 0)),
                    "latency_ms_total": float(values.get("latency_ms_total", 0.0)),
                    "latency_ms_avg": float(values.get("latency_ms_avg", 0.0)),
                }
                for name, values in self._op_stats.items()
            }
        return _WandbProxyStatus(
            available=self._api is not None,
            reason=self._unavailable_reason,
            cache=cache_payload,
            operations=operations_payload,
        )

    def _json_safe(self, value: Any, *, depth: int = 0) -> Any:
        if depth >= 5:
            return str(value)
        if value is None or isinstance(value, bool | int | str):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for idx, (key, child) in enumerate(value.items()):
                if idx >= 256:
                    break
                out[str(key)] = self._json_safe(child, depth=depth + 1)
            return out
        if isinstance(value, list | tuple | set):
            return [self._json_safe(child, depth=depth + 1) for child in list(value)[:512]]
        return str(value)

    def _summary_payload(self, summary: Any) -> dict[str, Any]:
        raw: Any = summary
        if hasattr(summary, "_json_dict"):
            raw = summary._json_dict
        if hasattr(raw, "items"):
            return self._json_safe(dict(raw), depth=0)
        return {}

    def _serialize_run_lite(self, run: Any) -> dict[str, Any]:
        summary = self._summary_payload(getattr(run, "summary", {}))
        summary_preview: dict[str, Any] = {}
        for key in DEFAULT_WANDB_HISTORY_KEYS:
            if key in summary:
                summary_preview[key] = summary[key]
        if "windows_emitted" in summary:
            summary_preview["windows_emitted"] = summary["windows_emitted"]
        if "episodes_total" in summary:
            summary_preview["episodes_total"] = summary["episodes_total"]

        return {
            "run_id": str(getattr(run, "id", "")),
            "name": self._json_safe(getattr(run, "name", None)),
            "state": self._json_safe(getattr(run, "state", None)),
            "url": self._json_safe(getattr(run, "url", None)),
            "created_at": self._json_safe(getattr(run, "created_at", None)),
            "updated_at": self._json_safe(getattr(run, "heartbeat_at", None)),
            "summary_preview": summary_preview,
        }

    def list_runs(self, *, entity: str, project: str, limit: int) -> list[dict[str, Any]]:
        start = time.perf_counter()
        ok = False
        try:
            cache_key = json.dumps(
                {
                    "op": "list_runs",
                    "entity": entity,
                    "project": project,
                    "limit": int(limit),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                ok = True
                return cached

            api = self._api_or_raise()
            path = f"{entity}/{project}"
            rows: list[dict[str, Any]] = []
            for idx, run in enumerate(
                api.runs(path=path, per_page=min(max(limit, 1), 100), order="-created_at")
            ):
                if idx >= limit:
                    break
                rows.append(self._serialize_run_lite(run))

            self._cache_set(cache_key, rows)
            ok = True
            return rows
        finally:
            self._record_operation(
                name="list_runs",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                ok=ok,
            )

    def get_run_summary(self, *, entity: str, project: str, run_id: str) -> dict[str, Any]:
        start = time.perf_counter()
        ok = False
        try:
            cache_key = json.dumps(
                {
                    "op": "run_summary",
                    "entity": entity,
                    "project": project,
                    "run_id": run_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                ok = True
                return cached

            api = self._api_or_raise()
            run = api.run(path=f"{entity}/{project}/{run_id}")
            summary = self._summary_payload(getattr(run, "summary", {}))
            payload = {
                "run_id": str(getattr(run, "id", run_id)),
                "name": self._json_safe(getattr(run, "name", None)),
                "state": self._json_safe(getattr(run, "state", None)),
                "url": self._json_safe(getattr(run, "url", None)),
                "created_at": self._json_safe(getattr(run, "created_at", None)),
                "updated_at": self._json_safe(getattr(run, "heartbeat_at", None)),
                "config": self._json_safe(dict(getattr(run, "config", {}) or {})),
                "summary": summary,
            }

            self._cache_set(cache_key, payload)
            ok = True
            return payload
        finally:
            self._record_operation(
                name="run_summary",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                ok=ok,
            )

    def get_run_history(
        self,
        *,
        entity: str,
        project: str,
        run_id: str,
        keys: list[str] | None,
        max_points: int,
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        ok = False
        try:
            cache_key = json.dumps(
                {
                    "op": "run_history",
                    "entity": entity,
                    "project": project,
                    "run_id": run_id,
                    "keys": keys or [],
                    "max_points": int(max_points),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                ok = True
                return cached

            api = self._api_or_raise()
            run = api.run(path=f"{entity}/{project}/{run_id}")
            scan_kwargs: dict[str, Any] = {
                "page_size": min(max(1, int(max_points)), 1000),
            }
            if keys is not None and len(keys) > 0:
                scan_kwargs["keys"] = list(keys)

            rows: list[dict[str, Any]] = []
            for idx, row in enumerate(run.scan_history(**scan_kwargs)):
                if idx >= max_points:
                    break
                if not isinstance(row, dict):
                    continue

                clean_row: dict[str, Any] = {}
                for key, value in row.items():
                    if not isinstance(key, str):
                        continue
                    clean_row[key] = self._json_safe(value)
                rows.append(clean_row)

            self._cache_set(cache_key, rows)
            ok = True
            return rows
        finally:
            self._record_operation(
                name="run_history",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                ok=ok,
            )


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
    wandb_proxy: Any | None = None,
    wandb_default_entity: str | None = None,
    wandb_default_project: str | None = None,
    wandb_api_key: str | None = None,
    wandb_cache_ttl_seconds: float = DEFAULT_WANDB_CACHE_TTL_SECONDS,
) -> FastAPI:
    app = FastAPI(title="Asteroid Prospector API", version="0.2.0")
    app.state.runs_root = runs_root
    app.state.play_sessions = _PlaySessionStore()
    app.state.wandb_proxy = (
        wandb_proxy
        if wandb_proxy is not None
        else _WandbProxyClient(
            api_key=wandb_api_key,
            cache_ttl_seconds=wandb_cache_ttl_seconds,
        )
    )
    app.state.wandb_default_entity = (
        wandb_default_entity.strip()
        if isinstance(wandb_default_entity, str) and wandb_default_entity.strip() != ""
        else None
    )
    app.state.wandb_default_project = (
        wandb_default_project.strip()
        if isinstance(wandb_default_project, str) and wandb_default_project.strip() != ""
        else None
    )

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

    def _resolve_wandb_scope(
        *,
        entity: str | None,
        project: str | None,
    ) -> tuple[str, str]:
        resolved_entity = (
            entity.strip() if isinstance(entity, str) and entity.strip() != "" else None
        ) or app.state.wandb_default_entity
        resolved_project = (
            project.strip() if isinstance(project, str) and project.strip() != "" else None
        ) or app.state.wandb_default_project

        if resolved_entity is None or resolved_project is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "W&B entity/project not configured. Provide ?entity=...&project=... "
                    "or configure ABP_WANDB_ENTITY and ABP_WANDB_PROJECT."
                ),
            )
        return str(resolved_entity), str(resolved_project)

    def _call_wandb_proxy(call: Callable[[], Any]) -> Any:
        proxy = app.state.wandb_proxy
        if proxy is None:
            raise HTTPException(status_code=503, detail="W&B proxy is unavailable.")
        try:
            return call()
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"W&B proxy request failed: {type(exc).__name__}: {exc}",
            ) from exc

    def _wandb_proxy_diagnostics() -> dict[str, Any]:
        proxy = app.state.wandb_proxy
        if proxy is None:
            return {
                "available": False,
                "reason": "W&B proxy is unavailable.",
                "cache": None,
                "operations": None,
            }

        diagnostics_fn = getattr(proxy, "diagnostics", None)
        if callable(diagnostics_fn):
            payload = diagnostics_fn()
            if isinstance(payload, _WandbProxyStatus):
                return {
                    "available": bool(payload.available),
                    "reason": payload.reason,
                    "cache": payload.cache,
                    "operations": payload.operations,
                }
            if isinstance(payload, dict):
                return {
                    "available": bool(payload.get("available", True)),
                    "reason": payload.get("reason"),
                    "cache": payload.get("cache"),
                    "operations": payload.get("operations"),
                }

        return {
            "available": True,
            "reason": None,
            "cache": None,
            "operations": None,
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/wandb/status")
    def wandb_status() -> dict[str, Any]:
        diagnostics = _wandb_proxy_diagnostics()
        available = bool(diagnostics.get("available", True))
        cache_payload = diagnostics.get("cache")

        cache_ttl_seconds: float | None = None
        if isinstance(cache_payload, dict):
            raw_ttl = cache_payload.get("ttl_seconds")
            if isinstance(raw_ttl, int | float):
                cache_ttl_seconds = float(raw_ttl)

        defaults_configured = (
            app.state.wandb_default_entity is not None
            and app.state.wandb_default_project is not None
        )
        operations_payload = diagnostics.get("operations")

        notes = _wandb_ops_notes(
            cache_ttl_seconds=cache_ttl_seconds,
            defaults_configured=defaults_configured,
            available=available,
            cache_payload=cache_payload if isinstance(cache_payload, dict) else None,
            operations_payload=(
                operations_payload if isinstance(operations_payload, dict) else None
            ),
        )

        reason = diagnostics.get("reason")
        return {
            "available": available,
            "reason": reason if isinstance(reason, str) and reason.strip() != "" else None,
            "defaults": {
                "entity": app.state.wandb_default_entity,
                "project": app.state.wandb_default_project,
            },
            "cache": cache_payload if isinstance(cache_payload, dict) else None,
            "operations": operations_payload if isinstance(operations_payload, dict) else None,
            "notes": notes,
        }

    @app.get("/api/wandb/runs/latest")
    def wandb_list_latest_runs(
        limit: int = Query(default=10, ge=1, le=50),
        entity: str | None = Query(default=None),
        project: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_entity, resolved_project = _resolve_wandb_scope(entity=entity, project=project)
        runs = _call_wandb_proxy(
            lambda: app.state.wandb_proxy.list_runs(
                entity=resolved_entity,
                project=resolved_project,
                limit=int(limit),
            )
        )
        return {
            "entity": resolved_entity,
            "project": resolved_project,
            "count": len(runs),
            "runs": runs,
        }

    @app.get("/api/wandb/runs/{wandb_run_id}/summary")
    def wandb_get_run_summary(
        wandb_run_id: str,
        entity: str | None = Query(default=None),
        project: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_entity, resolved_project = _resolve_wandb_scope(entity=entity, project=project)
        payload = _call_wandb_proxy(
            lambda: app.state.wandb_proxy.get_run_summary(
                entity=resolved_entity,
                project=resolved_project,
                run_id=wandb_run_id,
            )
        )
        return {
            "entity": resolved_entity,
            "project": resolved_project,
            "run": payload,
        }

    @app.get("/api/wandb/runs/{wandb_run_id}/history")
    def wandb_get_run_history(
        wandb_run_id: str,
        keys: str | None = Query(default=None),
        max_points: int = Query(default=1000, ge=1, le=5000),
        entity: str | None = Query(default=None),
        project: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_entity, resolved_project = _resolve_wandb_scope(entity=entity, project=project)
        history_keys = _parse_wandb_history_keys(keys)
        rows = _call_wandb_proxy(
            lambda: app.state.wandb_proxy.get_run_history(
                entity=resolved_entity,
                project=resolved_project,
                run_id=wandb_run_id,
                keys=history_keys,
                max_points=int(max_points),
            )
        )
        return {
            "entity": resolved_entity,
            "project": resolved_project,
            "run_id": wandb_run_id,
            "keys": history_keys,
            "count": len(rows),
            "rows": rows,
        }

    @app.get("/api/wandb/runs/{wandb_run_id}/iteration-view")
    def wandb_get_iteration_view(
        wandb_run_id: str,
        keys: str | None = Query(default=None),
        max_points: int = Query(default=1000, ge=1, le=5000),
        entity: str | None = Query(default=None),
        project: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_entity, resolved_project = _resolve_wandb_scope(entity=entity, project=project)
        history_keys = _parse_wandb_history_keys(keys)
        if history_keys is None:
            history_keys = list(DEFAULT_WANDB_HISTORY_KEYS)

        run_payload = _call_wandb_proxy(
            lambda: app.state.wandb_proxy.get_run_summary(
                entity=resolved_entity,
                project=resolved_project,
                run_id=wandb_run_id,
            )
        )
        history_rows = _call_wandb_proxy(
            lambda: app.state.wandb_proxy.get_run_history(
                entity=resolved_entity,
                project=resolved_project,
                run_id=wandb_run_id,
                keys=history_keys,
                max_points=int(max_points),
            )
        )
        run_summary = run_payload.get("summary", {})
        if not isinstance(run_summary, dict):
            run_summary = {}

        return {
            "entity": resolved_entity,
            "project": resolved_project,
            "run": run_payload,
            "history": {
                "keys": history_keys,
                "count": len(history_rows),
                "rows": history_rows,
            },
            "kpis": _extract_iteration_kpis(summary=run_summary, history_rows=history_rows),
        }

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

    @app.get("/api/runs/{run_id}/analytics/completeness")
    def get_run_analytics_completeness(
        run_id: str,
        stale_after_seconds: int = Query(default=21600, ge=60, le=604800),
        wandb_run_id: str | None = Query(default=None),
        wandb_entity: str | None = Query(default=None),
        wandb_project: str | None = Query(default=None),
        wandb_history_max_points: int = Query(default=1000, ge=1, le=5000),
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_id)
        metadata_path = run_dir / "run_metadata.json"
        metadata = _load_run_metadata(run_dir)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"run metadata not found: {run_id}")

        now_utc = datetime.now(UTC)
        metrics_path = _resolve_metrics_windows_path(run_dir=run_dir, metadata=metadata)
        metrics_path_rel = _as_relative_posix(metrics_path, start=run_dir)
        replay_index_path = _resolve_replay_index_path(run_dir=run_dir, metadata=metadata)
        replay_index_path_rel = _as_relative_posix(replay_index_path, start=run_dir)

        coverage: list[dict[str, Any]] = []

        def _append_coverage(
            *,
            key: str,
            label: str,
            observed_count: int,
            required_fields: tuple[str, ...],
            missing_fields: list[str],
            source: str,
            source_path: str | None,
            source_updated_at: str | None,
            notes: list[str],
            error: str | None = None,
            empty: bool = False,
        ) -> None:
            stale = _is_stale(
                updated_at=source_updated_at,
                now_utc=now_utc,
                stale_after_seconds=int(stale_after_seconds),
            )
            status = _coverage_status(
                error=error,
                missing_fields=missing_fields,
                empty=empty,
                stale=stale,
            )
            row_notes = list(notes)
            if error is not None:
                row_notes.append(error)
            if stale and source_updated_at is not None:
                row_notes.append(
                    f"source is older than stale threshold ({int(stale_after_seconds)}s)"
                )

            coverage.append(
                {
                    "key": key,
                    "label": label,
                    "status": status,
                    "observed_count": int(observed_count),
                    "required_fields": list(required_fields),
                    "missing_fields": sorted(set(missing_fields)),
                    "lineage": {
                        "source": source,
                        "path": source_path,
                        "updated_at": source_updated_at,
                    },
                    "notes": row_notes,
                }
            )

        metadata_missing = [
            field
            for field in ANALYTICS_METADATA_REQUIRED_FIELDS
            if not _value_present(metadata.get(field))
        ]
        metadata_updated_at = _clean_str(metadata.get("updated_at")) or _path_mtime_iso(
            metadata_path
        )
        _append_coverage(
            key="run_metadata",
            label="Run Metadata",
            observed_count=len(metadata),
            required_fields=ANALYTICS_METADATA_REQUIRED_FIELDS,
            missing_fields=metadata_missing,
            source="run_metadata.json",
            source_path="run_metadata.json",
            source_updated_at=metadata_updated_at,
            notes=[],
            empty=False,
        )

        metric_rows: list[dict[str, Any]] = []
        metrics_error: str | None = None
        if metrics_path.exists():
            try:
                metric_rows = _load_jsonl_rows(metrics_path)
            except HTTPException as exc:
                metrics_error = f"{exc.status_code}: {exc.detail}"
        latest_metric_row = metric_rows[-1] if len(metric_rows) > 0 else {}
        metric_missing = [
            field
            for field in ANALYTICS_WINDOW_REQUIRED_FIELDS
            if not _value_present(latest_metric_row.get(field))
        ]
        _append_coverage(
            key="window_metrics",
            label="Window Metrics",
            observed_count=len(metric_rows),
            required_fields=ANALYTICS_WINDOW_REQUIRED_FIELDS,
            missing_fields=metric_missing,
            source="metrics/windows.jsonl",
            source_path=metrics_path_rel,
            source_updated_at=_path_mtime_iso(metrics_path),
            notes=[] if metrics_path.exists() else ["metrics windows file not found"],
            error=metrics_error,
            empty=(not metrics_path.exists()) or (len(metric_rows) == 0),
        )

        replay_entries: list[dict[str, Any]] = []
        replay_updated_at: str | None = None
        replay_error: str | None = None
        if replay_index_path.exists():
            try:
                replay_payload = load_replay_index(path=replay_index_path, run_id=run_id)
                replay_updated_at = _clean_str(replay_payload.get("updated_at")) or _path_mtime_iso(
                    replay_index_path
                )
                maybe_entries = replay_payload.get("entries")
                if isinstance(maybe_entries, list):
                    replay_entries = [row for row in maybe_entries if isinstance(row, dict)]
                else:
                    replay_error = "500: Invalid replay index entries payload"
            except HTTPException as exc:
                replay_error = f"{exc.status_code}: {exc.detail}"
        latest_replay = replay_entries[-1] if len(replay_entries) > 0 else {}
        replay_missing = [
            field
            for field in ANALYTICS_REPLAY_REQUIRED_FIELDS
            if not _value_present(latest_replay.get(field))
        ]
        _append_coverage(
            key="replay_timeline",
            label="Replay Timeline",
            observed_count=len(replay_entries),
            required_fields=ANALYTICS_REPLAY_REQUIRED_FIELDS,
            missing_fields=replay_missing,
            source="replay_index.json",
            source_path=replay_index_path_rel,
            source_updated_at=replay_updated_at or _path_mtime_iso(replay_index_path),
            notes=[] if replay_index_path.exists() else ["replay index file not found"],
            error=replay_error,
            empty=(not replay_index_path.exists()) or (len(replay_entries) == 0),
        )

        resolved_wandb_run_id = _extract_wandb_run_id(
            metadata=metadata,
            run_id_override=wandb_run_id,
        )
        resolved_wandb_entity: str | None = None
        resolved_wandb_project: str | None = None
        wandb_scope_error: str | None = None

        if resolved_wandb_run_id is not None:
            try:
                resolved_wandb_entity, resolved_wandb_project = _resolve_wandb_scope(
                    entity=wandb_entity,
                    project=wandb_project,
                )
            except HTTPException as exc:
                if exc.status_code == 400:
                    wandb_scope_error = str(exc.detail)
                else:
                    raise

        summary_payload: dict[str, Any] | None = None
        summary_error: str | None = None
        if resolved_wandb_run_id is not None and wandb_scope_error is None:
            try:
                summary_payload = _call_wandb_proxy(
                    lambda: app.state.wandb_proxy.get_run_summary(
                        entity=str(resolved_wandb_entity),
                        project=str(resolved_wandb_project),
                        run_id=resolved_wandb_run_id,
                    )
                )
            except HTTPException as exc:
                summary_error = f"{exc.status_code}: {exc.detail}"

        wandb_summary = (
            summary_payload.get("summary", {}) if isinstance(summary_payload, dict) else {}
        )
        if not isinstance(wandb_summary, dict):
            wandb_summary = {}
        wandb_summary_missing = [
            field
            for field in ANALYTICS_WANDB_SUMMARY_REQUIRED_FIELDS
            if not _value_present(wandb_summary.get(field))
        ]
        wandb_summary_notes: list[str] = []
        if resolved_wandb_run_id is None:
            wandb_summary_notes.append(
                "W&B run id unavailable; pass wandb_run_id query param or populate "
                "metadata.wandb_run_url"
            )
        if wandb_scope_error is not None:
            wandb_summary_notes.append(wandb_scope_error)
        _append_coverage(
            key="wandb_summary",
            label="W&B Summary",
            observed_count=len(wandb_summary),
            required_fields=ANALYTICS_WANDB_SUMMARY_REQUIRED_FIELDS,
            missing_fields=wandb_summary_missing,
            source="wandb.summary",
            source_path="/api/wandb/runs/{wandb_run_id}/summary",
            source_updated_at=(
                _clean_str(summary_payload.get("updated_at"))
                if isinstance(summary_payload, dict)
                else None
            ),
            notes=wandb_summary_notes,
            error=summary_error,
            empty=(summary_payload is None),
        )

        history_rows: list[dict[str, Any]] = []
        history_error: str | None = None
        if resolved_wandb_run_id is not None and wandb_scope_error is None:
            try:
                history_rows = _call_wandb_proxy(
                    lambda: app.state.wandb_proxy.get_run_history(
                        entity=str(resolved_wandb_entity),
                        project=str(resolved_wandb_project),
                        run_id=resolved_wandb_run_id,
                        keys=list(DEFAULT_WANDB_HISTORY_KEYS),
                        max_points=int(wandb_history_max_points),
                    )
                )
            except HTTPException as exc:
                history_error = f"{exc.status_code}: {exc.detail}"

        latest_history = history_rows[-1] if len(history_rows) > 0 else {}
        history_missing = [
            field
            for field in DEFAULT_WANDB_HISTORY_KEYS
            if not _value_present(latest_history.get(field))
        ]
        history_notes: list[str] = []
        if resolved_wandb_run_id is None:
            history_notes.append(
                "W&B run id unavailable; pass wandb_run_id query param or populate "
                "metadata.wandb_run_url"
            )
        if wandb_scope_error is not None:
            history_notes.append(wandb_scope_error)
        _append_coverage(
            key="wandb_history",
            label="W&B History",
            observed_count=len(history_rows),
            required_fields=tuple(DEFAULT_WANDB_HISTORY_KEYS),
            missing_fields=history_missing,
            source="wandb.history",
            source_path="/api/wandb/runs/{wandb_run_id}/history",
            source_updated_at=(
                _clean_str(summary_payload.get("updated_at"))
                if isinstance(summary_payload, dict)
                else None
            ),
            notes=history_notes,
            error=history_error,
            empty=(len(history_rows) == 0),
        )

        status_counts = {status: 0 for status in ANALYTICS_STATUS_PRIORITY}
        for row in coverage:
            status = str(row.get("status", ""))
            if status in status_counts:
                status_counts[status] += 1

        overall_status = "ok"
        for candidate in ANALYTICS_STATUS_PRIORITY:
            if status_counts.get(candidate, 0) > 0:
                overall_status = candidate
                break

        run_context = {
            "trainer_backend": metadata.get("trainer_backend"),
            "status": metadata.get("status"),
            "started_at": metadata.get("started_at"),
            "updated_at": metadata.get("updated_at"),
            "finished_at": metadata.get("finished_at"),
            "run_config_path": metadata.get("config_path"),
            "metrics_windows_path": metrics_path_rel,
            "replay_index_path": replay_index_path_rel,
            "wandb_run_url": metadata.get("wandb_run_url"),
            "constellation_url": metadata.get("constellation_url"),
        }

        return {
            "run_id": run_id,
            "generated_at": now_utc.isoformat(),
            "stale_after_seconds": int(stale_after_seconds),
            "overall_status": overall_status,
            "status_counts": status_counts,
            "run_context": run_context,
            "wandb_scope": {
                "entity": resolved_wandb_entity,
                "project": resolved_wandb_project,
                "run_id": resolved_wandb_run_id,
                "scope_error": wandb_scope_error,
            },
            "coverage": coverage,
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
