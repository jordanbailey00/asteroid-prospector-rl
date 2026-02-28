from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPLAY_INDEX_SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_replay_index(*, run_id: str) -> dict[str, Any]:
    return {
        "schema_version": REPLAY_INDEX_SCHEMA_VERSION,
        "run_id": run_id,
        "updated_at": now_iso(),
        "entries": [],
    }


def load_replay_index(*, path: Path, run_id: str) -> dict[str, Any]:
    if not path.exists():
        return default_replay_index(run_id=run_id)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != REPLAY_INDEX_SCHEMA_VERSION:
        raise ValueError(
            "unsupported replay index schema version: " f"{payload.get('schema_version')}"
        )
    if str(payload.get("run_id")) != run_id:
        raise ValueError(
            f"replay index run_id mismatch: expected {run_id}, " f"got {payload.get('run_id')}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("replay index entries must be a list")
    return payload


def append_replay_entry(*, path: Path, run_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    index_payload = load_replay_index(path=path, run_id=run_id)
    index_payload["entries"].append(entry)
    index_payload["updated_at"] = now_iso()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return index_payload
