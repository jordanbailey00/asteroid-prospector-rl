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


def _entry_tags(entry: dict[str, Any]) -> set[str]:
    raw = entry.get("tags", [])
    if not isinstance(raw, list):
        return set()
    return {str(tag) for tag in raw}


def filter_replay_entries(
    entries: list[dict[str, Any]],
    *,
    tag: str | None = None,
    tags_any: list[str] | None = None,
    tags_all: list[str] | None = None,
    window_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    normalized_any = {str(value) for value in (tags_any or []) if str(value) != ""}
    normalized_all = {str(value) for value in (tags_all or []) if str(value) != ""}

    filtered: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        entry_tags = _entry_tags(entry)
        if tag is not None and tag not in entry_tags:
            continue
        if normalized_any and not (entry_tags & normalized_any):
            continue
        if normalized_all and not normalized_all.issubset(entry_tags):
            continue
        if window_id is not None and int(entry.get("window_id", -1)) != int(window_id):
            continue

        filtered.append(entry)

    filtered.sort(
        key=lambda item: (
            int(item.get("window_id", -1)),
            str(item.get("created_at", "")),
            str(item.get("replay_id", "")),
        ),
        reverse=True,
    )

    if limit is not None and limit >= 0:
        return filtered[:limit]
    return filtered


def get_replay_entry_by_id(index_payload: dict[str, Any], replay_id: str) -> dict[str, Any] | None:
    entries = index_payload.get("entries", [])
    if not isinstance(entries, list):
        return None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("replay_id")) == replay_id:
            return entry
    return None
