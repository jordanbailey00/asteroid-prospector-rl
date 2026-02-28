from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

REPLAY_SCHEMA_VERSION = 1

REQUIRED_FRAME_KEYS = (
    "schema_version",
    "frame_index",
    "t",
    "dt",
    "action",
    "reward",
    "terminated",
    "truncated",
    "render_state",
    "events",
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def frame_from_step(
    *,
    frame_index: int,
    t: int,
    dt: int,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    render_state: dict[str, Any],
    events: list[str],
    info: dict[str, Any] | None,
    include_info: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "frame_index": int(frame_index),
        "t": int(t),
        "dt": int(dt),
        "action": int(action),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "render_state": render_state,
        "events": list(events),
    }
    if include_info:
        payload["info"] = info or {}
    return payload


def validate_replay_frame(frame: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_FRAME_KEYS if key not in frame]
    if missing:
        raise ValueError(f"replay frame missing keys: {missing}")

    if int(frame["schema_version"]) != REPLAY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported replay schema version: {frame['schema_version']} "
            f"(expected {REPLAY_SCHEMA_VERSION})"
        )

    if int(frame["frame_index"]) < 0:
        raise ValueError("frame_index must be non-negative")
    if int(frame["dt"]) <= 0:
        raise ValueError("dt must be positive")

    if not isinstance(frame["render_state"], dict):
        raise ValueError("render_state must be an object")
    if not isinstance(frame["events"], list):
        raise ValueError("events must be a list")
