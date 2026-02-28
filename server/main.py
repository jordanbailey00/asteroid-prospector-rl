from __future__ import annotations

import os
from pathlib import Path

from .app import create_app


def _parse_csv_env(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    if raw is None:
        return None

    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values


RUNS_ROOT = Path(os.environ.get("ABP_RUNS_ROOT", "runs"))
CORS_ORIGINS = _parse_csv_env("ABP_CORS_ORIGINS")
CORS_ORIGIN_REGEX = os.environ.get("ABP_CORS_ORIGIN_REGEX")

app = create_app(
    runs_root=RUNS_ROOT,
    cors_allow_origins=CORS_ORIGINS,
    cors_allow_origin_regex=CORS_ORIGIN_REGEX,
)
