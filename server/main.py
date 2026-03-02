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


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)

    text = raw.strip()
    if text == "":
        return float(default)

    try:
        return float(text)
    except ValueError:
        return float(default)


def _read_optional_env(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None

    text = raw.strip()
    return text if text != "" else None


RUNS_ROOT = Path(os.environ.get("ABP_RUNS_ROOT", "runs"))
CORS_ORIGINS = _parse_csv_env("ABP_CORS_ORIGINS")
CORS_ORIGIN_REGEX = os.environ.get("ABP_CORS_ORIGIN_REGEX")
WANDB_DEFAULT_ENTITY = _read_optional_env("ABP_WANDB_ENTITY")
WANDB_DEFAULT_PROJECT = _read_optional_env("ABP_WANDB_PROJECT")
WANDB_API_KEY = _read_optional_env("WANDB_API_KEY") or _read_optional_env("ABP_WANDB_API_KEY")
WANDB_CACHE_TTL_SECONDS = _parse_float_env("ABP_WANDB_CACHE_TTL_SECONDS", 30.0)

app = create_app(
    runs_root=RUNS_ROOT,
    cors_allow_origins=CORS_ORIGINS,
    cors_allow_origin_regex=CORS_ORIGIN_REGEX,
    wandb_default_entity=WANDB_DEFAULT_ENTITY,
    wandb_default_project=WANDB_DEFAULT_PROJECT,
    wandb_api_key=WANDB_API_KEY,
    wandb_cache_ttl_seconds=WANDB_CACHE_TTL_SECONDS,
)
