from __future__ import annotations

import os
from pathlib import Path

from ops_console.app import create_ops_console_app


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_REPO_ROOT = Path(os.getenv("ABP_OPS_REPO_ROOT", Path(__file__).resolve().parents[1]))
_RUNS_ROOT = Path(os.getenv("ABP_OPS_RUNS_ROOT", _REPO_ROOT / "runs"))
_LOCAL_ONLY = _env_bool("ABP_OPS_ENFORCE_LOCAL_ONLY", default=True)

app = create_ops_console_app(
    repo_root=_REPO_ROOT,
    runs_root=_RUNS_ROOT,
    enforce_local_only=_LOCAL_ONLY,
)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ABP_OPS_HOST", "127.0.0.1")
    port = int(os.getenv("ABP_OPS_PORT", "8090"))
    uvicorn.run("ops_console.main:app", host=host, port=port, reload=False)
