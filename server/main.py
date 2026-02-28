from __future__ import annotations

import os
from pathlib import Path

from .app import create_app

RUNS_ROOT = Path(os.environ.get("ABP_RUNS_ROOT", "runs"))
app = create_app(runs_root=RUNS_ROOT)
