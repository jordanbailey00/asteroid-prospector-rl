# Asteroid Belt Prospector (M0)

This repository is currently at **Milestone M0**: scaffold + contract-only environment stub.

## Repo layout

- `engine_core/`: C core scaffold (authoritative engine target in later milestones)
- `python/`: Python env wrapper scaffold + M0 hello environment
- `server/`: API server scaffold
- `frontend/`: Web UI scaffold
- `training/`: trainer/evaluator scaffold (placeholder)
- `replay/`: replay pipeline scaffold (placeholder)
- `tests/`: Tier-0 contract tests for the M0 stub
- `tools/`: local check scripts

## Quick start (PowerShell)

```powershell
python -m pip install -U pip
python -m pip install numpy pytest black ruff pre-commit
pre-commit install
```

Run local quality gates (format/lint/test):

```powershell
.\tools\run_checks.ps1
```

Run tests only:

```powershell
pytest -q
```

Smoke-test the M0 env contract:

```powershell
$env:PYTHONPATH = "python"
python -c "from asteroid_prospector import HelloProspectorEnv; e = HelloProspectorEnv(); o, _ = e.reset(seed=123); print(o.shape, o.dtype, e.action_space.n)"
```

Expected output includes `(260,) float32 69`.
