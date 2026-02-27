# Asteroid Belt Prospector

Current milestone: **M1 (Python reference environment baseline)**.

## Repo layout

- `engine_core/`: C core scaffold (authoritative engine target in later milestones)
- `python/`: Python env package (`HelloProspectorEnv` + `ProspectorReferenceEnv`)
- `server/`: API server scaffold
- `frontend/`: Web UI scaffold
- `training/`: trainer/evaluator scaffold (placeholder)
- `replay/`: replay pipeline scaffold (placeholder)
- `tests/`: Tier-0/1/2 tests for the current Python implementation
- `tools/`: local quality/check scripts

## Quick start (PowerShell)

```powershell
python -m pip install -U pip
python -m pip install numpy pytest black ruff pre-commit clang-format
pre-commit install
```

## Quality gates

Run local format/lint/test checks:

```powershell
.\tools\run_checks.ps1
```

Run tests only:

```powershell
pytest -q
```

## Environment stubs

M0 contract-only env:

```powershell
$env:PYTHONPATH = "python"
python -c "from asteroid_prospector import HelloProspectorEnv; e=HelloProspectorEnv(); o,_=e.reset(seed=1); print(o.shape, o.dtype, e.action_space.n)"
```

M1 reference env:

```powershell
$env:PYTHONPATH = "python"
python -c "from asteroid_prospector import ProspectorReferenceEnv; e=ProspectorReferenceEnv(seed=1); o,_=e.reset(seed=1); print(o.shape, o.dtype, e.action_space.n); print(e.step(6)[1:])"
```

Expected interface values remain frozen:
- observation shape `(260,)`
- action space size `69` (`0..68`)
