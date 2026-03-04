# M9 Chunk 3 Drift Guardrails Execution (2026-03-04)

## Objective

Execute Chunk 3 operational drift hardening:

- revalidate production W&B proxy and replay websocket stability,
- add explicit backend CORS validation to deployment smoke checks,
- capture strict smoke evidence with CORS included in the gate.

## Implemented changes

- `tools/smoke_m9_deployment.py`
  - Added CORS origin resolution (`--cors-origin` override or derived from `--frontend-base`).
  - Added backend CORS checks:
    - `backend-cors-simple` (GET with `Origin` header)
    - `backend-cors-preflight` (OPTIONS preflight with method/header assertions)
  - Added CORS check skip semantics when no frontend/cors origin is supplied.

- `tests/test_smoke_m9_deployment.py`
  - Added CORS helper/session fakes and regression tests for:
    - explicit origin resolution,
    - positive simple CORS behavior,
    - preflight failure detection for missing GET method allowance.

## Validation

- `python -m ruff check tools/smoke_m9_deployment.py tests/test_smoke_m9_deployment.py`
- `python -m black --check tools/smoke_m9_deployment.py tests/test_smoke_m9_deployment.py`
- `python -m pytest -q tests/test_smoke_m9_deployment.py`

## Drift-check evidence

- Production W&B status telemetry snapshot:
  - `available=true`
  - `notes_count=0`
  - `ttl_seconds=30.0`
  - `hit_rate=0.84`
  - operations present: `list_runs`, `run_history`, `run_summary`

- Strict smoke with CORS gates enabled:
  - Command:
    - `python tools/smoke_m9_deployment.py --backend-http-base https://abp-backend-production.up.railway.app --backend-ws-base wss://abp-backend-production.up.railway.app --frontend-base https://frontend-nine-sandy-47.vercel.app --require-clean-wandb-status --output-path artifacts/deploy/m9-smoke-strict-20260304-chunk3-run1.json`
  - Result:
    - `pass=true`
    - `checks=15`
    - `pass_count=15`
    - `fail_count=0`
  - Artifact:
    - `artifacts/deploy/m9-smoke-strict-20260304-chunk3-run1.json`

## Outcome

- Chunk 3 is complete.
- Deployment smoke now catches CORS drift explicitly, in addition to websocket and W&B proxy health checks.
