# MVP Extensive Test Plan (2026-03-05)

## Purpose

Run a high-confidence validation campaign against the completed MVP codebase across:
- deterministic core/runtime contracts,
- backend/frontend integration,
- production deployment health,
- operator training workflow behavior.

## Preconditions

- Branch: `main` at the current MVP closeout commit.
- Backend production URLs:
  - `https://abp-backend-production.up.railway.app`
  - `wss://abp-backend-production.up.railway.app`
- Frontend production URL:
  - `https://frontend-nine-sandy-47.vercel.app`
- Backend W&B scope/env configured:
  - `WANDB_API_KEY`
  - `ABP_WANDB_ENTITY`
  - `ABP_WANDB_PROJECT`
- `ABP_RUNS_ROOT` contains at least one valid run with replay frames.

## Evidence output root

Use a dated directory for tomorrow's run:
- `artifacts/validation/20260305/`

Recommended report doc for results:
- `docs/MVP_TEST_REPORT_20260305.md`

## Execution order

### 1) Baseline repo health

Run:

```powershell
python -m pytest -q
npm --prefix frontend run lint
npm --prefix frontend run build
python tools/run_parity.py --seeds 2 --steps 512 --native-library engine_core/build/abp_core.dll
```

Pass criteria:
- pytest all green (allow existing skipped tests),
- lint/build pass,
- parity passes all 12 cases.

### 2) Deployment smoke (strict)

Run local strict smoke:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://abp-backend-production.up.railway.app" \
  --backend-ws-base "wss://abp-backend-production.up.railway.app" \
  --frontend-base "https://frontend-nine-sandy-47.vercel.app" \
  --require-clean-wandb-status \
  --output-path artifacts/validation/20260305/m9-smoke-strict-20260305-local.json
```

Pass criteria:
- `pass=true`,
- `checks=15`,
- no failing CORS/WS/W&B checks.

### 3) Deployment smoke (remote CI evidence)

Trigger workflow:
- `.github/workflows/m9-deployment-smoke.yml`
- set `require_clean_wandb_status=true`

Download artifact to:
- `artifacts/validation/20260305/ci-m9-smoke-<run_id>/`

Pass criteria:
- workflow job success,
- artifact JSON reports `pass=true` with `15/15` checks.

### 4) Public route/manual UX verification

Manual checks on production:
- `/`: replay loads, frame stepping/playing works, HUD updates.
- `/play`: session create/reset/step works; action groups/onboarding visible.
- `/analytics`: KPI cards/trends/completeness table render without secret exposure.

Record:
- any console errors,
- any route/API mismatches,
- screenshots for regressions.

### 5) Operator workflow validation (PufferLib-native)

Run at least one short trainer session (Docker Linux runtime):

```powershell
docker compose -f infra/docker-compose.yml run --rm -T trainer python training/train_puffer.py --trainer-backend puffer_ppo --total-env-steps 5000 --window-env-steps 1000 --ppo-num-envs 8 --ppo-num-workers 4 --wandb-mode online --wandb-project asteroid-prospector
```

Pass criteria:
- training runs without crash,
- terminal dashboard output present,
- W&B run/artifact entries appear,
- no dependency on custom in-repo ops dashboard.

### 6) Optional soak/perf checks (time permitting)

Commands:

```powershell
python tools/bench_m7.py --output-path artifacts/validation/20260305/bench-m7-20260305.json
python tools/stability_replay_long_run.py --cycles 20 --output-path artifacts/validation/20260305/replay-stability-20260305.json
```

Pass criteria:
- benchmark runner completes,
- stability runner reports no replay index/API drift failures.

## Stop-the-line conditions

Stop promotion and file issues immediately if any occur:
- parity mismatch,
- strict smoke failure,
- replay websocket failures in strict smoke,
- W&B status unavailable or strict notes present,
- frontend route regression on `/`, `/play`, or `/analytics`.

## Reporting template (for `docs/MVP_TEST_REPORT_20260305.md`)

Include:
- exact commit hash tested,
- executed commands,
- pass/fail summary by section,
- artifact paths,
- open defects and severity,
- go/no-go recommendation.
