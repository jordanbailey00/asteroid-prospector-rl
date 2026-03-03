# M9 Deployment Evidence - 2026-03-03

## Live endpoints

- Backend HTTP: `https://abp-backend-production.up.railway.app`
- Backend WS: `wss://abp-backend-production.up.railway.app`
- Frontend production alias: `https://frontend-nine-sandy-47.vercel.app`

## Hosting details

- Railway project: `asteroid-prospecting-rl-backend`
- Railway service: `abp-backend`
- Railway domain: `abp-backend-production.up.railway.app`
- Vercel project: `frontend` (`jordanbailey00s-projects` scope)

## Smoke evidence

### Local smoke (CLI)

Command:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://abp-backend-production.up.railway.app" \
  --backend-ws-base "wss://abp-backend-production.up.railway.app" \
  --frontend-base "https://frontend-nine-sandy-47.vercel.app" \
  --allow-empty-runs \
  --skip-wandb \
  --output-path artifacts/deploy/m9-smoke-live-20260303.json
```

Result:

- `pass=true`
- `checks=12`
- `pass_count=12`
- `fail_count=0`

Artifact:

- `artifacts/deploy/m9-smoke-live-20260303.json`

### GitHub Actions manual smoke

Workflow runs:

- `https://github.com/jordanbailey00/asteroid-prospector-rl/actions/runs/22639886521`
- `https://github.com/jordanbailey00/asteroid-prospector-rl/actions/runs/22640174270`

Inputs used:

- `backend_http_base=https://abp-backend-production.up.railway.app`
- `backend_ws_base=wss://abp-backend-production.up.railway.app`
- `frontend_base=https://frontend-nine-sandy-47.vercel.app`
- `allow_empty_runs=true`
- `skip_wandb=true`

Result:

- Job `smoke`: success on both manual runs
- Artifacts downloaded locally to:
  - `artifacts/deploy/ci-m9-smoke-22639886521/m9-smoke-22639886521.json`
  - `artifacts/deploy/ci-m9-smoke-22640174270/m9-smoke-22640174270.json`

## Remaining release-gate gaps

- W&B strict gate remains blocked in production until backend `WANDB_API_KEY` is configured.
- Replay websocket smoke is intermittently failing with `Unexpected websocket EOF` on `backend-replay-frames-ws`.
- Non-empty run/replay smoke gate is now enabled and passing (production `ABP_RUNS_ROOT` seeded with `runs/ws-profile-smoke`).

## Strict smoke follow-up (2026-03-03)

Backend production updates applied:

- Set Railway backend variable `ABP_WANDB_ENTITY=jordanbailey00`.
- Kept `ABP_RUNS_ROOT=runs` and deployed seeded run bundle `runs/ws-profile-smoke` to production.

Strict smoke command (no `--allow-empty-runs`, no `--skip-wandb`):

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://abp-backend-production.up.railway.app" \
  --backend-ws-base "wss://abp-backend-production.up.railway.app" \
  --frontend-base "https://frontend-nine-sandy-47.vercel.app" \
  --require-clean-wandb-status \
  --output-path artifacts/deploy/m9-smoke-strict-20260303.json
```

Result (first strict rerun):

- `pass=false`
- `checks=13`
- `pass_count=10`
- `fail_count=3`
- Non-empty run/replay gates now pass (`backend-runs-catalog`, `backend-replay-frames-http`, `backend-replay-frames-ws`).
- Remaining failures are W&B readiness (`wandb-status`, `wandb-latest-runs`, `wandb-status-post`) due missing backend `WANDB_API_KEY`.

Follow-up reruns intermittently failed `backend-replay-frames-ws` with `Unexpected websocket EOF`, producing `pass_count=9` and `fail_count=4` in artifacts like `artifacts/deploy/m9-smoke-strict-20260303-rerun3.json`.

Current blockers to clear strict smoke:

- Configure backend production `WANDB_API_KEY`.
- Stabilize production websocket replay stream reliability for `/ws/runs/{run_id}/replays/{replay_id}/frames`.
