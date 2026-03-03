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

- W&B strict gate not yet executed in production:
  - backend still missing production `WANDB_API_KEY`
  - smoke checks currently run with `--skip-wandb`
- Replay/run discovery gate is still in permissive mode (`--allow-empty-runs`) because production `ABP_RUNS_ROOT` has no published run artifacts yet.
