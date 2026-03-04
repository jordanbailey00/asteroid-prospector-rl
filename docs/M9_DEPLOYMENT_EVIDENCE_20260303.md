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

- W&B strict gate: resolved (production backend `WANDB_API_KEY` + scoped entity/project are active).
- Replay websocket strict smoke gate: resolved after close-handshake hardening (default retries `3` now stable in production evidence).
- Non-empty run/replay smoke gate: enabled and passing (production `ABP_RUNS_ROOT` seeded with `runs/ws-profile-smoke`).
- Remaining MVP gaps are now baseline bot implementation and benchmark automation (tracked in `docs/PROJECT_STATUS.md`).

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

## W&B credentials activation (2026-03-03)

Applied production backend scope/auth updates:

- Set backend `WANDB_API_KEY` in Railway secret vars.
- Updated backend default W&B scope to:
  - `ABP_WANDB_ENTITY=jordanbaileypmp-georgia-institute-of-technology`
  - `ABP_WANDB_PROJECT=asteroid-prospector`
- Bootstrapped project visibility with a minimal run (`run_id=de2xq5nz`) under that scope.

Strict smoke rerun (W&B required, non-empty runs required):

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://abp-backend-production.up.railway.app" \
  --backend-ws-base "wss://abp-backend-production.up.railway.app" \
  --frontend-base "https://frontend-nine-sandy-47.vercel.app" \
  --require-clean-wandb-status \
  --output-path artifacts/deploy/m9-smoke-strict-20260303-post-wandb-attempt1.json
```

Result:

- `pass=true`
- `checks=13`
- `pass_count=13`
- `fail_count=0`

## Websocket resilience deploy + strict smoke rerun (2026-03-04)

Production deploy action:

```powershell
railway up --service abp-backend --environment production --ci --message "deploy 394e7af websocket resilience patch"
```

Deploy result:

- Railway deploy: success
- Build logs: `https://railway.com/project/d526c56e-db13-4e5e-916a-e76ec86d9b2e/service/4317e6ca-fe17-40fc-921b-7aedd395564f?id=e7642084-875d-4e33-9513-1f6007b09a43`

Strict smoke reruns (default websocket retries = 3):

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://abp-backend-production.up.railway.app" \
  --backend-ws-base "wss://abp-backend-production.up.railway.app" \
  --frontend-base "https://frontend-nine-sandy-47.vercel.app" \
  --require-clean-wandb-status \
  --output-path artifacts/deploy/m9-smoke-strict-20260303-post-ws-hardening-attempt1.json
```

Repeat runs using the same command were captured to:

- `artifacts/deploy/m9-smoke-strict-20260303-post-ws-hardening-attempt1.json`
- `artifacts/deploy/m9-smoke-strict-20260303-post-ws-hardening-attempt2.json`
- `artifacts/deploy/m9-smoke-strict-20260303-post-ws-hardening-attempt3.json`
- `artifacts/deploy/m9-smoke-strict-20260303-post-ws-hardening-attempt4.json`

Observed result across attempts 1-4:

- `pass=false`
- `checks=13`
- `pass_count=12`
- `fail_count=1`
- Consistent failing check: `backend-replay-frames-ws` with `RuntimeError: Unexpected websocket EOF`

Diagnostic strict smoke with higher retry cap:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://abp-backend-production.up.railway.app" \
  --backend-ws-base "wss://abp-backend-production.up.railway.app" \
  --frontend-base "https://frontend-nine-sandy-47.vercel.app" \
  --require-clean-wandb-status \
  --ws-check-attempts 10 \
  --output-path artifacts/deploy/m9-smoke-strict-20260303-post-ws-hardening-attempt5-ws10.json
```

Diagnostic result:

- `pass=true`
- `checks=13`
- `pass_count=13`
- `fail_count=0`
- `backend-replay-frames-ws` recovered on `attempt=5/10`

Conclusion:

- Deployment of commit `394e7af` is live.
- At this stage (pre-close-handshake patch), strict smoke with default retry budget (`3`) still failed due intermittent websocket EOF on production replay frames.
- This release-gate-open conclusion is superseded by the 2026-03-04 close-handshake section below.


## Websocket close-handshake deploy + strict reliability rerun (2026-03-04)

Production deploy action:

```powershell
railway up --service abp-backend --environment production --ci --message "deploy websocket close-handshake stabilization"
```

Deploy result:

- Railway deploy: success
- Build logs: `https://railway.com/project/d526c56e-db13-4e5e-916a-e76ec86d9b2e/service/4317e6ca-fe17-40fc-921b-7aedd395564f?id=c37c2505-f896-4dab-be98-c7801c544d43&`

Strict smoke command template (default websocket retries = 3):

```powershell
python tools/smoke_m9_deployment.py   --backend-http-base "https://abp-backend-production.up.railway.app"   --backend-ws-base "wss://abp-backend-production.up.railway.app"   --frontend-base "https://frontend-nine-sandy-47.vercel.app"   --require-clean-wandb-status   --output-path artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-runN.json
```

Artifacts captured:

- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run1.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run2.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run3.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run4.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run5.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run6.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run7.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run8.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run9.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run10.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run11.json`
- `artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run12.json`

Observed result across runs 1-12:

- `pass=true` for all 12 runs
- `checks=13`
- `pass_count=13`
- `fail_count=0`
- `backend-replay-frames-ws` succeeded with `WS replay frames returned message type=frames` on every run.

Conclusion:

- Strict deployment smoke now passes reliably in production with default websocket retry budget (`--ws-check-attempts=3`).
- Replay websocket release gate is closed for this patch line.

## Release-cut evidence refresh (2026-03-04)

Local strict smoke rerun command:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://abp-backend-production.up.railway.app" \
  --backend-ws-base "wss://abp-backend-production.up.railway.app" \
  --frontend-base "https://frontend-nine-sandy-47.vercel.app" \
  --require-clean-wandb-status \
  --output-path artifacts/deploy/m9-smoke-strict-20260304-chunk1-run1.json
```

Local strict rerun result:

- `pass=true`
- `checks=13`
- `pass_count=13`
- `fail_count=0`

Local artifact:

- `artifacts/deploy/m9-smoke-strict-20260304-chunk1-run1.json`

Manual GitHub Actions strict smoke rerun:

- Workflow run: `https://github.com/jordanbailey00/asteroid-prospector-rl/actions/runs/22655188824`
- Result: `success`
- Downloaded artifact:
  - `artifacts/deploy/ci-m9-smoke-22655188824/m9-deployment-smoke-22655188824/m9-smoke-22655188824.json`

Release-cut conclusion refresh:

- Both local and CI strict deployment smoke checks are green at the current production URLs.

## Drift guardrail refresh with CORS smoke checks (2026-03-04)

After adding explicit CORS checks to deployment smoke (`backend-cors-simple`, `backend-cors-preflight`), reran strict production smoke with the same live URLs:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://abp-backend-production.up.railway.app" \
  --backend-ws-base "wss://abp-backend-production.up.railway.app" \
  --frontend-base "https://frontend-nine-sandy-47.vercel.app" \
  --require-clean-wandb-status \
  --output-path artifacts/deploy/m9-smoke-strict-20260304-chunk3-run1.json
```

Result:

- `pass=true`
- `checks=15`
- `pass_count=15`
- `fail_count=0`
- New CORS checks both passed:
  - `backend-cors-simple`
  - `backend-cors-preflight`

Artifact:

- `artifacts/deploy/m9-smoke-strict-20260304-chunk3-run1.json`

Manual GitHub Actions strict smoke rerun after CORS guardrail patch:

- Workflow run: `https://github.com/jordanbailey00/asteroid-prospector-rl/actions/runs/22655777348`
- Result: `success`
- Downloaded artifact:
  - `artifacts/deploy/ci-m9-smoke-22655777348/m9-deployment-smoke-22655777348/m9-smoke-22655777348.json`
- Summary: `pass=true`, `checks=15`, `pass_count=15`, `fail_count=0`
