# M9 Deployment Runbook (Vercel + External Backend)

Last updated: 2026-03-02

This runbook aligns with M9.3 requirements:

- frontend hosted on Vercel,
- backend hosted on websocket-capable infrastructure,
- secrets server-side only,
- post-deploy smoke checks for replay WS and analytics routes.

## 1) Production topology

- Frontend: Vercel (`frontend/`)
- Backend: separate FastAPI host with HTTPS + WSS
- Data/Artifacts: backend host filesystem or mounted volume for `runs/`

Do not host backend websocket routes on Vercel serverless functions.

## 2) Backend environment

Required backend environment variables:

- `ABP_RUNS_ROOT` (path to training/eval run artifacts)
- `ABP_CORS_ORIGINS` (comma-separated frontend origins)
- `ABP_CORS_ORIGIN_REGEX` (optional; can include Vercel preview wildcard)
- `ABP_WANDB_ENTITY` (default W&B entity for proxy routes)
- `ABP_WANDB_PROJECT` (default W&B project for proxy routes)
- `WANDB_API_KEY` (server-side only; never expose to frontend)
- `ABP_WANDB_CACHE_TTL_SECONDS` (optional; default `30`)

Backend startup example:

```powershell
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## 3) Frontend environment (Vercel)

Set these in Vercel project environment variables:

- `NEXT_PUBLIC_BACKEND_HTTP_BASE` = `https://<backend-host>`
- `NEXT_PUBLIC_BACKEND_WS_BASE` = `wss://<backend-host>`

Vercel build root should target `frontend/`.

## 4) Post-deploy smoke checks

Use the deployment smoke tool:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://<backend-host>" \
  --backend-ws-base "wss://<backend-host>" \
  --frontend-base "https://<vercel-app-domain>" \
  --wandb-entity "<wandb-entity>" \
  --wandb-project "<wandb-project>" \
  --output-path artifacts/deploy/smoke-m9.json
```

If you already know target replay identifiers:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://<backend-host>" \
  --run-id "<run_id>" \
  --replay-id "<replay_id>"
```

Notes:

- Script validates:
  - backend `/health`, runs catalog, replay HTTP frames,
  - replay websocket stream route,
  - frontend routes (`/`, `/play`, `/analytics`),
  - W&B proxy latest-runs endpoint.
- Exit code is non-zero when any check fails.

## 5) Release gate

Consider deployment successful only when:

1. `tools/smoke_m9_deployment.py` exits `0`.
2. Frontend `/analytics` loads W&B-backed data without client-side secrets.
3. Replay websocket route is reachable from frontend origin.
