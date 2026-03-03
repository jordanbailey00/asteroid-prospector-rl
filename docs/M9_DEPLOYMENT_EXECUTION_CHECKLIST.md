# M9 Deployment Execution Checklist

Last updated: 2026-03-03

Purpose: capture the concrete values, release actions, and evidence needed to close M9.3 deployment alignment.

## 1) Fill release values

| Item | Value | Owner | Status |
| --- | --- | --- | --- |
| Backend HTTP base | `https://<backend-host>` |  |  |
| Backend WS base | `wss://<backend-host>` |  |  |
| Frontend base | `https://<vercel-domain>` |  |  |
| W&B entity | `<wandb-entity>` |  |  |
| W&B project | `<wandb-project>` |  |  |
| Artifact runs root | `<backend-runs-path>` |  |  |

## 2) Configure backend host

- Start from `server/.env.production.example`.
- Set real values for `ABP_RUNS_ROOT`, `ABP_CORS_ORIGINS`, `ABP_WANDB_ENTITY`, `ABP_WANDB_PROJECT`, and `ABP_WANDB_CACHE_TTL_SECONDS`.
- Inject `WANDB_API_KEY` through host secret manager (never committed).
- Restart backend process and confirm `GET /health` returns 200.

## 3) Configure Vercel project

- Start from `frontend/.env.production.example`.
- Set `NEXT_PUBLIC_BACKEND_HTTP_BASE` and `NEXT_PUBLIC_BACKEND_WS_BASE` for both preview and production.
- Redeploy Vercel project rooted at `frontend/`.

## 4) Run post-deploy smoke checks

Local smoke run:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://<backend-host>" \
  --backend-ws-base "wss://<backend-host>" \
  --frontend-base "https://<vercel-domain>" \
  --wandb-entity "<wandb-entity>" \
  --wandb-project "<wandb-project>" \
  --require-clean-wandb-status \
  --output-path artifacts/deploy/smoke-m9-<date>.json
```

Manual CI smoke run:

- Open `.github/workflows/m9-deployment-smoke.yml`.
- Set `backend_http_base`, `frontend_base`, `wandb_entity`, `wandb_project`.
- Set `require_clean_wandb_status=true`.

## 5) Record release evidence

- Save local smoke artifact under `artifacts/deploy/`.
- Attach GitHub Actions artifact URL.
- Capture backend `GET /api/wandb/status` before and after smoke calls.
- Link evidence in `docs/PROJECT_STATUS.md` commit ledger notes.

## 6) M9.3 done criteria

- [ ] Frontend is live on Vercel with working API + WS connectivity.
- [ ] Local smoke command exits 0 with `--require-clean-wandb-status`.
- [ ] Manual CI smoke workflow exits 0 with artifact uploaded.
- [ ] `/api/wandb/status` reports `available=true` with no unresolved release-blocking notes.
