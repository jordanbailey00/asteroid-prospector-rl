# frontend

M6/M6.5 frontend implementation with M8 replay transport upgrades for the Asteroid Prospector MVP.

## Routes

- `/` replay player + window analytics + sector/minimap presentation (observer mode)
- `/play` human pilot mode (ephemeral play sessions) + sector/minimap presentation
- `/analytics` historical run/window analytics (read-only)

Public product boundary:

- Public routes are for observing replay, playing the game, and viewing analytics.
- Public routes must not expose training mutation controls.
- Training control/experiment tuning belongs in a separate private local dashboard.

## Backend contract

The UI calls the M5 API server endpoints:

- `/api/runs`, `/api/runs/{run_id}`
- `/api/runs/{run_id}/replays` + replay detail/frames
- `/api/runs/{run_id}/metrics/windows`
- `/api/wandb/runs/latest`
- `/api/wandb/runs/{wandb_run_id}/summary`
- `/api/wandb/runs/{wandb_run_id}/history`
- `/api/wandb/runs/{wandb_run_id}/iteration-view`
- `/api/play/session` lifecycle endpoints

Set backend URL via:

- `NEXT_PUBLIC_BACKEND_HTTP_BASE` (default `http://127.0.0.1:8000`)
- `NEXT_PUBLIC_BACKEND_WS_BASE` (optional; defaults from HTTP base as `ws://`/`wss://`)

## M6.5 presentation layer

Manifest-driven presentation resources now live at:

- `frontend/public/assets/manifests/graphics_manifest.json`
- `frontend/public/assets/manifests/audio_manifest.json`
- `frontend/lib/action_effects_manifest.json`
- `frontend/public/assets/backgrounds/starfield.svg`

Runtime modules:

- `frontend/lib/assets.ts` manifest loading and typed contracts
- `frontend/lib/presentation.ts` action/event VFX + cue mapping helpers
- `frontend/lib/audio.ts` cue playback (file-backed when available, synth fallback)
- `frontend/components/sector-view.tsx` sector canvas + mini-map renderer

## Local development

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

Build + lint:

```powershell
npm --prefix frontend run lint
npm --prefix frontend run build
```

## Validation

Manifest mapping tests are covered by:

- `tests/test_frontend_presentation.py`

These checks ensure:

- action IDs `0..68` are fully mapped
- mapped VFX keys resolve in `graphics_manifest.json`
- mapped cue keys resolve in `audio_manifest.json`
- referenced background files exist

## Notes

- Replay playback supports selectable transport: HTTP frame pagination (`/frames`) or websocket chunked stream (`/ws/.../frames`).
- Audio cues are file-backed from `/assets/audio/...`; synth fallback is only used when browser playback is blocked or a cue intentionally has no files.
- Vercel deployment should point `NEXT_PUBLIC_BACKEND_HTTP_BASE` to your hosted FastAPI origin.
- Current UX realignment plan and gap matrix: `docs/PUBLIC_UX_REALIGNMENT_PLAN_20260303.md`.
- RL training remains non-pixel simulation; pixel rendering is a presentation layer for replay and human play.
## Deployment smoke check

After Vercel deploy, validate end-to-end routing and replay websocket transport:

```powershell
python tools/smoke_m9_deployment.py \
  --backend-http-base "https://<backend-host>" \
  --frontend-base "https://<vercel-domain>" \
  --wandb-entity "<wandb-entity>" \
  --wandb-project "<wandb-project>"
```

For the full release checklist, use `docs/M9_DEPLOYMENT_RUNBOOK.md`. Deployment smoke now verifies W&B latest + summary + history + iteration-view proxy paths plus a post-operation W&B status gate.

Manual CI run: `.github/workflows/m9-deployment-smoke.yml` (workflow_dispatch).
