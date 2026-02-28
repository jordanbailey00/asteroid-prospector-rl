# frontend

M6 + M6.5 frontend implementation for the Asteroid Prospector MVP.

## Routes

- `/` replay player + window analytics + sector/minimap presentation
- `/play` human pilot mode (ephemeral play sessions) + sector/minimap presentation
- `/analytics` historical run/window analytics

## Backend contract

The UI calls the M5 API server endpoints:

- `/api/runs`, `/api/runs/{run_id}`
- `/api/runs/{run_id}/replays` + replay detail/frames
- `/api/runs/{run_id}/metrics/windows`
- `/api/play/session` lifecycle endpoints

Set backend URL via:

- `NEXT_PUBLIC_BACKEND_HTTP_BASE` (default `http://127.0.0.1:8000`)

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

- Replay playback still uses HTTP frame download (`/frames`) and client-side timer controls.
- Audio cues currently default to procedural synth fallback; real OGG assets can be added later without changing UI code, as long as manifests are updated.
- Vercel deployment should point `NEXT_PUBLIC_BACKEND_HTTP_BASE` to your hosted FastAPI origin.
