# frontend

M6 frontend integration for the Asteroid Prospector MVP.

## Routes

- `/` replay player + window analytics
- `/play` human pilot mode (ephemeral play sessions)
- `/analytics` historical run/window analytics

## Backend contract

The UI calls the M5 API server endpoints:

- `/api/runs`, `/api/runs/{run_id}`
- `/api/runs/{run_id}/replays` + replay detail/frames
- `/api/runs/{run_id}/metrics/windows`
- `/api/play/session` lifecycle endpoints

Set the backend base URL via:

- `NEXT_PUBLIC_BACKEND_HTTP_BASE` (default `http://127.0.0.1:8000`)

## Local development

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

Build check:

```powershell
npm --prefix frontend run build
```

## Notes

- Replay playback currently uses HTTP frame download (`/frames`) and client-side timer controls.
- Vercel deployment should point `NEXT_PUBLIC_BACKEND_HTTP_BASE` to your hosted FastAPI origin.
