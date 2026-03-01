# server

M5 FastAPI API surface for run/replay catalog, metrics retrieval, replay frame fetch, and human play sessions.

## Files

- `server/app.py` - app factory (`create_app`) and endpoint handlers.
- `server/main.py` - default ASGI entrypoint (`app`) using env configuration.

## Run locally

```powershell
python -m uvicorn server.main:app --reload --port 8000
```

## Environment variables

- `ABP_RUNS_ROOT` (default `runs`)
- `ABP_CORS_ORIGINS` (comma-separated list; optional)
- `ABP_CORS_ORIGIN_REGEX` (optional; defaults to `https://.*\.vercel\.app`)

Examples:

```powershell
$env:ABP_RUNS_ROOT='C:\path\to\runs'
$env:ABP_CORS_ORIGINS='http://localhost:3000,http://127.0.0.1:3000'
python -m uvicorn server.main:app --reload --port 8000
```

## Implemented endpoints

- `GET /health`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/metrics/windows`
- `GET /api/runs/{run_id}/replays`
- `GET /api/runs/{run_id}/replays/{replay_id}`
- `GET /api/runs/{run_id}/replays/{replay_id}/frames`
- `WS /ws/runs/{run_id}/replays/{replay_id}/frames`
- `POST /api/play/session`
- `POST /api/play/session/{session_id}/reset`
- `POST /api/play/session/{session_id}/step`
- `DELETE /api/play/session/{session_id}`

Replay list filters (`/api/runs/{run_id}/replays`):
- `tag`
- `tags_any` (comma-separated)
- `tags_all` (comma-separated)
- `window_id`
- `limit`

Metrics query (`/api/runs/{run_id}/metrics/windows`):
- `limit`
- `order` (`asc` or `desc`)

Replay frame pagination (`/frames`):
- `offset`
- `limit`

Notes:
- Play sessions are process-local, ephemeral, and in-memory.
- Replay frame delivery supports both HTTP pagination and websocket chunked streaming (`offset`, `limit`, `batch_size`).
- Websocket replay stream tuning query params: `max_chunk_bytes` (default `262144`) and `yield_every_batches` (default `8`) for chunk size/backpressure tuning under large artifacts.
