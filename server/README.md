# server

Initial M5 FastAPI API surface for run/replay catalog and replay fetch.

## Files

- `server/app.py` - app factory (`create_app`) and endpoint handlers.
- `server/main.py` - default ASGI entrypoint (`app`) using `ABP_RUNS_ROOT` or `runs/`.

## Run locally

```powershell
python -m uvicorn server.main:app --reload --port 8000
```

Optional custom runs root:

```powershell
$env:ABP_RUNS_ROOT='C:\path\to\runs'
python -m uvicorn server.main:app --reload --port 8000
```

## Implemented endpoints

- `GET /health`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/replays`
- `GET /api/runs/{run_id}/replays/{replay_id}`
- `GET /api/runs/{run_id}/replays/{replay_id}/frames`

Replay list filters (`/api/runs/{run_id}/replays`):
- `tag`
- `tags_any` (comma-separated)
- `tags_all` (comma-separated)
- `window_id`
- `limit`

Replay frame pagination (`/frames`):
- `offset`
- `limit`
