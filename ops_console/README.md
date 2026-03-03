# ops_console

Local-only training operations dashboard for M9.5.

Purpose:
- launch and stop training jobs,
- adjust run specs/hyperparameters before launch,
- monitor live logs and active run telemetry,
- inspect recent run artifacts (checkpoints/replays/windows).

This dashboard is intentionally separate from the public frontend (`/`, `/play`, `/analytics`) and should not be deployed to Vercel/public routing.

## Run locally

```powershell
python -m ops_console.main
```

Default address: `http://127.0.0.1:8090`

## Environment variables

- `ABP_OPS_REPO_ROOT` (default repo root inferred from file path)
- `ABP_OPS_RUNS_ROOT` (default `<repo_root>/runs`)
- `ABP_OPS_ENFORCE_LOCAL_ONLY` (default `true`)
- `ABP_OPS_HOST` (default `127.0.0.1`)
- `ABP_OPS_PORT` (default `8090`)

## Runtime modes

The dashboard supports two launch runtimes:

- `host_python`
  - Launches `training/train_puffer.py` directly via local Python.
  - Good for `random` backend and host-side utilities.

- `docker_trainer`
  - Launches training through:
  - `docker compose -f infra/docker-compose.yml run --rm -T trainer ...`
  - Recommended for `puffer_ppo` on Windows hosts.

## API surface

- `GET /health`
- `GET /api/profiles`
- `GET /api/job`
- `POST /api/job/start`
- `POST /api/job/stop`
- `GET /api/job/logs?tail=...`
- `GET /api/runs?limit=...`
