# Asteroid Prospector RL

Asteroid Prospector RL is an end-to-end reinforcement learning project for a deterministic asteroid-belt simulation with:
- a native C simulation core,
- PPO training and eval replay generation,
- FastAPI replay/play APIs (HTTP + websocket replay transport),
- and a Next.js frontend for replay, play, and analytics.

## Current Status (2026-03-01)

- Completed milestones: `M0`, `M1`, `M2`, `M2.5`, `M3`, `M4`, `M5`, `M6`, `M6.5`, `M8`.
- Active milestone: `M9` (throughput program + W&B analytics integration + Vercel alignment).
- Remaining milestone: `M7` baseline bots + benchmark automation.

Trainer/runtime baseline:
- `pufferlib-core==3.0.17`
- `gymnasium==1.2.3`
- `torch==2.10.0`
- `wandb==0.25.0`

Published trainer base image:
- `jordanbailey00/rl-puffer-base:py311-puffercore3.0.17`
- digest `sha256:723c58843d9ed563fa66c0927da975bdbab5355c913ec965dbea25a2af67bb71`

## What Is Implemented

- Frozen RL interface (`OBS_DIM=260`, `N_ACTIONS=69`) with deterministic Python reference env.
- Native C core plus Python bridge, parity harness, and batched native bridge APIs (`step_many/reset_many`).
- Windowed training loop with checkpointing, run metadata, and optional W&B logging.
- Policy-driven eval replays with replay schema/index and replay tags (`every_window`, `best_so_far`, `milestone:*`).
- API server endpoints for runs, metrics, replays, replay frames (HTTP + WS), and ephemeral human play sessions.
- Frontend pages for replay (`/`), play (`/play`), and analytics (`/analytics`) using backend APIs.
- Kenney asset-backed presentation and audio manifests with validation tests.
- Throughput profiling, matrix calibration, and floor-gate tooling under `tools/` and `artifacts/throughput/`.

## Immediate Next Work

1. Add backend W&B proxy endpoints for run summaries/history/iteration views.
2. Extend analytics UI for current iteration, full history, and last-10 iteration drilldown.
3. Complete deployment path (Vercel frontend + websocket-capable backend) with production env/CORS checks.
4. Implement baseline bots and automate PPO-vs-baseline benchmark reporting.

## Quick Start

1. Run local checks:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_checks.ps1
```

2. Build trainer image:

```powershell
$env:DOCKER_BUILDKIT='1'
docker compose -f infra/docker-compose.yml build --progress=plain trainer
```

3. Run a short PPO smoke job:

```powershell
docker compose -f infra/docker-compose.yml run --rm -T trainer python training/train_puffer.py --trainer-backend puffer_ppo --total-env-steps 300 --window-env-steps 100 --checkpoint-every-windows 1 --ppo-num-envs 4 --ppo-num-workers 2 --ppo-rollout-steps 32 --ppo-num-minibatches 2 --ppo-update-epochs 1 --wandb-mode disabled
```

## Repository Layout

- `engine_core/` native C simulation core
- `python/` wrappers and reference env
- `training/` trainer, eval runner, metrics/logging
- `replay/` replay schema and index helpers
- `server/` FastAPI API layer
- `frontend/` Next.js web app
- `infra/` Docker trainer runtime
- `tools/` profiling, parity, checklist, and benchmark tooling
- `tests/` contract/parity/runtime/API/frontend tests
- `docs/` specs, checklist, status, and ADR log

## Documentation Entry Points

- `docs/DOCS_INDEX.md`
- `docs/BUILD_CHECKLIST.md`
- `docs/PROJECT_STATUS.md`
- `docs/DECISION_LOG.md`

## License

MIT (`LICENSE`).
