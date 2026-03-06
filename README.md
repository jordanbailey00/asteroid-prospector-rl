<p align="center">
  <img src="asteroidmining.jpg" alt="Asteroid Prospector RL header art" width="960" />
</p>

<p align="center">
  <a href="https://github.com/jordanbailey00/asteroid-prospector-rl/actions/workflows/ci.yml">
    <img src="https://github.com/jordanbailey00/asteroid-prospector-rl/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
  <a href="https://github.com/jordanbailey00/asteroid-prospector-rl/actions/workflows/m7-nightly-regression.yml">
    <img src="https://github.com/jordanbailey00/asteroid-prospector-rl/actions/workflows/m7-nightly-regression.yml/badge.svg" alt="Nightly Regression" />
  </a>
  <img src="https://img.shields.io/badge/status-MVP%20Complete-success" alt="Status: MVP Complete" />
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white" alt="Backend: FastAPI" />
  <img src="https://img.shields.io/badge/frontend-Next.js-000000?logo=nextdotjs&logoColor=white" alt="Frontend: Next.js" />
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" />
</p>

<h1 align="center">Asteroid Prospector RL</h1>

Asteroid Prospector RL is an end-to-end reinforcement learning system for a deterministic asteroid-belt simulation. It includes a native simulation core, training pipelines, replay generation, API services, and a web frontend for replay/play/analytics.

## What This Project Is

- A deterministic resource-trading and navigation environment with a frozen RL interface (`OBS_DIM=260`, `N_ACTIONS=69`).
- A full training and evaluation pipeline centered on PPO, plus tooling for profiling, parity, and validation.
- A deployable product surface with FastAPI backend services and a Next.js frontend for replay, human play, and analytics.

## High-Level Goals

1. Train policies that can survive, mine, trade, and optimize profit in a dynamic asteroid economy.
2. Keep the simulation deterministic and reproducible for debugging and benchmark fairness.
3. Maintain parity between Python reference and native core behavior under fixed seeds and action traces.
4. Provide production-oriented tooling for evaluation, deployment smoke tests, and release operations.
5. Expose training outcomes through usable replay and analytics interfaces.

## How It Was Built

- `engine_core/`: native C simulation runtime with batched stepping APIs.
- `python/`: reference environment, wrappers, and frozen interface contracts.
- `training/`: PPO training loop, checkpoint/window lifecycle, eval replay generation, and W&B-capable logging.
- `replay/`: replay schema and indexing/query helpers.
- `server/`: FastAPI APIs for runs, metrics, replays, play sessions, and websocket replay transport.
- `frontend/`: Next.js app with replay (`/`), play (`/play`), and analytics (`/analytics`) pages.
- `tools/`: parity runners, throughput profilers, smoke checks, and benchmark automation.
- `tests/`: contract, parity, runtime, API, tooling, and frontend regression coverage.

## Setup

### Prerequisites

- Python `3.11+`
- Node.js `18+` and npm
- Docker Desktop (for reproducible Linux trainer runtime)
- PowerShell (commands below use PowerShell syntax)

### 1. Run local checks

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_checks.ps1
```

### 2. Build the trainer image

```powershell
$env:DOCKER_BUILDKIT='1'
docker compose -f infra/docker-compose.yml build --progress=plain trainer
```

### 3. Run a short PPO smoke training job

```powershell
docker compose -f infra/docker-compose.yml run --rm -T trainer python training/train_puffer.py --trainer-backend puffer_ppo --total-env-steps 300 --window-env-steps 100 --checkpoint-every-windows 1 --ppo-num-envs 4 --ppo-num-workers 2 --ppo-rollout-steps 32 --ppo-num-minibatches 2 --ppo-update-epochs 1 --wandb-mode disabled
```

### 4. Run backend and frontend locally (optional product surface)

```powershell
python -m uvicorn server.main:app --reload --port 8000
npm --prefix frontend install
npm --prefix frontend run dev
```

## Repository Layout

- `engine_core/` native C simulation core
- `python/` wrappers and reference environment
- `training/` trainer, eval runner, and metrics/logging
- `replay/` replay schema and index helpers
- `server/` FastAPI API layer
- `frontend/` Next.js web app
- `infra/` Docker trainer runtime
- `tools/` profiling, parity, smoke, and benchmark tooling
- `tests/` contract/parity/runtime/API/frontend tests
- `docs/` specs, checklist, status, and ADR log

## Documentation Entry Points

- `docs/DOCS_INDEX.md`
- `docs/PROJECT_STATUS.md`
- `docs/DECISION_LOG.md`
- `docs/BUILD_CHECKLIST.md`

## License

MIT (`LICENSE`).
