# Asteroid Prospector RL

Asteroid Prospector RL is an end-to-end reinforcement learning project for a strategic asteroid-belt simulation with deterministic training/eval, replay generation, API delivery, and web playback.

## Current Status (2026-02-28)

- Completed milestones: `M0`, `M1`, `M2`, `M2.5`, `M3`, `M4`, `M5`, `M6`.
- Active milestone: `M6.5` (final graphics/audio verification + polish).
- Trainer runtime baseline: `pufferlib-core==3.0.17` (runtime `import pufferlib` reports `3.0.3`).
- Published reusable trainer image:
  - `jordanbailey00/rl-puffer-base:py311-puffercore3.0.17`
  - digest `sha256:723c58843d9ed563fa66c0927da975bdbab5355c913ec965dbea25a2af67bb71`

## What Is Implemented

- Frozen RL interface (`OBS_DIM=260`, `N_ACTIONS=69`) with deterministic Python reference env.
- Native C core and Python bridge with parity harness and deterministic RNG alignment.
- Dockerized PPO training loop with windowed metrics, checkpointing, and optional W&B logging.
- Policy-driven eval replay generation from serialized PPO checkpoints.
- Replay schema/indexing with `every_window`, `best_so_far`, and `milestone:*` tags.
- FastAPI backend for run catalog, metrics windows, replay fetch, and human play sessions.
- Next.js frontend for replay playback, human play mode, and historical analytics.
- Kenney asset-backed presentation layer (world/UI/VFX/audio manifests and runtime wiring).

## Repository Layout

- `engine_core/` native C simulation core
- `python/` reference env and wrappers
- `training/` PPO trainer + eval runner
- `replay/` replay schema/index helpers
- `server/` FastAPI API layer
- `frontend/` Next.js replay/play/analytics UI
- `infra/` Docker trainer runtime and compose config
- `tests/` contract, parity, replay, API, and frontend-presentation tests
- `docs/` specs, checklist, status, and ADR decision log

## Quick Start

1. Run local quality gates:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_checks.ps1
```

2. Build trainer image:

```powershell
$env:DOCKER_BUILDKIT='1'
docker compose -f infra/docker-compose.yml build --progress=plain trainer
```

3. Verify runtime versions:

```powershell
docker compose -f infra/docker-compose.yml run --rm trainer python -c "import importlib.metadata as im, pufferlib; print(im.version('pufferlib-core')); print(pufferlib.__version__)"
```

4. Run a short PPO smoke training job:

```powershell
docker compose -f infra/docker-compose.yml run --rm -T trainer python training/train_puffer.py --trainer-backend puffer_ppo --total-env-steps 300 --window-env-steps 100 --checkpoint-every-windows 1 --ppo-num-envs 4 --ppo-num-workers 2 --ppo-rollout-steps 32 --ppo-num-minibatches 2 --ppo-update-epochs 1 --wandb-mode disabled
```

## Documentation Entry Points

- `docs/DOCS_INDEX.md` (authoritative read order)
- `docs/PROJECT_STATUS.md` (current milestone board)
- `docs/DECISION_LOG.md` (ADR history)
- `docs/BUILD_CHECKLIST.md` (ordered implementation plan)

## License

MIT (`LICENSE`).
