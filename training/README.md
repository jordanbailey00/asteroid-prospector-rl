# training

M3/M4 training pipeline with checkpointing, optional W&B logging, and eval replay generation.

Current implementation:
- `training/train_puffer.py` (windowed run loop + metadata persistence; `random` and `puffer_ppo` backends)
- `training/puffer_backend.py` (PufferLib vectorized PPO training loop)
- `training/policy.py` (shared actor-critic architecture + checkpoint serialization helpers)
- `training/eval_runner.py` (per-window eval episodes + replay recording)
- `training/windowing.py` (window aggregation keyed by `window_env_steps`)
- `training/logging.py` (JSONL metrics sink + W&B adapter for checkpoints/replays)

## Local Windows usage

`puffer_ppo` requires Linux runtime (PufferLib is not supported natively on Windows in this project setup).
Use Docker compose:

```powershell
$env:DOCKER_BUILDKIT='1'
docker compose -f infra/docker-compose.yml build --progress=plain trainer
```

Smoke check:

```powershell
docker compose -f infra/docker-compose.yml run --rm trainer python -c "import pufferlib; print('ok')"
```

Run short PPO training in container:

```powershell
docker compose -f infra/docker-compose.yml run --rm -T trainer python training/train_puffer.py --trainer-backend puffer_ppo --total-env-steps 2000 --window-env-steps 500 --ppo-num-envs 8 --ppo-num-workers 4 --ppo-rollout-steps 128 --ppo-num-minibatches 4 --ppo-update-epochs 4 --wandb-mode disabled
```

## Random backend (host or container)

```powershell
python training/train_puffer.py --trainer-backend random --total-env-steps 6000 --window-env-steps 2000 --wandb-mode disabled
```

## Enable replay generation (M4)

Generate one eval replay per checkpointed window:

```powershell
python training/train_puffer.py --trainer-backend random --total-env-steps 6000 --window-env-steps 2000 --checkpoint-every-windows 1 --eval-replays-per-window 1 --eval-max-steps-per-episode 512 --no-eval-include-info --wandb-mode disabled
```

For PPO runs, eval replay generation is policy-driven from serialized checkpoint policy state.
Use `--eval-policy-deterministic` (default) for argmax actions or `--eval-policy-stochastic` for sampled actions.

Milestone tag thresholds are configurable with comma-separated lists:
- `--eval-milestone-profit-thresholds` (default `100,500,1000`)
- `--eval-milestone-return-thresholds` (default `10,25,50`)
- `--eval-milestone-survival-thresholds` (default `1.0`)

Replay tags:
- `every_window` on all generated replays
- `best_so_far` when replay return exceeds all previous run replays
- `milestone:*` tags when configured thresholds are met

## W&B offline mode

```powershell
python training/train_puffer.py --wandb-mode offline --wandb-project asteroid-prospector
```

## Artifacts

- `runs/{run_id}/checkpoints/ckpt_{window_id}.pt`
  - `random` backend checkpoints are JSON payloads (`checkpoint_format=json_v1`)
  - `puffer_ppo` checkpoints are torch payloads (`checkpoint_format=ppo_torch_v1`) and include serialized policy state
- `runs/{run_id}/metrics/windows.jsonl`
- `runs/{run_id}/replays/{replay_id}.jsonl.gz` (when eval enabled)
- `runs/{run_id}/replay_index.json` (when eval enabled)
- `runs/{run_id}/config.json`
- `runs/{run_id}/run_metadata.json`

`run_metadata.json` live-updated fields used by API/frontend milestones:
- `status` (`running`/`completed`/`failed`)
- `latest_window`
- `latest_checkpoint`
- `latest_replay` (present when eval replay generation is enabled)
- `replay_index_path` (present when eval replay generation is enabled)
- `wandb_run_url` and `constellation_url` (if available)
