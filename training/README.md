# training

M3 windowed training pipeline with checkpointing, optional W&B logging, and backend-selectable training loops.

Current implementation:
- `training/train_puffer.py` (windowed run loop + metadata persistence; `random` and `puffer_ppo` backends)
- `training/puffer_backend.py` (PufferLib vectorized PPO training loop)
- `training/windowing.py` (window aggregation keyed by `window_env_steps`)
- `training/logging.py` (JSONL metrics sink + W&B adapter)

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

## W&B offline mode

```powershell
python training/train_puffer.py --wandb-mode offline --wandb-project asteroid-prospector
```

Artifacts:
- `runs/{run_id}/checkpoints/ckpt_{window_id}.pt`
- `runs/{run_id}/metrics/windows.jsonl`
- `runs/{run_id}/config.json`
- `runs/{run_id}/run_metadata.json`

`run_metadata.json` includes live-updated fields used by API/frontend milestones:
- `status` (`running`/`completed`/`failed`)
- `latest_window`
- `latest_checkpoint`
- `latest_replay` (currently `null` until M4)
- `replay_index_path` (currently `null` until M4)
- `wandb_run_url` and `constellation_url` (if available)
