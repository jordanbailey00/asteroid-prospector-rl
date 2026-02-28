# training

M3 windowed training pipeline with checkpointing and optional W&B logging.

Current implementation:
- `training/train_puffer.py` (windowed run loop; random-policy backend)
- `training/windowing.py` (window aggregation keyed by `window_env_steps`)
- `training/logging.py` (JSONL metrics sink + W&B adapter)

Run locally:

```powershell
python training/train_puffer.py --total-env-steps 6000 --window-env-steps 2000 --wandb-mode disabled
```

W&B offline mode:

```powershell
python training/train_puffer.py --wandb-mode offline --wandb-project asteroid-prospector
```

Artifacts:
- `runs/{run_id}/checkpoints/ckpt_{window_id}.pt`
- `runs/{run_id}/metrics/windows.jsonl`
- `runs/{run_id}/config.json`
- `runs/{run_id}/run_metadata.json`
