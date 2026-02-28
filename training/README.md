# training

M3 windowed training pipeline with checkpointing and optional W&B logging.

Current implementation:
- `training/train_puffer.py` (windowed run loop; `random` backend plus explicit `puffer_ppo` blocker path)
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

`puffer_ppo` backend note:
- `--trainer-backend puffer_ppo` is accepted for contract clarity, but it currently fails fast with a clear error.
- On Windows, the error explains that upstream PufferLib support is unavailable.

Artifacts:
- `runs/{run_id}/checkpoints/ckpt_{window_id}.pt`
- `runs/{run_id}/metrics/windows.jsonl`
- `runs/{run_id}/config.json`
- `runs/{run_id}/run_metadata.json`

`run_metadata.json` now includes live-updated fields used by upcoming API/frontend work:
- `status` (`running`/`completed`/`failed`)
- `latest_window`
- `latest_checkpoint`
- `latest_replay` (currently `null` until M4)
- `replay_index_path` (currently `null` until M4)
- `wandb_run_url` and `constellation_url` (if available)
