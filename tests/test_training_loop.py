import json
from pathlib import Path

import pytest

from training import TrainConfig, run_training


def test_training_runner_emits_windows_and_checkpoints(tmp_path: Path) -> None:
    cfg = TrainConfig(
        run_root=tmp_path,
        run_id="test-run",
        total_env_steps=120,
        window_env_steps=40,
        checkpoint_every_windows=1,
        seed=11,
        wandb_mode="disabled",
    )

    summary = run_training(cfg)

    assert summary["run_id"] == "test-run"
    assert summary["status"] == "completed"
    assert summary["windows_emitted"] >= 3
    assert summary["checkpoints_written"] >= 3

    run_dir = tmp_path / "test-run"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "run_metadata.json").exists()

    metrics_path = run_dir / "metrics" / "windows.jsonl"
    assert metrics_path.exists()

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 3
    assert rows[0]["window_id"] == 0
    assert rows[0]["env_steps_in_window"] == 40
    assert "profit_mean" in rows[0]
    assert "survival_rate" in rows[0]

    checkpoints = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    assert len(checkpoints) >= 3

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["target_env_steps"] == 120
    assert metadata["windows_emitted"] == len(rows)
    assert metadata["checkpoints_written"] == len(checkpoints)
    assert metadata["latest_replay"] is None
    assert metadata["replay_index_path"] is None
    assert metadata["metrics_windows_path"] == "metrics/windows.jsonl"
    assert metadata["latest_window"]["window_id"] == rows[-1]["window_id"]
    assert metadata["latest_window"]["metrics_row_path"] == "metrics/windows.jsonl"
    assert metadata["latest_checkpoint"]["window_id"] == rows[-1]["window_id"]
    assert metadata["latest_checkpoint"]["path"].startswith("checkpoints/")


def test_training_runner_puffer_backend_explicitly_blocked(tmp_path: Path) -> None:
    cfg = TrainConfig(
        run_root=tmp_path,
        run_id="blocked-run",
        total_env_steps=40,
        window_env_steps=20,
        checkpoint_every_windows=1,
        seed=3,
        wandb_mode="disabled",
        trainer_backend="puffer_ppo",
    )

    with pytest.raises((RuntimeError, NotImplementedError), match="puffer_ppo"):
        run_training(cfg)
