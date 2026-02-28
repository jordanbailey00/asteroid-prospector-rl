import json
from pathlib import Path

import pytest

from training.eval_runner import _load_checkpoint_payload
from training.train_puffer import write_checkpoint


def test_write_checkpoint_json_format_for_random_backend(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints" / "ckpt_000001.pt"
    write_checkpoint(
        path=ckpt,
        run_id="run-a",
        window_id=1,
        env_steps_total=100,
        trainer_backend="random",
    )

    payload = json.loads(ckpt.read_text(encoding="utf-8"))
    assert payload["checkpoint_format"] == "json_v1"
    assert payload["trainer_backend"] == "random"
    assert payload["env_steps_total"] == 100


def test_write_checkpoint_torch_format_for_puffer_backend(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    ckpt = tmp_path / "checkpoints" / "ckpt_000002.pt"
    write_checkpoint(
        path=ckpt,
        run_id="run-b",
        window_id=2,
        env_steps_total=200,
        trainer_backend="puffer_ppo",
        extra_payload={
            "policy_arch": "mlp-256x256-tanh-v1",
            "obs_shape": [260],
            "n_actions": 69,
            "model_state_dict": {"dummy": torch.zeros(1)},
        },
    )

    payload = _load_checkpoint_payload(ckpt)
    assert payload["checkpoint_format"] == "ppo_torch_v1"
    assert payload["trainer_backend"] == "puffer_ppo"
    assert payload["window_id"] == 2
    assert "model_state_dict" in payload
