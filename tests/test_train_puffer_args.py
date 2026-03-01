import sys

from training.train_puffer import _parse_args


def test_parse_args_accepts_ppo_env_impl(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_puffer.py",
            "--trainer-backend",
            "random",
            "--ppo-env-impl",
            "native",
        ],
    )

    cfg = _parse_args()

    assert cfg.ppo_env_impl == "native"
