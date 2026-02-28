from pathlib import Path

from training.logging import WandbWindowLogger


class _FakeArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class _FakeRun:
    def __init__(self) -> None:
        self.url = "https://example.test/run/123"
        self.summary: dict[str, float | int] = {}
        self.logged: list[tuple[dict[str, float | int], int]] = []
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.finished = False

    def log(self, payload: dict[str, float | int], *, step: int) -> None:
        self.logged.append((payload, step))

    def log_artifact(self, artifact: _FakeArtifact, *, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self) -> None:
        self.finished = True


def test_wandb_logger_logs_windows_and_checkpoint_artifacts(tmp_path: Path) -> None:
    run = _FakeRun()
    logger = WandbWindowLogger(run=run, artifact_ctor=_FakeArtifact)

    logger.log_window({"window_id": 3, "reward_mean": 1.25}, step=300)

    ckpt = tmp_path / "ckpt_000003.pt"
    ckpt.write_text("checkpoint", encoding="utf-8")
    logger.log_checkpoint(checkpoint_path=ckpt, run_id="run-abc", window_id=3)

    replay = tmp_path / "replay-000003-xyz.jsonl.gz"
    replay.write_text("{}\n", encoding="utf-8")
    logger.log_replay(
        replay_path=replay,
        run_id="run-abc",
        window_id=3,
        replay_id="replay-000003-xyz",
        tags=["every_window", "best_so_far"],
    )

    logger.finish({"env_steps_total": 300})

    assert logger.run_url == run.url
    assert run.logged == [({"window_id": 3, "reward_mean": 1.25}, 300)]
    assert len(run.artifacts) == 2

    ckpt_artifact, ckpt_aliases = run.artifacts[0]
    assert ckpt_artifact.name == "model-run-abc"
    assert ckpt_artifact.type == "model"
    assert ckpt_artifact.files == [str(ckpt)]
    assert ckpt_aliases == ["latest", "window-3"]

    replay_artifact, replay_aliases = run.artifacts[1]
    assert replay_artifact.name == "replay-run-abc-000003-replay-000003-xyz"
    assert replay_artifact.type == "replay"
    assert replay_artifact.files == [str(replay)]
    assert replay_aliases == ["latest", "window-3", "every_window", "best_so_far"]

    assert run.summary["env_steps_total"] == 300
    assert run.finished is True
