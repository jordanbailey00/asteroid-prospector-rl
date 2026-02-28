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

    logger.finish({"env_steps_total": 300})

    assert logger.run_url == run.url
    assert run.logged == [({"window_id": 3, "reward_mean": 1.25}, 300)]
    assert len(run.artifacts) == 1

    artifact, aliases = run.artifacts[0]
    assert artifact.name == "model-run-abc"
    assert artifact.type == "model"
    assert artifact.files == [str(ckpt)]
    assert aliases == ["latest", "window-3"]
    assert run.summary["env_steps_total"] == 300
    assert run.finished is True
