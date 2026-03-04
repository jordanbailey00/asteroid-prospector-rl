from pathlib import Path
from typing import Any

from training.logging import WandbBenchmarkLogger, WandbWindowLogger


class _FakeArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []
        self.named_files: list[tuple[str, str | None]] = []
        self.metadata: dict[str, Any] = {}

    def add_file(self, path: str, name: str | None = None) -> None:
        self.files.append(path)
        self.named_files.append((path, name))


class _FakeRun:
    def __init__(self) -> None:
        self.url = "https://example.test/run/123"
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.finished = False

    def log(self, payload: dict[str, Any], *, step: int) -> None:
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


def test_wandb_benchmark_logger_logs_metrics_and_lineage_artifact(tmp_path: Path) -> None:
    run = _FakeRun()
    logger = WandbBenchmarkLogger(run=run, artifact_ctor=_FakeArtifact, job_type="eval")

    report_path = tmp_path / "m7-report.json"
    report_path.write_text("{}\n", encoding="utf-8")

    ckpt_a = tmp_path / "ckpt_a.pt"
    ckpt_a.write_text("checkpoint-a", encoding="utf-8")
    ckpt_b = tmp_path / "ckpt_b.pt"
    ckpt_b.write_text("checkpoint-b", encoding="utf-8")

    report = {
        "generated_at": "2026-03-04T00:00:00Z",
        "summary": {"pass": True, "seed_count": 2},
        "comparison": {"reference_policy": "ppo"},
        "artifacts": {"training_run_ids": ["run-a", "run-b"]},
    }

    logger.log_metrics({"benchmark/pass": 1.0}, step=0)
    artifact_info = logger.log_benchmark_report(
        report_path=report_path,
        run_id="m7-protocol-test",
        report=report,
        lineage_paths=[ckpt_a, ckpt_b],
    )
    logger.finish({"benchmark_pass": True})

    assert run.logged == [({"benchmark/pass": 1.0}, 0)]
    assert len(run.artifacts) == 1

    artifact, aliases = run.artifacts[0]
    assert artifact.name == "benchmark-m7-protocol-test"
    assert artifact.type == "benchmark"
    assert artifact.files[0] == str(report_path)
    assert str(ckpt_a) in artifact.files
    assert str(ckpt_b) in artifact.files
    assert aliases == ["latest", "m7-protocol-test", "job-eval"]

    assert artifact.metadata["reference_policy"] == "ppo"
    assert artifact.metadata["summary_pass"] is True
    assert artifact.metadata["seed_count"] == 2

    assert artifact_info["artifact_name"] == "benchmark-m7-protocol-test"
    assert artifact_info["artifact_aliases"] == ["latest", "m7-protocol-test", "job-eval"]
    assert artifact_info["lineage_file_count"] == 2

    assert run.summary["benchmark_pass"] is True
    assert run.finished is True
