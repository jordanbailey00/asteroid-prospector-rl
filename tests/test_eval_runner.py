import gzip
import json
from pathlib import Path

from replay.index import REPLAY_INDEX_SCHEMA_VERSION
from replay.schema import validate_replay_frame
from training.eval_runner import EvalReplayConfig, run_eval_and_record_replay


def _write_checkpoint(path: Path, *, run_id: str, window_id: int, env_steps_total: int) -> None:
    payload = {
        "run_id": run_id,
        "window_id": window_id,
        "env_steps_total": env_steps_total,
        "created_at": "2026-02-28T00:00:00+00:00",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_replay_frames(path: Path) -> list[dict]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_eval_runner_writes_replay_and_index(tmp_path: Path) -> None:
    run_id = "eval-run-a"
    run_dir = tmp_path / run_id
    ckpt = run_dir / "checkpoints" / "ckpt_000001.pt"
    _write_checkpoint(ckpt, run_id=run_id, window_id=1, env_steps_total=120)

    result = run_eval_and_record_replay(
        EvalReplayConfig(
            run_id=run_id,
            run_dir=run_dir,
            checkpoint_path=ckpt,
            window_id=1,
            trainer_backend="random",
            env_time_max=5000.0,
            base_seed=100,
            num_episodes=2,
            max_steps_per_episode=20,
            include_info=True,
        )
    )

    assert result.replay_path.exists()
    assert result.replay_path_relative.startswith("replays/")
    assert result.replay_index_path.exists()
    assert "every_window" in result.replay_entry["tags"]
    assert "best_so_far" in result.replay_entry["tags"]

    frames = _read_replay_frames(result.replay_path)
    assert frames
    assert "info" in frames[0]
    for frame in frames:
        validate_replay_frame(frame)

    index_payload = json.loads(result.replay_index_path.read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == REPLAY_INDEX_SCHEMA_VERSION
    assert index_payload["run_id"] == run_id
    assert len(index_payload["entries"]) == 1
    assert index_payload["entries"][0]["replay_id"] == result.replay_id


def test_eval_runner_skips_best_tag_when_prior_return_is_higher(tmp_path: Path) -> None:
    run_id = "eval-run-b"
    run_dir = tmp_path / run_id

    replay_index_path = run_dir / "replay_index.json"
    replay_index_path.parent.mkdir(parents=True, exist_ok=True)
    replay_index_path.write_text(
        json.dumps(
            {
                "schema_version": REPLAY_INDEX_SCHEMA_VERSION,
                "run_id": run_id,
                "updated_at": "2026-02-28T00:00:00+00:00",
                "entries": [
                    {
                        "run_id": run_id,
                        "window_id": 0,
                        "replay_id": "baseline",
                        "return_total": 1_000_000_000.0,
                        "tags": ["every_window", "best_so_far"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ckpt = run_dir / "checkpoints" / "ckpt_000002.pt"
    _write_checkpoint(ckpt, run_id=run_id, window_id=2, env_steps_total=240)

    result = run_eval_and_record_replay(
        EvalReplayConfig(
            run_id=run_id,
            run_dir=run_dir,
            checkpoint_path=ckpt,
            window_id=2,
            trainer_backend="random",
            env_time_max=5000.0,
            base_seed=200,
            num_episodes=1,
            max_steps_per_episode=15,
            include_info=False,
        )
    )

    assert "every_window" in result.replay_entry["tags"]
    assert "best_so_far" not in result.replay_entry["tags"]

    frames = _read_replay_frames(result.replay_path)
    assert frames
    assert "info" not in frames[0]

    index_payload = json.loads(replay_index_path.read_text(encoding="utf-8"))
    assert len(index_payload["entries"]) == 2
    assert index_payload["entries"][1]["replay_id"] == result.replay_id
