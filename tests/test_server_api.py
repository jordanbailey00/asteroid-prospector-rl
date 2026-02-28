import gzip
import json
from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_replay(path: Path, frames: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wt", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame))
            handle.write("\n")


def _make_run(tmp_path: Path, run_id: str, *, updated_at: str) -> None:
    run_dir = tmp_path / run_id

    metadata = {
        "run_id": run_id,
        "status": "completed",
        "trainer_backend": "random",
        "env_steps_total": 120,
        "windows_emitted": 3,
        "checkpoints_written": 3,
        "latest_checkpoint": {"path": "checkpoints/ckpt_000002.pt"},
        "latest_replay": None,
        "replay_index_path": "replay_index.json",
        "started_at": "2026-02-28T00:00:00+00:00",
        "updated_at": updated_at,
        "finished_at": updated_at,
    }
    _write_json(run_dir / "run_metadata.json", metadata)

    replay_entry = {
        "run_id": run_id,
        "window_id": 2,
        "replay_id": f"{run_id}-replay",
        "replay_path": f"replays/{run_id}-replay.jsonl.gz",
        "checkpoint_path": "checkpoints/ckpt_000002.pt",
        "tags": ["every_window"],
        "return_total": 12.0,
        "profit": 50.0,
        "survival": 1.0,
        "steps": 4,
        "terminated": False,
        "truncated": False,
        "checkpoint_env_steps_total": 120,
        "created_at": updated_at,
    }
    replay_index = {
        "schema_version": 1,
        "run_id": run_id,
        "updated_at": updated_at,
        "entries": [replay_entry],
    }
    _write_json(run_dir / "replay_index.json", replay_index)

    _write_replay(
        run_dir / replay_entry["replay_path"],
        [
            {"frame_index": 0, "action": 1, "reward": 0.1},
            {"frame_index": 1, "action": 2, "reward": 0.2},
        ],
    )


def test_runs_catalog_and_run_detail(tmp_path: Path) -> None:
    _make_run(tmp_path, "run-old", updated_at="2026-02-28T00:00:00+00:00")
    _make_run(tmp_path, "run-new", updated_at="2026-02-28T01:00:00+00:00")

    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    runs_resp = client.get("/api/runs")
    assert runs_resp.status_code == 200
    runs_payload = runs_resp.json()

    assert runs_payload["count"] == 2
    assert [row["run_id"] for row in runs_payload["runs"]] == ["run-new", "run-old"]

    run_resp = client.get("/api/runs/run-new")
    assert run_resp.status_code == 200
    run_payload = run_resp.json()

    assert run_payload["run_id"] == "run-new"
    assert run_payload["metadata"]["trainer_backend"] == "random"
    assert run_payload["replay_count"] == 1


def test_replay_catalog_filters_and_frame_fetch(tmp_path: Path) -> None:
    run_id = "run-api"
    run_dir = tmp_path / run_id

    metadata = {
        "run_id": run_id,
        "status": "completed",
        "trainer_backend": "random",
        "replay_index_path": "replay_index.json",
        "updated_at": "2026-02-28T02:00:00+00:00",
    }
    _write_json(run_dir / "run_metadata.json", metadata)

    entry_a = {
        "run_id": run_id,
        "window_id": 0,
        "replay_id": "replay-a",
        "replay_path": "replays/replay-a.jsonl.gz",
        "checkpoint_path": "checkpoints/ckpt_000000.pt",
        "tags": ["every_window", "best_so_far"],
        "return_total": 5.0,
        "profit": 20.0,
        "survival": 1.0,
        "steps": 3,
        "created_at": "2026-02-28T01:00:00+00:00",
    }
    entry_b = {
        "run_id": run_id,
        "window_id": 1,
        "replay_id": "replay-b",
        "replay_path": "replays/replay-b.jsonl.gz",
        "checkpoint_path": "checkpoints/ckpt_000001.pt",
        "tags": ["every_window", "milestone:profit:100"],
        "return_total": 6.0,
        "profit": 120.0,
        "survival": 1.0,
        "steps": 4,
        "created_at": "2026-02-28T02:00:00+00:00",
    }
    replay_index = {
        "schema_version": 1,
        "run_id": run_id,
        "updated_at": "2026-02-28T02:00:00+00:00",
        "entries": [entry_a, entry_b],
    }
    _write_json(run_dir / "replay_index.json", replay_index)

    _write_replay(
        run_dir / "replays/replay-a.jsonl.gz",
        [
            {"frame_index": 0, "action": 1},
            {"frame_index": 1, "action": 2},
        ],
    )
    _write_replay(
        run_dir / "replays/replay-b.jsonl.gz",
        [
            {"frame_index": 0, "action": 3},
            {"frame_index": 1, "action": 4},
            {"frame_index": 2, "action": 5},
        ],
    )

    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    list_resp = client.get(f"/api/runs/{run_id}/replays", params={"tag": "every_window"})
    assert list_resp.status_code == 200
    assert list_resp.json()["count"] == 2

    filtered_resp = client.get(
        f"/api/runs/{run_id}/replays",
        params={"tags_any": "milestone:profit:100", "window_id": 1},
    )
    assert filtered_resp.status_code == 200
    filtered_payload = filtered_resp.json()
    assert filtered_payload["count"] == 1
    assert filtered_payload["replays"][0]["replay_id"] == "replay-b"

    replay_resp = client.get(f"/api/runs/{run_id}/replays/replay-b")
    assert replay_resp.status_code == 200
    assert replay_resp.json()["replay"]["window_id"] == 1

    frames_resp = client.get(
        f"/api/runs/{run_id}/replays/replay-b/frames",
        params={"offset": 1, "limit": 1},
    )
    assert frames_resp.status_code == 200
    frames_payload = frames_resp.json()
    assert frames_payload["count"] == 1
    assert frames_payload["has_more"] is True
    assert frames_payload["frames"][0]["action"] == 4
