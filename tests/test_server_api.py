import gzip
import json
from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


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
        "metrics_windows_path": "metrics/windows.jsonl",
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

    _write_jsonl(
        run_dir / "metrics/windows.jsonl",
        [
            {"window_id": 0, "env_steps_total": 40, "reward_mean": 1.0},
            {"window_id": 1, "env_steps_total": 80, "reward_mean": 1.1},
            {"window_id": 2, "env_steps_total": 120, "reward_mean": 1.2},
        ],
    )

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


def test_run_metrics_windows_endpoint(tmp_path: Path) -> None:
    run_id = "run-metrics"
    _make_run(tmp_path, run_id, updated_at="2026-02-28T03:00:00+00:00")

    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    latest_resp = client.get(f"/api/runs/{run_id}/metrics/windows", params={"limit": 1})
    assert latest_resp.status_code == 200
    latest_payload = latest_resp.json()
    assert latest_payload["count"] == 1
    assert latest_payload["windows"][0]["window_id"] == 2

    asc_resp = client.get(
        f"/api/runs/{run_id}/metrics/windows",
        params={"limit": 2, "order": "asc"},
    )
    assert asc_resp.status_code == 200
    asc_payload = asc_resp.json()
    assert [row["window_id"] for row in asc_payload["windows"]] == [0, 1]


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


def test_replay_frame_websocket_stream(tmp_path: Path) -> None:
    run_id = "run-ws"
    _make_run(tmp_path, run_id, updated_at="2026-02-28T05:00:00+00:00")

    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    with client.websocket_connect(
        f"/ws/runs/{run_id}/replays/{run_id}-replay/frames?offset=0&limit=2&batch_size=1"
    ) as ws:
        first = ws.receive_json()
        second = ws.receive_json()
        complete = ws.receive_json()

    assert first["type"] == "frames"
    assert first["count"] == 1
    assert first["frames"][0]["action"] == 1

    assert second["type"] == "frames"
    assert second["count"] == 1
    assert second["frames"][0]["action"] == 2

    assert complete["type"] == "complete"
    assert complete["count"] == 2
    assert complete["has_more"] is False
    assert complete["chunks_sent"] == 2


def test_replay_frame_websocket_max_chunk_bytes_splits_stream(tmp_path: Path) -> None:
    run_id = "run-ws-chunk-bytes"
    _make_run(tmp_path, run_id, updated_at="2026-02-28T05:30:00+00:00")

    _write_replay(
        tmp_path / run_id / "replays" / f"{run_id}-replay.jsonl.gz",
        [
            {"frame_index": 0, "action": 1, "blob": "x" * 1500},
            {"frame_index": 1, "action": 2, "blob": "y" * 1500},
        ],
    )

    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    with client.websocket_connect(
        f"/ws/runs/{run_id}/replays/{run_id}-replay/frames"
        "?offset=0&limit=2&batch_size=10&max_chunk_bytes=1024"
    ) as ws:
        first = ws.receive_json()
        second = ws.receive_json()
        complete = ws.receive_json()

    assert first["type"] == "frames"
    assert first["count"] == 1
    assert first["chunk_bytes"] > 1024

    assert second["type"] == "frames"
    assert second["count"] == 1
    assert second["chunk_bytes"] > 1024

    assert complete["type"] == "complete"
    assert complete["count"] == 2
    assert complete["chunks_sent"] == 2
    assert complete["max_chunk_bytes"] == 1024


def test_replay_frame_websocket_invalid_chunk_bytes_param(tmp_path: Path) -> None:
    run_id = "run-ws-invalid-param"
    _make_run(tmp_path, run_id, updated_at="2026-02-28T05:45:00+00:00")

    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    with client.websocket_connect(
        f"/ws/runs/{run_id}/replays/{run_id}-replay/frames?max_chunk_bytes=1"
    ) as ws:
        payload = ws.receive_json()

    assert payload["type"] == "error"
    assert payload["status_code"] == 400
    assert "max_chunk_bytes" in payload["detail"]


def test_replay_frame_websocket_unknown_replay(tmp_path: Path) -> None:
    run_id = "run-ws-missing"
    _make_run(tmp_path, run_id, updated_at="2026-02-28T06:00:00+00:00")

    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    with client.websocket_connect(
        f"/ws/runs/{run_id}/replays/does-not-exist/frames?offset=0&limit=10"
    ) as ws:
        payload = ws.receive_json()

    assert payload["type"] == "error"
    assert payload["status_code"] == 404
    assert "replay_id not found" in payload["detail"]


def test_play_session_lifecycle(tmp_path: Path) -> None:
    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    create_resp = client.post(
        "/api/play/session",
        json={"seed": 123, "env_time_max": 2000.0},
    )
    assert create_resp.status_code == 200
    created = create_resp.json()
    session_id = created["session_id"]

    assert created["seed"] == 123
    assert isinstance(created["obs"], list)
    assert len(created["obs"]) == 260
    assert created["steps"] == 0

    step_resp = client.post(
        f"/api/play/session/{session_id}/step",
        json={"action": 0},
    )
    assert step_resp.status_code == 200
    stepped = step_resp.json()
    assert stepped["session_id"] == session_id
    assert stepped["action"] == 0
    assert isinstance(stepped["reward"], float)
    assert len(stepped["obs"]) == 260
    assert stepped["steps"] == 1

    reset_resp = client.post(
        f"/api/play/session/{session_id}/reset",
        json={"seed": 124},
    )
    assert reset_resp.status_code == 200
    reset_payload = reset_resp.json()
    assert reset_payload["seed"] == 124
    assert reset_payload["steps"] == 0

    delete_resp = client.delete(f"/api/play/session/{session_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    missing_resp = client.post(
        f"/api/play/session/{session_id}/step",
        json={"action": 0},
    )
    assert missing_resp.status_code == 404


def test_cors_preflight_allows_localhost_origin(tmp_path: Path) -> None:
    app = create_app(
        runs_root=tmp_path,
        cors_allow_origins=["http://localhost:3000"],
        cors_allow_origin_regex=None,
    )
    client = TestClient(app)

    response = client.options(
        "/api/runs",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code in {200, 204}
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
