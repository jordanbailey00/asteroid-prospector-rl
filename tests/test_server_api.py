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
            {
                "window_id": 0,
                "env_steps_total": 40,
                "reward_mean": 1.0,
                "return_mean": 2.0,
                "profit_mean": 20.0,
                "survival_rate": 0.9,
                "overheat_ticks_mean": 0.1,
                "pirate_encounters_mean": 0.2,
                "value_lost_to_pirates_mean": 1.0,
                "mining_ticks_mean": 3.0,
                "scan_count_mean": 1.0,
            },
            {
                "window_id": 1,
                "env_steps_total": 80,
                "reward_mean": 1.1,
                "return_mean": 2.2,
                "profit_mean": 25.0,
                "survival_rate": 0.95,
                "overheat_ticks_mean": 0.08,
                "pirate_encounters_mean": 0.15,
                "value_lost_to_pirates_mean": 0.8,
                "mining_ticks_mean": 3.2,
                "scan_count_mean": 1.2,
            },
            {
                "window_id": 2,
                "env_steps_total": 120,
                "reward_mean": 1.2,
                "return_mean": 2.4,
                "profit_mean": 30.0,
                "survival_rate": 1.0,
                "overheat_ticks_mean": 0.05,
                "pirate_encounters_mean": 0.1,
                "value_lost_to_pirates_mean": 0.6,
                "mining_ticks_mean": 3.5,
                "scan_count_mean": 1.3,
            },
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


def test_run_analytics_completeness_endpoint_reports_ok_coverage(tmp_path: Path) -> None:
    run_id = "run-complete"
    _make_run(tmp_path, run_id, updated_at="2026-03-03T01:00:00+00:00")

    app = create_app(
        runs_root=tmp_path,
        wandb_proxy=_FakeWandbProxy(),
        wandb_default_entity="team-astro",
        wandb_default_project="asteroid-prospector",
    )
    client = TestClient(app)

    response = client.get(
        f"/api/runs/{run_id}/analytics/completeness",
        params={
            "wandb_run_id": "wb-iter-002",
            "stale_after_seconds": 604800,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["overall_status"] == "ok"
    assert payload["status_counts"]["ok"] == 5

    coverage = {row["key"]: row for row in payload["coverage"]}
    assert coverage["run_metadata"]["status"] == "ok"
    assert coverage["window_metrics"]["status"] == "ok"
    assert coverage["replay_timeline"]["status"] == "ok"
    assert coverage["wandb_summary"]["status"] == "ok"
    assert coverage["wandb_history"]["status"] == "ok"


def test_run_analytics_completeness_endpoint_missing_and_stale_states(tmp_path: Path) -> None:
    run_id = "run-gappy"
    run_dir = tmp_path / run_id

    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "status": "completed",
            "trainer_backend": "random",
            "updated_at": "2000-01-01T00:00:00+00:00",
            "metrics_windows_path": "metrics/missing.jsonl",
            "replay_index_path": "replay_index.json",
            "started_at": "2000-01-01T00:00:00+00:00",
            "finished_at": "2000-01-01T00:01:00+00:00",
        },
    )
    _write_json(
        run_dir / "replay_index.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "updated_at": "2000-01-01T00:00:00+00:00",
            "entries": [],
        },
    )

    app = create_app(runs_root=tmp_path, wandb_proxy=_FakeWandbProxy())
    client = TestClient(app)

    response = client.get(
        f"/api/runs/{run_id}/analytics/completeness",
        params={"stale_after_seconds": 60},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["overall_status"] == "missing"

    coverage = {row["key"]: row for row in payload["coverage"]}
    assert coverage["run_metadata"]["status"] == "stale"
    assert coverage["window_metrics"]["status"] == "missing"
    assert coverage["replay_timeline"]["status"] == "missing"
    assert coverage["wandb_summary"]["status"] == "missing"
    assert coverage["wandb_history"]["status"] == "missing"


def test_run_analytics_completeness_endpoint_wandb_error_state(tmp_path: Path) -> None:
    run_id = "run-wandb-error"
    _make_run(tmp_path, run_id, updated_at="2026-03-03T02:00:00+00:00")

    run_dir = tmp_path / run_id
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["wandb_run_url"] = "https://wandb.ai/team-astro/asteroid-prospector/runs/wb-iter-002"
    _write_json(run_dir / "run_metadata.json", metadata)

    app = create_app(
        runs_root=tmp_path,
        wandb_proxy=_FailingWandbProxy(),
        wandb_default_entity="team-astro",
        wandb_default_project="asteroid-prospector",
    )
    client = TestClient(app)

    response = client.get(f"/api/runs/{run_id}/analytics/completeness")
    assert response.status_code == 200

    payload = response.json()
    assert payload["overall_status"] == "error"

    coverage = {row["key"]: row for row in payload["coverage"]}
    assert coverage["wandb_summary"]["status"] == "error"
    assert coverage["wandb_history"]["status"] == "error"


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


class _FakeWandbProxy:
    def __init__(self) -> None:
        self.calls = {
            "list_runs": 0,
            "get_run_summary": 0,
            "get_run_history": 0,
        }

    def list_runs(self, *, entity: str, project: str, limit: int) -> list[dict]:
        self.calls["list_runs"] += 1
        rows = [
            {
                "run_id": "wb-iter-002",
                "name": "iter-002",
                "state": "finished",
                "url": f"https://wandb.ai/{entity}/{project}/runs/wb-iter-002",
                "summary_preview": {
                    "_step": 200,
                    "return_mean": 17.5,
                    "profit_mean": 110.0,
                    "survival_rate": 1.0,
                },
            },
            {
                "run_id": "wb-iter-001",
                "name": "iter-001",
                "state": "finished",
                "url": f"https://wandb.ai/{entity}/{project}/runs/wb-iter-001",
                "summary_preview": {
                    "_step": 100,
                    "return_mean": 12.0,
                    "profit_mean": 70.0,
                    "survival_rate": 0.95,
                },
            },
        ]
        return rows[: int(limit)]

    def get_run_summary(self, *, entity: str, project: str, run_id: str) -> dict:
        self.calls["get_run_summary"] += 1
        return {
            "run_id": run_id,
            "name": f"name-{run_id}",
            "state": "finished",
            "url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
            "summary": {
                "window_id": 5,
                "env_steps_total": 1200,
                "reward_mean": 1.3,
                "return_mean": 18.0,
                "profit_mean": 115.0,
                "survival_rate": 1.0,
            },
            "config": {
                "seed": 7,
            },
        }

    def get_run_history(
        self,
        *,
        entity: str,
        project: str,
        run_id: str,
        keys: list[str] | None,
        max_points: int,
    ) -> list[dict]:
        self.calls["get_run_history"] += 1
        del entity
        del project
        del run_id

        rows = [
            {
                "_step": 100,
                "window_id": 4,
                "env_steps_total": 1000,
                "reward_mean": 1.1,
                "return_mean": 16.0,
                "profit_mean": 95.0,
                "survival_rate": 0.9,
            },
            {
                "_step": 200,
                "window_id": 5,
                "env_steps_total": 1200,
                "reward_mean": 1.3,
                "return_mean": 18.0,
                "profit_mean": 115.0,
                "survival_rate": 1.0,
            },
        ]

        if keys is None:
            selected = rows
        else:
            selected = []
            for row in rows:
                selected.append({key: row[key] for key in keys if key in row})
        return selected[: int(max_points)]


class _FailingWandbProxy:
    def list_runs(self, *, entity: str, project: str, limit: int) -> list[dict]:
        del entity
        del project
        del limit
        raise RuntimeError("W&B proxy unavailable in test")

    def get_run_summary(self, *, entity: str, project: str, run_id: str) -> dict:
        del entity
        del project
        del run_id
        raise RuntimeError("W&B proxy unavailable in test")

    def get_run_history(
        self,
        *,
        entity: str,
        project: str,
        run_id: str,
        keys: list[str] | None,
        max_points: int,
    ) -> list[dict]:
        del entity
        del project
        del run_id
        del keys
        del max_points
        raise RuntimeError("W&B proxy unavailable in test")


class _DiagnosticWandbProxy:
    def diagnostics(self) -> dict:
        return {
            "available": False,
            "reason": "simulated diagnostic outage",
            "cache": {
                "ttl_seconds": 0.0,
                "entries": 0,
                "hits": 1,
                "misses": 29,
                "expired": 0,
                "sets": 0,
            },
            "operations": {
                "list_runs": {
                    "calls": 10,
                    "errors": 2,
                    "latency_ms_total": 180.0,
                    "latency_ms_avg": 18.0,
                },
                "run_summary": {
                    "calls": 3,
                    "errors": 0,
                    "latency_ms_total": 48.0,
                    "latency_ms_avg": 16.0,
                },
            },
        }


class _LowHitRateWandbProxy:
    def diagnostics(self) -> dict:
        return {
            "available": True,
            "reason": None,
            "cache": {
                "ttl_seconds": 30.0,
                "entries": 2,
                "hits": 2,
                "misses": 38,
                "expired": 1,
                "sets": 40,
            },
            "operations": {
                "list_runs": {
                    "calls": 5,
                    "errors": 1,
                    "latency_ms_total": 100.0,
                    "latency_ms_avg": 20.0,
                },
                "run_summary": {
                    "calls": 4,
                    "errors": 0,
                    "latency_ms_total": 56.0,
                    "latency_ms_avg": 14.0,
                },
            },
        }


def test_wandb_proxy_endpoints(tmp_path: Path) -> None:
    wandb_proxy = _FakeWandbProxy()
    app = create_app(
        runs_root=tmp_path,
        wandb_proxy=wandb_proxy,
        wandb_default_entity="team-astro",
        wandb_default_project="asteroid-prospector",
    )
    client = TestClient(app)

    latest_resp = client.get("/api/wandb/runs/latest", params={"limit": 2})
    assert latest_resp.status_code == 200
    latest_payload = latest_resp.json()
    assert latest_payload["entity"] == "team-astro"
    assert latest_payload["project"] == "asteroid-prospector"
    assert latest_payload["count"] == 2
    assert latest_payload["runs"][0]["run_id"] == "wb-iter-002"

    summary_resp = client.get("/api/wandb/runs/wb-iter-002/summary")
    assert summary_resp.status_code == 200
    summary_payload = summary_resp.json()
    assert summary_payload["run"]["summary"]["profit_mean"] == 115.0

    history_resp = client.get(
        "/api/wandb/runs/wb-iter-002/history",
        params={"keys": "_step,return_mean,profit_mean", "max_points": 10},
    )
    assert history_resp.status_code == 200
    history_payload = history_resp.json()
    assert history_payload["count"] == 2
    assert history_payload["rows"][0]["_step"] == 100
    assert history_payload["rows"][1]["profit_mean"] == 115.0

    view_resp = client.get(
        "/api/wandb/runs/wb-iter-002/iteration-view",
        params={"keys": "_step,window_id,return_mean,profit_mean,survival_rate"},
    )
    assert view_resp.status_code == 200
    view_payload = view_resp.json()
    assert view_payload["history"]["count"] == 2
    assert view_payload["kpis"]["window_id"] == 5
    assert view_payload["kpis"]["return_mean"] == 18.0
    assert view_payload["kpis"]["profit_mean"] == 115.0

    assert wandb_proxy.calls["list_runs"] == 1
    assert wandb_proxy.calls["get_run_summary"] == 2
    assert wandb_proxy.calls["get_run_history"] == 2


def test_wandb_status_endpoint_with_diagnostics(tmp_path: Path) -> None:
    app = create_app(runs_root=tmp_path, wandb_proxy=_DiagnosticWandbProxy())
    client = TestClient(app)

    response = client.get("/api/wandb/status")
    assert response.status_code == 200

    payload = response.json()
    assert payload["available"] is False
    assert payload["reason"] == "simulated diagnostic outage"
    assert payload["defaults"]["entity"] is None
    assert payload["defaults"]["project"] is None
    assert payload["cache"]["ttl_seconds"] == 0.0
    assert payload["operations"]["list_runs"]["calls"] == 10
    assert payload["operations"]["list_runs"]["errors"] == 2

    notes = payload["notes"]
    assert any("ABP_WANDB_ENTITY" in note for note in notes)
    assert any("cache is disabled" in note for note in notes)
    assert any("hit ratio is low" in note for note in notes)
    assert any("reported errors" in note for note in notes)
    assert any("WANDB_API_KEY" in note for note in notes)


def test_wandb_status_endpoint_flags_low_hit_ratio_and_operation_errors(tmp_path: Path) -> None:
    app = create_app(
        runs_root=tmp_path,
        wandb_proxy=_LowHitRateWandbProxy(),
        wandb_default_entity="team-astro",
        wandb_default_project="asteroid-prospector",
    )
    client = TestClient(app)

    response = client.get("/api/wandb/status")
    assert response.status_code == 200

    payload = response.json()
    assert payload["available"] is True
    assert payload["reason"] is None
    assert payload["operations"]["list_runs"]["errors"] == 1

    notes = payload["notes"]
    assert any("hit ratio is low" in note for note in notes)
    assert any("reported errors" in note for note in notes)


def test_wandb_status_endpoint_without_diagnostics(tmp_path: Path) -> None:
    app = create_app(
        runs_root=tmp_path,
        wandb_proxy=_FakeWandbProxy(),
        wandb_default_entity="team-astro",
        wandb_default_project="asteroid-prospector",
    )
    client = TestClient(app)

    response = client.get("/api/wandb/status")
    assert response.status_code == 200

    payload = response.json()
    assert payload["available"] is True
    assert payload["reason"] is None
    assert payload["cache"] is None
    assert payload["operations"] is None
    assert payload["defaults"]["entity"] == "team-astro"
    assert payload["defaults"]["project"] == "asteroid-prospector"
    assert payload["notes"] == []


def test_wandb_proxy_requires_scope_when_not_configured(tmp_path: Path) -> None:
    app = create_app(runs_root=tmp_path, wandb_proxy=_FakeWandbProxy())
    client = TestClient(app)

    response = client.get("/api/wandb/runs/latest")
    assert response.status_code == 400
    assert "W&B entity/project not configured" in response.json()["detail"]


def test_wandb_proxy_surfaces_unavailable_errors(tmp_path: Path) -> None:
    app = create_app(
        runs_root=tmp_path,
        wandb_proxy=_FailingWandbProxy(),
        wandb_default_entity="team-astro",
        wandb_default_project="asteroid-prospector",
    )
    client = TestClient(app)

    response = client.get("/api/wandb/runs/latest")
    assert response.status_code == 503
    assert "W&B proxy unavailable" in response.json()["detail"]
