from pathlib import Path

from fastapi.testclient import TestClient

from ops_console.app import build_training_launch_plan, create_ops_console_app


class FakeManager:
    def __init__(self) -> None:
        self.running = False
        self.last_payload = None
        self.logs = ["boot", "step 1", "step 2"]

    def profiles_payload(self):
        return {
            "default_profile": "random_eval",
            "runtimes": ["host_python", "docker_trainer"],
            "supported_overrides": ["total_env_steps", "window_env_steps"],
            "profiles": [
                {
                    "name": "random_eval",
                    "description": "test profile",
                    "runtime": "host_python",
                    "defaults": {"total_env_steps": 1000},
                }
            ],
        }

    def status(self):
        return {
            "running": self.running,
            "has_job": self.last_payload is not None,
            "run_id": None if self.last_payload is None else self.last_payload.get("run_id"),
        }

    def start_job(self, **kwargs):
        if self.running:
            raise RuntimeError("already running")
        self.running = True
        self.last_payload = kwargs
        return {
            "running": True,
            "has_job": True,
            "run_id": kwargs.get("run_id") or "generated",
        }

    def stop_job(self, *, force: bool):
        self.running = False
        return {
            "running": False,
            "has_job": self.last_payload is not None,
            "force": force,
        }

    def tail_logs(self, *, tail_lines: int):
        lines = self.logs[-tail_lines:] if tail_lines > 0 else []
        return {
            "count": len(lines),
            "lines": lines,
            "log_path": "artifacts/ops_console/logs/fake.log",
        }

    def list_runs(self, *, limit: int):
        return {
            "count": 1,
            "runs": [
                {
                    "run_id": "fake-run",
                    "status": "running",
                    "env_steps_total": 123,
                    "windows_emitted": 1,
                    "checkpoint_count": 1,
                    "replay_count": 0,
                    "steps_per_second_estimate": 12.5,
                    "updated_at": "2026-03-03T00:00:00+00:00",
                }
            ][:limit],
        }


def test_build_training_launch_plan_applies_overrides(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runs_root = repo_root / "runs"
    (repo_root / "training").mkdir(parents=True)
    (repo_root / "infra").mkdir(parents=True)
    (repo_root / "training" / "train_puffer.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "infra" / "docker-compose.yml").write_text("services:\n", encoding="utf-8")

    plan = build_training_launch_plan(
        repo_root=repo_root,
        runs_root=runs_root,
        profile_name="random_eval",
        runtime_override="host_python",
        run_id="ops-test-run",
        overrides={
            "total_env_steps": 4321,
            "window_env_steps": 321,
            "wandb_mode": "offline",
        },
        extra_args=["--eval-policy-deterministic"],
        python_executable="python",
    )

    assert plan.run_id == "ops-test-run"
    assert plan.runtime == "host_python"
    assert plan.command[0] == "python"
    assert "--run-root" in plan.command
    assert str(runs_root) in plan.command
    assert "--run-id" in plan.command
    assert "ops-test-run" in plan.command
    assert "--total-env-steps" in plan.command
    assert "4321" in plan.command
    assert plan.effective_config["wandb_mode"] == "offline"


def test_build_training_launch_plan_rejects_unknown_override(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "training").mkdir(parents=True)
    (repo_root / "infra").mkdir(parents=True)
    (repo_root / "training" / "train_puffer.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "infra" / "docker-compose.yml").write_text("services:\n", encoding="utf-8")

    try:
        build_training_launch_plan(
            repo_root=repo_root,
            runs_root=repo_root / "runs",
            profile_name="random_smoke",
            runtime_override="host_python",
            run_id=None,
            overrides={"unknown_key": 1},
            extra_args=[],
            python_executable="python",
        )
    except ValueError as exc:
        assert "Unsupported override key" in str(exc)
        return

    raise AssertionError("expected ValueError for unknown override key")


def test_ops_console_api_routes_with_fake_manager(tmp_path: Path) -> None:
    manager = FakeManager()
    app = create_ops_console_app(
        repo_root=tmp_path,
        runs_root=tmp_path / "runs",
        manager=manager,
        enforce_local_only=False,
    )
    client = TestClient(app)

    profiles_resp = client.get("/api/profiles")
    assert profiles_resp.status_code == 200
    assert profiles_resp.json()["default_profile"] == "random_eval"

    start_resp = client.post(
        "/api/job/start",
        json={
            "profile": "random_eval",
            "runtime": "host_python",
            "run_id": "ops-test",
            "overrides": {"total_env_steps": 1111},
            "extra_args": ["--wandb-mode", "disabled"],
        },
    )
    assert start_resp.status_code == 200
    assert start_resp.json()["running"] is True

    status_resp = client.get("/api/job")
    assert status_resp.status_code == 200
    assert status_resp.json()["running"] is True

    logs_resp = client.get("/api/job/logs", params={"tail": 2})
    assert logs_resp.status_code == 200
    assert logs_resp.json()["count"] == 2

    runs_resp = client.get("/api/runs", params={"limit": 10})
    assert runs_resp.status_code == 200
    assert runs_resp.json()["count"] == 1

    stop_resp = client.post("/api/job/stop", json={"force": False})
    assert stop_resp.status_code == 200
    assert stop_resp.json()["running"] is False

    # Endpoint should stay functional after stop.
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["ok"] is True
