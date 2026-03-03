from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

LOCAL_CORS_ORIGINS = (
    "http://localhost:8090",
    "http://127.0.0.1:8090",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
SUPPORTED_RUNTIMES = ("host_python", "docker_trainer")
DEFAULT_PROFILE = "random_eval"

TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "random_smoke": {
        "description": "Fast random-policy smoke run (no eval replays).",
        "runtime": "host_python",
        "defaults": {
            "trainer_backend": "random",
            "total_env_steps": 6000,
            "window_env_steps": 2000,
            "checkpoint_every_windows": 1,
            "eval_replays_per_window": 0,
            "wandb_mode": "disabled",
        },
    },
    "random_eval": {
        "description": "Random-policy run with replay generation each checkpoint window.",
        "runtime": "host_python",
        "defaults": {
            "trainer_backend": "random",
            "total_env_steps": 12000,
            "window_env_steps": 2000,
            "checkpoint_every_windows": 1,
            "eval_replays_per_window": 1,
            "eval_max_steps_per_episode": 512,
            "wandb_mode": "disabled",
        },
    },
    "ppo_short": {
        "description": "Short PPO run for local iteration checks (Docker trainer runtime).",
        "runtime": "docker_trainer",
        "defaults": {
            "trainer_backend": "puffer_ppo",
            "total_env_steps": 30000,
            "window_env_steps": 5000,
            "checkpoint_every_windows": 1,
            "eval_replays_per_window": 1,
            "ppo_num_envs": 8,
            "ppo_num_workers": 4,
            "ppo_rollout_steps": 128,
            "ppo_num_minibatches": 4,
            "ppo_update_epochs": 4,
            "ppo_env_impl": "auto",
            "wandb_mode": "disabled",
        },
    },
}

CONFIG_FLAG_BY_KEY: dict[str, str] = {
    "run_id": "--run-id",
    "total_env_steps": "--total-env-steps",
    "window_env_steps": "--window-env-steps",
    "checkpoint_every_windows": "--checkpoint-every-windows",
    "seed": "--seed",
    "env_time_max": "--env-time-max",
    "trainer_backend": "--trainer-backend",
    "wandb_mode": "--wandb-mode",
    "wandb_project": "--wandb-project",
    "eval_replays_per_window": "--eval-replays-per-window",
    "eval_max_steps_per_episode": "--eval-max-steps-per-episode",
    "ppo_num_envs": "--ppo-num-envs",
    "ppo_num_workers": "--ppo-num-workers",
    "ppo_rollout_steps": "--ppo-rollout-steps",
    "ppo_num_minibatches": "--ppo-num-minibatches",
    "ppo_update_epochs": "--ppo-update-epochs",
    "ppo_learning_rate": "--ppo-learning-rate",
    "ppo_gamma": "--ppo-gamma",
    "ppo_gae_lambda": "--ppo-gae-lambda",
    "ppo_clip_coef": "--ppo-clip-coef",
    "ppo_ent_coef": "--ppo-ent-coef",
    "ppo_vf_coef": "--ppo-vf-coef",
    "ppo_max_grad_norm": "--ppo-max-grad-norm",
    "ppo_vector_backend": "--ppo-vector-backend",
    "ppo_env_impl": "--ppo-env-impl",
}

CLI_FIELD_ORDER = (
    "run_id",
    "total_env_steps",
    "window_env_steps",
    "checkpoint_every_windows",
    "seed",
    "env_time_max",
    "trainer_backend",
    "wandb_mode",
    "wandb_project",
    "eval_replays_per_window",
    "eval_max_steps_per_episode",
    "ppo_num_envs",
    "ppo_num_workers",
    "ppo_rollout_steps",
    "ppo_num_minibatches",
    "ppo_update_epochs",
    "ppo_learning_rate",
    "ppo_gamma",
    "ppo_gae_lambda",
    "ppo_clip_coef",
    "ppo_ent_coef",
    "ppo_vf_coef",
    "ppo_max_grad_norm",
    "ppo_vector_backend",
    "ppo_env_impl",
)

INT_CONFIG_KEYS = {
    "total_env_steps",
    "window_env_steps",
    "checkpoint_every_windows",
    "seed",
    "eval_replays_per_window",
    "eval_max_steps_per_episode",
    "ppo_num_envs",
    "ppo_num_workers",
    "ppo_rollout_steps",
    "ppo_num_minibatches",
    "ppo_update_epochs",
}
POSITIVE_INT_KEYS = {
    "total_env_steps",
    "window_env_steps",
    "checkpoint_every_windows",
    "eval_max_steps_per_episode",
    "ppo_num_envs",
    "ppo_num_workers",
    "ppo_rollout_steps",
    "ppo_num_minibatches",
    "ppo_update_epochs",
}
NONNEGATIVE_INT_KEYS = {"seed", "eval_replays_per_window"}
FLOAT_CONFIG_KEYS = {
    "env_time_max",
    "ppo_learning_rate",
    "ppo_gamma",
    "ppo_gae_lambda",
    "ppo_clip_coef",
    "ppo_ent_coef",
    "ppo_vf_coef",
    "ppo_max_grad_norm",
}
STRING_CONFIG_KEYS = {
    "run_id",
    "trainer_backend",
    "wandb_mode",
    "wandb_project",
    "ppo_vector_backend",
    "ppo_env_impl",
}
CHOICE_CONFIG: dict[str, set[str]] = {
    "trainer_backend": {"random", "puffer_ppo"},
    "wandb_mode": {"disabled", "offline", "online"},
    "ppo_vector_backend": {"serial", "multiprocessing"},
    "ppo_env_impl": {"reference", "native", "auto"},
}

SUPPORTED_OVERRIDE_KEYS = tuple(sorted(key for key in CONFIG_FLAG_BY_KEY.keys() if key != "run_id"))


class StartJobRequest(BaseModel):
    profile: str = DEFAULT_PROFILE
    runtime: str | None = None
    run_id: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)


class StopJobRequest(BaseModel):
    force: bool = False


@dataclass(frozen=True)
class TrainingLaunchPlan:
    run_id: str
    profile: str
    runtime: str
    command: list[str]
    effective_config: dict[str, Any]


@dataclass
class ManagedTrainingJob:
    run_id: str
    profile: str
    runtime: str
    command: list[str]
    effective_config: dict[str, Any]
    log_path: Path
    started_at: str
    process: subprocess.Popen[str]
    log_handle: TextIO | None
    finished_at: str | None = None
    return_code: int | None = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_ops_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"ops-{timestamp}-{uuid4().hex[:6]}"


def _normalize_runtime(value: str | None, *, fallback: str) -> str:
    runtime = fallback if value is None else str(value).strip()
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(
            f"Unsupported runtime: {runtime!r}. Expected one of {', '.join(SUPPORTED_RUNTIMES)}"
        )
    return runtime


def _normalize_config_value(key: str, value: Any) -> Any:
    if key not in CONFIG_FLAG_BY_KEY:
        raise ValueError(f"Unsupported override key: {key}")

    if key in INT_CONFIG_KEYS:
        if isinstance(value, bool):
            raise ValueError(f"Override {key!r} must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Override {key!r} must be an integer") from exc

        if key in POSITIVE_INT_KEYS and parsed <= 0:
            raise ValueError(f"Override {key!r} must be > 0")
        if key in NONNEGATIVE_INT_KEYS and parsed < 0:
            raise ValueError(f"Override {key!r} must be >= 0")
        return parsed

    if key in FLOAT_CONFIG_KEYS:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Override {key!r} must be numeric") from exc
        return parsed

    if key in STRING_CONFIG_KEYS:
        text = str(value).strip()
        if text == "":
            raise ValueError(f"Override {key!r} must be a non-empty string")
        choices = CHOICE_CONFIG.get(key)
        if choices is not None and text not in choices:
            expected = ", ".join(sorted(choices))
            raise ValueError(f"Override {key!r} must be one of: {expected}")
        return text

    return value


def _build_cli_arguments(config: dict[str, Any], *, runs_root: Path) -> list[str]:
    args: list[str] = ["--run-root", str(runs_root)]
    for key in CLI_FIELD_ORDER:
        if key not in config:
            continue
        value = config[key]
        if value is None:
            continue
        args.extend([CONFIG_FLAG_BY_KEY[key], str(value)])
    return args


def build_training_launch_plan(
    *,
    repo_root: Path,
    runs_root: Path,
    profile_name: str,
    runtime_override: str | None,
    run_id: str | None,
    overrides: dict[str, Any],
    extra_args: list[str],
    python_executable: str | None = None,
) -> TrainingLaunchPlan:
    profile = TRAINING_PROFILES.get(profile_name)
    if profile is None:
        expected = ", ".join(sorted(TRAINING_PROFILES.keys()))
        raise ValueError(f"Unknown profile {profile_name!r}. Expected one of: {expected}")

    runtime = _normalize_runtime(runtime_override, fallback=str(profile["runtime"]))

    effective_config: dict[str, Any] = {}
    for key, value in dict(profile.get("defaults", {})).items():
        effective_config[key] = _normalize_config_value(key, value)

    for key, raw_value in overrides.items():
        effective_config[key] = _normalize_config_value(key, raw_value)

    if run_id is not None and str(run_id).strip() != "":
        effective_config["run_id"] = _normalize_config_value("run_id", run_id)

    resolved_run_id = str(effective_config.get("run_id") or default_ops_run_id())
    effective_config["run_id"] = resolved_run_id

    cli_args = _build_cli_arguments(effective_config, runs_root=runs_root)
    if extra_args:
        cli_args.extend([str(item) for item in extra_args])

    train_script = repo_root / "training" / "train_puffer.py"
    if not train_script.exists():
        raise ValueError(f"Missing training script: {train_script}")

    if runtime == "host_python":
        command = [python_executable or sys.executable, str(train_script), *cli_args]
    else:
        compose_file = repo_root / "infra" / "docker-compose.yml"
        if not compose_file.exists():
            raise ValueError(f"Missing docker compose file: {compose_file}")
        command = [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "run",
            "--rm",
            "-T",
            "trainer",
            "python",
            "training/train_puffer.py",
            *cli_args,
        ]

    return TrainingLaunchPlan(
        run_id=resolved_run_id,
        profile=profile_name,
        runtime=runtime,
        command=command,
        effective_config=effective_config,
    )


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_last_jsonl_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        text = line.strip()
        if text == "":
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or value.strip() == "":
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _estimate_steps_per_second(metadata: dict[str, Any], *, as_of: datetime) -> float | None:
    started_at = _parse_datetime(metadata.get("started_at"))
    if started_at is None:
        return None

    env_steps_total = metadata.get("env_steps_total")
    if isinstance(env_steps_total, bool):
        return None
    try:
        steps = float(env_steps_total)
    except (TypeError, ValueError):
        return None

    elapsed = (as_of - started_at).total_seconds()
    if elapsed <= 0.0:
        return None
    return steps / elapsed


def _count_replays(run_dir: Path) -> int:
    replay_index = _read_json_file(run_dir / "replay_index.json")
    if replay_index is not None:
        entries = replay_index.get("entries")
        if isinstance(entries, list):
            return len(entries)

    replays_dir = run_dir / "replays"
    if not replays_dir.exists():
        return 0
    return len(list(replays_dir.glob("*.jsonl.gz")))


def build_run_snapshot(*, runs_root: Path, run_id: str) -> dict[str, Any] | None:
    run_dir = runs_root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    metadata = _read_json_file(run_dir / "run_metadata.json") or {}
    latest_window = _read_last_jsonl_row(run_dir / "metrics" / "windows.jsonl")

    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_count = len(list(checkpoints_dir.glob("ckpt_*"))) if checkpoints_dir.exists() else 0
    replay_count = _count_replays(run_dir)

    now = datetime.now(UTC)
    steps_per_second_estimate = _estimate_steps_per_second(metadata, as_of=now)

    return {
        "run_id": run_id,
        "status": metadata.get("status", "unknown"),
        "env_steps_total": metadata.get("env_steps_total"),
        "windows_emitted": metadata.get("windows_emitted"),
        "checkpoints_written": metadata.get("checkpoints_written"),
        "checkpoint_count": checkpoint_count,
        "replay_count": replay_count,
        "latest_checkpoint": metadata.get("latest_checkpoint"),
        "latest_replay": metadata.get("latest_replay"),
        "updated_at": metadata.get("updated_at"),
        "steps_per_second_estimate": steps_per_second_estimate,
        "latest_window": latest_window,
    }


def list_run_snapshots(*, runs_root: Path, limit: int) -> list[dict[str, Any]]:
    run_dirs = [item for item in runs_root.glob("*") if item.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    snapshots: list[dict[str, Any]] = []
    for run_dir in run_dirs[: max(limit, 0)]:
        snapshot = build_run_snapshot(runs_root=runs_root, run_id=run_dir.name)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


class TrainingOpsManager:
    """Owns local training process lifecycle for the operator dashboard."""

    def __init__(self, *, repo_root: Path, runs_root: Path) -> None:
        self._repo_root = repo_root
        self._runs_root = runs_root
        self._lock = threading.Lock()
        self._job: ManagedTrainingJob | None = None

    def profiles_payload(self) -> dict[str, Any]:
        profiles = [
            {
                "name": name,
                "description": profile["description"],
                "runtime": profile["runtime"],
                "defaults": profile["defaults"],
            }
            for name, profile in sorted(TRAINING_PROFILES.items())
        ]
        return {
            "default_profile": DEFAULT_PROFILE,
            "runtimes": list(SUPPORTED_RUNTIMES),
            "supported_overrides": list(SUPPORTED_OVERRIDE_KEYS),
            "profiles": profiles,
        }

    def _refresh_locked(self) -> None:
        if self._job is None:
            return

        return_code = self._job.process.poll()
        if return_code is None:
            return

        if self._job.return_code is None:
            self._job.return_code = return_code
        if self._job.finished_at is None:
            self._job.finished_at = now_iso()

        if self._job.log_handle is not None and not self._job.log_handle.closed:
            self._job.log_handle.flush()
            self._job.log_handle.close()
            self._job.log_handle = None

    def start_job(
        self,
        *,
        profile: str,
        runtime: str | None,
        run_id: str | None,
        overrides: dict[str, Any],
        extra_args: list[str],
    ) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if self._job is not None and self._job.process.poll() is None:
                raise RuntimeError("A training job is already running")

            launch_plan = build_training_launch_plan(
                repo_root=self._repo_root,
                runs_root=self._runs_root,
                profile_name=profile,
                runtime_override=runtime,
                run_id=run_id,
                overrides=overrides,
                extra_args=extra_args,
            )

            logs_root = self._repo_root / "artifacts" / "ops_console" / "logs"
            logs_root.mkdir(parents=True, exist_ok=True)
            log_path = logs_root / f"{launch_plan.run_id}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(
                f"[{now_iso()}] launching profile={launch_plan.profile} "
                f"runtime={launch_plan.runtime}\\n"
            )
            log_handle.write("command: " + " ".join(launch_plan.command) + "\\n")
            log_handle.flush()

            try:
                process = subprocess.Popen(
                    launch_plan.command,
                    cwd=self._repo_root,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception:
                log_handle.close()
                raise

            self._job = ManagedTrainingJob(
                run_id=launch_plan.run_id,
                profile=launch_plan.profile,
                runtime=launch_plan.runtime,
                command=launch_plan.command,
                effective_config=launch_plan.effective_config,
                log_path=log_path,
                started_at=now_iso(),
                process=process,
                log_handle=log_handle,
            )

            return self._status_locked()

    def stop_job(self, *, force: bool) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if self._job is None:
                return {
                    "running": False,
                    "has_job": False,
                    "message": "No local training job has been launched yet.",
                }

            process = self._job.process
            if process.poll() is None:
                if force:
                    process.kill()
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()

            self._refresh_locked()
            payload = self._status_locked()
            payload["message"] = "Training job stopped."
            return payload

    def _status_locked(self) -> dict[str, Any]:
        self._refresh_locked()

        if self._job is None:
            return {
                "running": False,
                "has_job": False,
                "active_run": None,
            }

        return_code = self._job.process.poll()
        running = return_code is None
        started_at = _parse_datetime(self._job.started_at)
        uptime_seconds: float | None = None
        if started_at is not None:
            uptime_seconds = max((datetime.now(UTC) - started_at).total_seconds(), 0.0)

        active_run = build_run_snapshot(runs_root=self._runs_root, run_id=self._job.run_id)

        return {
            "running": running,
            "has_job": True,
            "pid": self._job.process.pid,
            "run_id": self._job.run_id,
            "profile": self._job.profile,
            "runtime": self._job.runtime,
            "command": self._job.command,
            "effective_config": self._job.effective_config,
            "log_path": str(self._job.log_path),
            "started_at": self._job.started_at,
            "finished_at": self._job.finished_at,
            "return_code": (
                self._job.return_code if self._job.return_code is not None else return_code
            ),
            "uptime_seconds": uptime_seconds,
            "active_run": active_run,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def tail_logs(self, *, tail_lines: int) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            if self._job is None:
                return {
                    "count": 0,
                    "lines": [],
                    "log_path": None,
                }

            log_path = self._job.log_path
            if not log_path.exists():
                return {
                    "count": 0,
                    "lines": [],
                    "log_path": str(log_path),
                }

            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-tail_lines:] if tail_lines > 0 else []
            return {
                "count": len(tail),
                "lines": tail,
                "log_path": str(log_path),
            }

    def list_runs(self, *, limit: int) -> dict[str, Any]:
        snapshots = list_run_snapshots(runs_root=self._runs_root, limit=limit)
        return {
            "count": len(snapshots),
            "runs": snapshots,
        }


def _is_local_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip().lower()
    if normalized in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return True
    return normalized.startswith("127.")


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
    <title>ABP Local Training Ops Console</title>
    <style>
      body {
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto,
          Helvetica, Arial, sans-serif;
        background: #0b1220;
        color: #e8eefb;
      }
      .shell {
        width: min(1400px, 96vw);
        margin: 1rem auto 1.4rem;
        display: grid;
        gap: 0.85rem;
      }
      .grid {
        display: grid;
        gap: 0.85rem;
        grid-template-columns: 420px minmax(0, 1fr);
      }
      .card {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.92);
        padding: 0.8rem;
      }
      h1, h2 {
        margin: 0 0 0.5rem;
      }
      h1 {
        font-size: 1.2rem;
      }
      h2 {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }
      .stack {
        display: grid;
        gap: 0.55rem;
      }
      label {
        display: grid;
        gap: 0.2rem;
        font-size: 0.76rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #9fb4de;
      }
      input, select, textarea, button {
        font: inherit;
      }
      input, select, textarea {
        width: 100%;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 9px;
        background: rgba(15, 23, 42, 0.72);
        color: #e8eefb;
        padding: 0.48rem 0.56rem;
      }
      textarea {
        min-height: 90px;
      }
      button {
        border: 1px solid transparent;
        border-radius: 9px;
        padding: 0.45rem 0.7rem;
        font-weight: 700;
        cursor: pointer;
      }
      button.primary {
        background: #f97316;
        color: #111827;
      }
      button.alt {
        background: #1d4ed8;
        color: #eff6ff;
      }
      button.warn {
        background: #dc2626;
        color: #fff7f7;
      }
      .row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
      }
      .row > * {
        flex: 1;
      }
      pre {
        margin: 0;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 10px;
        background: rgba(2, 6, 23, 0.85);
        color: #dcfce7;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.76rem;
        line-height: 1.35;
        padding: 0.65rem;
        overflow: auto;
        max-height: 320px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
      }
      th, td {
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        text-align: left;
        padding: 0.4rem;
        vertical-align: top;
      }
      .muted {
        color: #95a7cc;
      }
      @media (max-width: 1100px) {
        .grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main class=\"shell\">
      <section class=\"card\">
        <h1>ABP Local Training Ops Console</h1>
        <p class=\"muted\">
          Local-only dashboard for launching/stopping training jobs, watching logs, and
          tracking checkpoints/replays. Keep this off public deployments.
        </p>
      </section>

      <section class=\"grid\">
        <section class=\"card stack\">
          <h2>Launch Control</h2>
          <label>
            Profile
            <select id=\"profile\"></select>
          </label>
          <label>
            Runtime
            <select id=\"runtime\"></select>
          </label>
          <label>
            Run ID (optional)
            <input id=\"runId\" placeholder=\"ops-custom-20260303\" />
          </label>
          <label>
            Overrides JSON
            <textarea id=\"overrides\" spellcheck=\"false\">{}</textarea>
          </label>
          <label>
            Extra Args (one per line)
            <textarea id=\"extraArgs\" spellcheck=\"false\"></textarea>
          </label>
          <div class=\"row\">
            <button id=\"startBtn\" class=\"primary\">Start Job</button>
            <button id=\"stopBtn\" class=\"warn\">Stop Job</button>
            <button id=\"refreshBtn\" class=\"alt\">Refresh</button>
          </div>
          <pre id=\"launchHints\">Loading profile details...</pre>
        </section>

        <section class=\"stack\">
          <section class=\"card stack\">
            <h2>Active Job Status</h2>
            <pre id=\"status\">loading...</pre>
          </section>
          <section class=\"card stack\">
            <h2>Live Log Tail</h2>
            <pre id=\"logs\">loading...</pre>
          </section>
        </section>
      </section>

      <section class=\"card stack\">
        <h2>Recent Runs</h2>
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Env Steps</th>
              <th>Windows</th>
              <th>Checkpoints</th>
              <th>Replays</th>
              <th>Steps/Sec Est</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody id=\"runsBody\"></tbody>
        </table>
      </section>
    </main>

    <script>
      const profileSelect = document.getElementById('profile');
      const runtimeSelect = document.getElementById('runtime');
      const runIdInput = document.getElementById('runId');
      const overridesInput = document.getElementById('overrides');
      const extraArgsInput = document.getElementById('extraArgs');
      const launchHints = document.getElementById('launchHints');
      const statusPanel = document.getElementById('status');
      const logsPanel = document.getElementById('logs');
      const runsBody = document.getElementById('runsBody');

      let profileMap = {};

      async function fetchJson(url, options) {
        const response = await fetch(url, options || {});
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || response.statusText || 'request failed');
        }
        return payload;
      }

      function parseExtraArgs() {
        return extraArgsInput.value
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter((line) => line !== '');
      }

      function renderHints() {
        const profileName = profileSelect.value;
        const selected = profileMap[profileName];
        if (!selected) {
          launchHints.textContent = 'No profile selected.';
          return;
        }
        launchHints.textContent = JSON.stringify(selected, null, 2);
      }

      async function loadProfiles() {
        const payload = await fetchJson('/api/profiles');
        profileMap = {};
        profileSelect.innerHTML = '';
        runtimeSelect.innerHTML = '';

        for (const runtime of payload.runtimes) {
          const opt = document.createElement('option');
          opt.value = runtime;
          opt.textContent = runtime;
          runtimeSelect.appendChild(opt);
        }

        for (const profile of payload.profiles) {
          profileMap[profile.name] = profile;
          const opt = document.createElement('option');
          opt.value = profile.name;
          opt.textContent = `${profile.name} (${profile.runtime})`;
          profileSelect.appendChild(opt);
        }

        profileSelect.value = payload.default_profile;
        runtimeSelect.value = profileMap[payload.default_profile].runtime;
        renderHints();
      }

      async function refreshStatus() {
        const status = await fetchJson('/api/job');
        statusPanel.textContent = JSON.stringify(status, null, 2);
      }

      async function refreshLogs() {
        const payload = await fetchJson('/api/job/logs?tail=140');
        logsPanel.textContent = payload.lines.join('\n');
      }

      async function refreshRuns() {
        const payload = await fetchJson('/api/runs?limit=20');
        runsBody.innerHTML = '';
        for (const row of payload.runs) {
          const tr = document.createElement('tr');
          const stepsPerSec = (typeof row.steps_per_second_estimate === 'number')
            ? row.steps_per_second_estimate.toFixed(2)
            : '-';
          tr.innerHTML = `
            <td>${row.run_id || '-'}</td>
            <td>${row.status || '-'}</td>
            <td>${row.env_steps_total ?? '-'}</td>
            <td>${row.windows_emitted ?? '-'}</td>
            <td>${row.checkpoint_count ?? '-'}</td>
            <td>${row.replay_count ?? '-'}</td>
            <td>${stepsPerSec}</td>
            <td>${row.updated_at || '-'}</td>
          `;
          runsBody.appendChild(tr);
        }
      }

      async function refreshAll() {
        await Promise.all([refreshStatus(), refreshLogs(), refreshRuns()]);
      }

      async function startJob() {
        let overrides;
        try {
          overrides = JSON.parse(overridesInput.value || '{}');
        } catch (error) {
          alert('Overrides JSON is invalid.');
          return;
        }

        const payload = {
          profile: profileSelect.value,
          runtime: runtimeSelect.value,
          run_id: runIdInput.value.trim() || null,
          overrides,
          extra_args: parseExtraArgs(),
        };

        try {
          await fetchJson('/api/job/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          await refreshAll();
        } catch (error) {
          alert(error.message || String(error));
        }
      }

      async function stopJob() {
        try {
          await fetchJson('/api/job/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ force: false }),
          });
          await refreshAll();
        } catch (error) {
          alert(error.message || String(error));
        }
      }

      profileSelect.addEventListener('change', () => {
        const selected = profileMap[profileSelect.value];
        if (selected) {
          runtimeSelect.value = selected.runtime;
        }
        renderHints();
      });

      document.getElementById('startBtn').addEventListener('click', startJob);
      document.getElementById('stopBtn').addEventListener('click', stopJob);
      document.getElementById('refreshBtn').addEventListener('click', refreshAll);

      (async () => {
        await loadProfiles();
        await refreshAll();
        setInterval(refreshAll, 4000);
      })();
    </script>
  </body>
</html>
"""


def create_ops_console_app(
    *,
    repo_root: Path,
    runs_root: Path,
    manager: TrainingOpsManager | None = None,
    enforce_local_only: bool = True,
) -> FastAPI:
    app = FastAPI(title="ABP Local Training Ops Console", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(LOCAL_CORS_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.manager = manager or TrainingOpsManager(repo_root=repo_root, runs_root=runs_root)

    if enforce_local_only:

        @app.middleware("http")
        async def _local_only_guard(request: Request, call_next):
            client_host = request.client.host if request.client is not None else None
            if not _is_local_host(client_host):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Local-only dashboard. Bind and access via localhost."},
                )
            return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        status = app.state.manager.status()
        return {
            "ok": True,
            "local_only": enforce_local_only,
            "running": bool(status.get("running", False)),
        }

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return _dashboard_html()

    @app.get("/api/profiles")
    async def profiles() -> dict[str, Any]:
        return app.state.manager.profiles_payload()

    @app.get("/api/job")
    async def job_status() -> dict[str, Any]:
        return app.state.manager.status()

    @app.post("/api/job/start")
    async def job_start(payload: StartJobRequest) -> dict[str, Any]:
        try:
            return app.state.manager.start_job(
                profile=payload.profile,
                runtime=payload.runtime,
                run_id=payload.run_id,
                overrides=payload.overrides,
                extra_args=payload.extra_args,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/job/stop")
    async def job_stop(payload: StopJobRequest) -> dict[str, Any]:
        return app.state.manager.stop_job(force=payload.force)

    @app.get("/api/job/logs")
    async def job_logs(tail: int = Query(default=120, ge=0, le=2000)) -> dict[str, Any]:
        return app.state.manager.tail_logs(tail_lines=tail)

    @app.get("/api/runs")
    async def list_runs(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        return app.state.manager.list_runs(limit=limit)

    return app
