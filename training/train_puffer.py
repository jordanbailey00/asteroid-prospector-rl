from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from training.logging import JsonlWindowLogger, RunPaths, WandbWindowLogger
    from training.windowing import INFO_METRIC_KEYS, WindowMetricsAggregator
else:
    from .logging import JsonlWindowLogger, RunPaths, WandbWindowLogger
    from .windowing import INFO_METRIC_KEYS, WindowMetricsAggregator

DEFAULT_N_ACTIONS = 69
SUPPORTED_TRAINER_BACKENDS = ("random", "puffer_ppo")


def _ensure_python_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python_src = repo_root / "python"
    if str(python_src) not in sys.path:
        sys.path.insert(0, str(python_src))


@dataclass(frozen=True)
class TrainConfig:
    run_root: Path = Path("runs")
    run_id: str | None = None
    total_env_steps: int = 6000
    window_env_steps: int = 2000
    checkpoint_every_windows: int = 1
    seed: int = 0
    env_time_max: float = 20000.0

    trainer_backend: str = "random"
    wandb_mode: str = "disabled"  # disabled|offline|online
    wandb_project: str = "asteroid-prospector"

    flush_partial_window: bool = False


def default_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sha = "nogit"
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
            .lower()
        )
    except Exception:
        pass
    return f"{ts}-{sha}"


def resolve_run_paths(run_root: Path, run_id: str) -> RunPaths:
    run_dir = run_root / run_id
    checkpoints_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        metrics_dir=metrics_dir,
        metrics_windows_path=metrics_dir / "windows.jsonl",
        config_path=run_dir / "config.json",
        metadata_path=run_dir / "run_metadata.json",
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def as_posix_relative(path: Path, *, start: Path) -> str:
    try:
        return path.relative_to(start).as_posix()
    except ValueError:
        return path.as_posix()


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_backend(backend: str) -> None:
    if backend == "random":
        return
    if backend != "puffer_ppo":
        raise ValueError(f"Unsupported trainer_backend: {backend}")
    if sys.platform.startswith("win"):
        raise RuntimeError(
            "trainer_backend='puffer_ppo' is currently blocked on Windows because PufferLib "
            "does not support this platform. Run under WSL2/Linux or use "
            "--trainer-backend random."
        )
    try:
        import pufferlib  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "trainer_backend='puffer_ppo' requested but pufferlib is not installed. "
            "Install pufferlib on a supported platform or use --trainer-backend random."
        ) from exc
    raise NotImplementedError(
        "trainer_backend='puffer_ppo' dependency checks passed, but PPO loop wiring is not "
        "implemented in this repository yet."
    )


def choose_action(*, rng: np.random.Generator, obs: np.ndarray, backend: str) -> int:
    del obs
    if backend != "random":
        raise ValueError(f"Unsupported trainer_backend: {backend}")
    return int(rng.integers(0, DEFAULT_N_ACTIONS))


def write_checkpoint(*, path: Path, run_id: str, window_id: int, env_steps_total: int) -> None:
    payload = {
        "run_id": run_id,
        "window_id": int(window_id),
        "env_steps_total": int(env_steps_total),
        "created_at": now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_training(cfg: TrainConfig) -> dict[str, Any]:
    _ensure_python_src_on_path()

    from asteroid_prospector import N_ACTIONS, ProspectorReferenceEnv, ReferenceEnvConfig

    if cfg.total_env_steps <= 0:
        raise ValueError("total_env_steps must be positive")
    if cfg.window_env_steps <= 0:
        raise ValueError("window_env_steps must be positive")
    if cfg.checkpoint_every_windows <= 0:
        raise ValueError("checkpoint_every_windows must be positive")
    validate_backend(cfg.trainer_backend)

    run_id = cfg.run_id or default_run_id()
    run_paths = resolve_run_paths(cfg.run_root, run_id)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_paths.metrics_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(cfg)
    config_payload["run_id"] = run_id
    config_payload["run_root"] = str(cfg.run_root)
    run_paths.config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    wandb_logger = WandbWindowLogger.create(
        run_id=run_id,
        project=cfg.wandb_project,
        config=config_payload,
        mode=cfg.wandb_mode,
        tags=["m3", "windowed-training"],
    )

    env = ProspectorReferenceEnv(
        config=ReferenceEnvConfig(time_max=cfg.env_time_max),
        seed=cfg.seed,
    )
    obs, _ = env.reset(seed=cfg.seed)

    aggregator = WindowMetricsAggregator(run_id=run_id, window_env_steps=cfg.window_env_steps)
    jsonl_logger = JsonlWindowLogger(path=run_paths.metrics_windows_path)

    rng = np.random.default_rng(cfg.seed + 17)
    episode_seed = cfg.seed

    windows_emitted = 0
    checkpoints_written = 0

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "trainer_backend": cfg.trainer_backend,
        "target_env_steps": cfg.total_env_steps,
        "window_env_steps": cfg.window_env_steps,
        "env_steps_total": 0,
        "episodes_total": 0,
        "windows_emitted": 0,
        "checkpoints_written": 0,
        "latest_window": None,
        "latest_checkpoint": None,
        "latest_replay": None,
        "metrics_windows_path": as_posix_relative(
            run_paths.metrics_windows_path, start=run_paths.run_dir
        ),
        "checkpoints_dir": as_posix_relative(run_paths.checkpoints_dir, start=run_paths.run_dir),
        "replays_dir": "replays",
        "replay_index_path": None,
        "config_path": as_posix_relative(run_paths.config_path, start=run_paths.run_dir),
        "wandb_run_url": wandb_logger.run_url if wandb_logger is not None else None,
        "constellation_url": None,
        "info_metric_keys": list(INFO_METRIC_KEYS),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "finished_at": None,
    }
    write_metadata(run_paths.metadata_path, metadata)

    try:
        while aggregator.env_steps_total < cfg.total_env_steps:
            action = choose_action(rng=rng, obs=obs, backend=cfg.trainer_backend)
            if action >= N_ACTIONS:
                action = action % N_ACTIONS

            obs, reward, terminated, truncated, info = env.step(action)

            records = aggregator.record_step(
                reward=float(reward),
                info=info,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

            for record in records:
                payload = record.to_dict()
                jsonl_logger.log_window(payload)
                windows_emitted += 1

                if wandb_logger is not None:
                    wandb_logger.log_window(payload, step=record.env_steps_total)

                metadata["env_steps_total"] = aggregator.env_steps_total
                metadata["episodes_total"] = aggregator.episodes_total
                metadata["windows_emitted"] = windows_emitted
                metadata["latest_window"] = {
                    "window_id": record.window_id,
                    "window_complete": record.window_complete,
                    "env_steps_start": record.env_steps_start,
                    "env_steps_end": record.env_steps_end,
                    "env_steps_in_window": record.env_steps_in_window,
                    "env_steps_total": record.env_steps_total,
                    "metrics_row_path": metadata["metrics_windows_path"],
                }

                if record.window_id % cfg.checkpoint_every_windows == 0:
                    ckpt_path = run_paths.checkpoints_dir / f"ckpt_{record.window_id:06d}.pt"
                    write_checkpoint(
                        path=ckpt_path,
                        run_id=run_id,
                        window_id=record.window_id,
                        env_steps_total=record.env_steps_total,
                    )
                    checkpoints_written += 1
                    metadata["checkpoints_written"] = checkpoints_written
                    metadata["latest_checkpoint"] = {
                        "window_id": record.window_id,
                        "env_steps_total": record.env_steps_total,
                        "path": as_posix_relative(ckpt_path, start=run_paths.run_dir),
                        "created_at": now_iso(),
                    }
                    if wandb_logger is not None:
                        wandb_logger.log_checkpoint(
                            checkpoint_path=ckpt_path,
                            run_id=run_id,
                            window_id=record.window_id,
                        )

                metadata["updated_at"] = now_iso()
                write_metadata(run_paths.metadata_path, metadata)

            if terminated or truncated:
                episode_seed += 1
                obs, _ = env.reset(seed=episode_seed)

        if cfg.flush_partial_window:
            partial = aggregator.flush_partial()
            if partial is not None:
                payload = partial.to_dict()
                jsonl_logger.log_window(payload)
                windows_emitted += 1
                if wandb_logger is not None:
                    wandb_logger.log_window(payload, step=partial.env_steps_total)
                metadata["env_steps_total"] = aggregator.env_steps_total
                metadata["episodes_total"] = aggregator.episodes_total
                metadata["windows_emitted"] = windows_emitted
                metadata["latest_window"] = {
                    "window_id": partial.window_id,
                    "window_complete": partial.window_complete,
                    "env_steps_start": partial.env_steps_start,
                    "env_steps_end": partial.env_steps_end,
                    "env_steps_in_window": partial.env_steps_in_window,
                    "env_steps_total": partial.env_steps_total,
                    "metrics_row_path": metadata["metrics_windows_path"],
                }
                metadata["updated_at"] = now_iso()
                write_metadata(run_paths.metadata_path, metadata)

        summary = {
            "run_id": run_id,
            "status": "completed",
            "trainer_backend": cfg.trainer_backend,
            "env_steps_total": aggregator.env_steps_total,
            "episodes_total": aggregator.episodes_total,
            "windows_emitted": windows_emitted,
            "checkpoints_written": checkpoints_written,
            "window_env_steps": cfg.window_env_steps,
            "info_metric_keys": list(INFO_METRIC_KEYS),
            "latest_window": metadata["latest_window"],
            "latest_checkpoint": metadata["latest_checkpoint"],
            "latest_replay": metadata["latest_replay"],
            "replay_index_path": metadata["replay_index_path"],
            "wandb_run_url": wandb_logger.run_url if wandb_logger is not None else None,
            "constellation_url": metadata["constellation_url"],
            "finished_at": now_iso(),
        }

        if wandb_logger is not None:
            wandb_logger.finish(
                {
                    "env_steps_total": aggregator.env_steps_total,
                    "episodes_total": aggregator.episodes_total,
                    "windows_emitted": windows_emitted,
                }
            )

        metadata.update(summary)
        metadata["updated_at"] = summary["finished_at"]
        metadata["finished_at"] = summary["finished_at"]
        write_metadata(run_paths.metadata_path, metadata)
        return summary
    except Exception as exc:
        failure_time = now_iso()
        metadata.update(
            {
                "status": "failed",
                "env_steps_total": aggregator.env_steps_total,
                "episodes_total": aggregator.episodes_total,
                "windows_emitted": windows_emitted,
                "checkpoints_written": checkpoints_written,
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": failure_time,
                "finished_at": failure_time,
            }
        )
        write_metadata(run_paths.metadata_path, metadata)
        if wandb_logger is not None:
            wandb_logger.finish(
                {
                    "status": "failed",
                    "env_steps_total": aggregator.env_steps_total,
                    "episodes_total": aggregator.episodes_total,
                    "windows_emitted": windows_emitted,
                }
            )
        raise


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Windowed training runner (M3)")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--total-env-steps", type=int, default=6000)
    parser.add_argument("--window-env-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every-windows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-time-max", type=float, default=20000.0)
    parser.add_argument(
        "--trainer-backend",
        choices=list(SUPPORTED_TRAINER_BACKENDS),
        default="random",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
    parser.add_argument("--wandb-project", type=str, default="asteroid-prospector")
    parser.add_argument("--flush-partial-window", action="store_true")

    args = parser.parse_args()
    return TrainConfig(
        run_root=args.run_root,
        run_id=args.run_id,
        total_env_steps=args.total_env_steps,
        window_env_steps=args.window_env_steps,
        checkpoint_every_windows=args.checkpoint_every_windows,
        seed=args.seed,
        env_time_max=args.env_time_max,
        trainer_backend=args.trainer_backend,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        flush_partial_window=args.flush_partial_window,
    )


def main() -> int:
    cfg = _parse_args()
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
