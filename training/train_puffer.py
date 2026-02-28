from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from training.eval_runner import EvalReplayConfig, run_eval_and_record_replay
    from training.logging import JsonlWindowLogger, RunPaths, WandbWindowLogger
    from training.windowing import INFO_METRIC_KEYS, WindowMetricsAggregator, WindowRecord
else:
    from .eval_runner import EvalReplayConfig, run_eval_and_record_replay
    from .logging import JsonlWindowLogger, RunPaths, WandbWindowLogger
    from .windowing import INFO_METRIC_KEYS, WindowMetricsAggregator, WindowRecord

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

    # M4 eval/replay parameters.
    eval_replays_per_window: int = 0
    eval_max_steps_per_episode: int = 512
    eval_include_info: bool = True
    eval_policy_deterministic: bool = True
    eval_seed_offset: int = 100000
    eval_milestone_profit_thresholds: tuple[float, ...] = (100.0, 500.0, 1000.0)
    eval_milestone_return_thresholds: tuple[float, ...] = (10.0, 25.0, 50.0)
    eval_milestone_survival_thresholds: tuple[float, ...] = (1.0,)

    # PPO backend parameters (used when trainer_backend == "puffer_ppo")
    ppo_num_envs: int = 8
    ppo_num_workers: int = 4
    ppo_rollout_steps: int = 128
    ppo_num_minibatches: int = 4
    ppo_update_epochs: int = 4
    ppo_learning_rate: float = 3.0e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_coef: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_vector_backend: str = "multiprocessing"  # serial|multiprocessing


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


def parse_thresholds_csv(raw: str) -> tuple[float, ...]:
    text = raw.strip()
    if text == "":
        return tuple()

    values: list[float] = []
    for item in text.split(","):
        part = item.strip()
        if part == "":
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"Invalid threshold value: {part!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"Invalid threshold value: {part!r}")
        if value < 0.0:
            raise ValueError(f"Threshold values must be non-negative: {part!r}")
        values.append(value)

    return tuple(sorted(set(values)))


def validate_thresholds(name: str, values: tuple[float, ...]) -> None:
    for value in values:
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} contains non-finite value: {value}")
        if float(value) < 0.0:
            raise ValueError(f"{name} contains negative value: {value}")


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
            "trainer_backend='puffer_ppo' requires Linux runtime. Use Docker compose service "
            "'trainer' or WSL2/Linux directly."
        )

    try:
        import pufferlib  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "trainer_backend='puffer_ppo' requested but pufferlib is not installed."
        ) from exc

    try:
        import torch  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "trainer_backend='puffer_ppo' requested but torch is not installed."
        ) from exc


def choose_action(*, rng: np.random.Generator, obs: np.ndarray, backend: str) -> int:
    del obs
    if backend != "random":
        raise ValueError(f"Unsupported trainer_backend: {backend}")
    return int(rng.integers(0, DEFAULT_N_ACTIONS))


def write_checkpoint(
    *,
    path: Path,
    run_id: str,
    window_id: int,
    env_steps_total: int,
    trainer_backend: str,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "window_id": int(window_id),
        "env_steps_total": int(env_steps_total),
        "trainer_backend": trainer_backend,
        "created_at": now_iso(),
    }
    if extra_payload is not None:
        payload.update(extra_payload)

    path.parent.mkdir(parents=True, exist_ok=True)

    if trainer_backend == "puffer_ppo":
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - guarded by validate_backend
            raise RuntimeError("torch is required to write puffer_ppo checkpoints") from exc

        payload["checkpoint_format"] = "ppo_torch_v1"
        torch.save(payload, path)
        return

    payload["checkpoint_format"] = "json_v1"
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
    if cfg.eval_replays_per_window < 0:
        raise ValueError("eval_replays_per_window must be non-negative")
    if cfg.eval_max_steps_per_episode <= 0:
        raise ValueError("eval_max_steps_per_episode must be positive")
    validate_thresholds(
        "eval_milestone_profit_thresholds",
        cfg.eval_milestone_profit_thresholds,
    )
    validate_thresholds(
        "eval_milestone_return_thresholds",
        cfg.eval_milestone_return_thresholds,
    )
    validate_thresholds(
        "eval_milestone_survival_thresholds",
        cfg.eval_milestone_survival_thresholds,
    )
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

    aggregator = WindowMetricsAggregator(run_id=run_id, window_env_steps=cfg.window_env_steps)
    jsonl_logger = JsonlWindowLogger(path=run_paths.metrics_windows_path)

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

    ppo_checkpoint_state_getter: Callable[[], dict[str, Any]] | None = None

    def emit_window_record(record: WindowRecord) -> None:
        nonlocal windows_emitted
        nonlocal checkpoints_written

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
            checkpoint_extra_payload: dict[str, Any] | None = None
            if cfg.trainer_backend == "puffer_ppo":
                if ppo_checkpoint_state_getter is None:
                    raise RuntimeError(
                        "puffer_ppo checkpoint requested before policy state getter was registered"
                    )
                checkpoint_extra_payload = ppo_checkpoint_state_getter()

            write_checkpoint(
                path=ckpt_path,
                run_id=run_id,
                window_id=record.window_id,
                env_steps_total=record.env_steps_total,
                trainer_backend=cfg.trainer_backend,
                extra_payload=checkpoint_extra_payload,
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

            if cfg.eval_replays_per_window > 0:
                eval_result = run_eval_and_record_replay(
                    EvalReplayConfig(
                        run_id=run_id,
                        run_dir=run_paths.run_dir,
                        checkpoint_path=ckpt_path,
                        window_id=record.window_id,
                        trainer_backend=cfg.trainer_backend,
                        env_time_max=cfg.env_time_max,
                        base_seed=cfg.seed + cfg.eval_seed_offset,
                        num_episodes=cfg.eval_replays_per_window,
                        max_steps_per_episode=cfg.eval_max_steps_per_episode,
                        include_info=cfg.eval_include_info,
                        policy_deterministic=cfg.eval_policy_deterministic,
                        milestone_profit_thresholds=cfg.eval_milestone_profit_thresholds,
                        milestone_return_thresholds=cfg.eval_milestone_return_thresholds,
                        milestone_survival_thresholds=cfg.eval_milestone_survival_thresholds,
                    )
                )
                metadata["latest_replay"] = eval_result.replay_entry
                metadata["replay_index_path"] = eval_result.replay_index_path_relative

                if wandb_logger is not None:
                    replay_tags_raw = eval_result.replay_entry.get("tags", [])
                    replay_tags = replay_tags_raw if isinstance(replay_tags_raw, list) else []
                    wandb_logger.log_replay(
                        replay_path=eval_result.replay_path,
                        run_id=run_id,
                        window_id=record.window_id,
                        replay_id=eval_result.replay_id,
                        tags=[str(tag) for tag in replay_tags],
                    )

        metadata["updated_at"] = now_iso()
        write_metadata(run_paths.metadata_path, metadata)

    try:
        if cfg.trainer_backend == "random":
            env = ProspectorReferenceEnv(
                config=ReferenceEnvConfig(time_max=cfg.env_time_max),
                seed=cfg.seed,
            )
            obs, _ = env.reset(seed=cfg.seed)

            rng = np.random.default_rng(cfg.seed + 17)
            episode_seed = cfg.seed

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
                    emit_window_record(record)

                if terminated or truncated:
                    episode_seed += 1
                    obs, _ = env.reset(seed=episode_seed)
        else:
            if cfg.trainer_backend != "puffer_ppo":
                raise ValueError(f"Unsupported trainer_backend: {cfg.trainer_backend}")

            if __package__ is None or __package__ == "":
                from training.puffer_backend import PpoConfig, run_puffer_ppo_training
            else:
                from .puffer_backend import PpoConfig, run_puffer_ppo_training

            def on_step(
                reward: float, info: dict[str, Any], terminated: bool, truncated: bool
            ) -> bool:
                records = aggregator.record_step(
                    reward=reward,
                    info=info,
                    terminated=terminated,
                    truncated=truncated,
                )
                for record in records:
                    emit_window_record(record)
                return aggregator.env_steps_total >= cfg.total_env_steps

            def register_checkpoint_state_getter(getter: Callable[[], dict[str, Any]]) -> None:
                nonlocal ppo_checkpoint_state_getter
                ppo_checkpoint_state_getter = getter

            ppo_summary = run_puffer_ppo_training(
                cfg=PpoConfig(
                    total_env_steps=cfg.total_env_steps,
                    seed=cfg.seed,
                    env_time_max=cfg.env_time_max,
                    num_envs=cfg.ppo_num_envs,
                    num_workers=cfg.ppo_num_workers,
                    rollout_steps=cfg.ppo_rollout_steps,
                    num_minibatches=cfg.ppo_num_minibatches,
                    update_epochs=cfg.ppo_update_epochs,
                    learning_rate=cfg.ppo_learning_rate,
                    gamma=cfg.ppo_gamma,
                    gae_lambda=cfg.ppo_gae_lambda,
                    clip_coef=cfg.ppo_clip_coef,
                    ent_coef=cfg.ppo_ent_coef,
                    vf_coef=cfg.ppo_vf_coef,
                    max_grad_norm=cfg.ppo_max_grad_norm,
                    vector_backend=cfg.ppo_vector_backend,
                ),
                on_step=on_step,
                register_checkpoint_state_getter=register_checkpoint_state_getter,
            )
            metadata.update(ppo_summary)
            metadata["updated_at"] = now_iso()
            write_metadata(run_paths.metadata_path, metadata)

        if cfg.flush_partial_window:
            partial = aggregator.flush_partial()
            if partial is not None:
                emit_window_record(partial)

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

        if cfg.trainer_backend == "puffer_ppo":
            for key in (
                "ppo_device",
                "ppo_num_envs",
                "ppo_num_workers",
                "ppo_rollout_steps",
                "ppo_policy_updates",
                "ppo_vector_backend",
                "ppo_policy_arch",
            ):
                if key in metadata:
                    summary[key] = metadata[key]

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
    parser = argparse.ArgumentParser(description="Windowed training runner (M3/M4)")
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

    parser.add_argument("--eval-replays-per-window", type=int, default=0)
    parser.add_argument("--eval-max-steps-per-episode", type=int, default=512)

    eval_info_group = parser.add_mutually_exclusive_group()
    eval_info_group.add_argument(
        "--eval-include-info",
        dest="eval_include_info",
        action="store_true",
        help="Include raw info payload in replay frames (default: enabled).",
    )
    eval_info_group.add_argument(
        "--no-eval-include-info",
        dest="eval_include_info",
        action="store_false",
        help="Exclude raw info payload from replay frames.",
    )

    eval_policy_group = parser.add_mutually_exclusive_group()
    eval_policy_group.add_argument(
        "--eval-policy-deterministic",
        dest="eval_policy_deterministic",
        action="store_true",
        help="Use argmax action selection for PPO eval replays (default).",
    )
    eval_policy_group.add_argument(
        "--eval-policy-stochastic",
        dest="eval_policy_deterministic",
        action="store_false",
        help="Sample PPO eval actions from the checkpoint policy distribution.",
    )

    parser.set_defaults(eval_include_info=True)
    parser.set_defaults(eval_policy_deterministic=True)
    parser.add_argument("--eval-seed-offset", type=int, default=100000)
    parser.add_argument(
        "--eval-milestone-profit-thresholds",
        type=str,
        default="100,500,1000",
        help="Comma-separated profit thresholds for milestone tags.",
    )
    parser.add_argument(
        "--eval-milestone-return-thresholds",
        type=str,
        default="10,25,50",
        help="Comma-separated return thresholds for milestone tags.",
    )
    parser.add_argument(
        "--eval-milestone-survival-thresholds",
        type=str,
        default="1.0",
        help="Comma-separated survival thresholds for milestone tags.",
    )

    parser.add_argument("--ppo-num-envs", type=int, default=8)
    parser.add_argument("--ppo-num-workers", type=int, default=4)
    parser.add_argument("--ppo-rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-num-minibatches", type=int, default=4)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--ppo-gamma", type=float, default=0.99)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip-coef", type=float, default=0.2)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-vf-coef", type=float, default=0.5)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--ppo-vector-backend",
        choices=["serial", "multiprocessing"],
        default="multiprocessing",
    )

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
        eval_replays_per_window=args.eval_replays_per_window,
        eval_max_steps_per_episode=args.eval_max_steps_per_episode,
        eval_include_info=args.eval_include_info,
        eval_policy_deterministic=args.eval_policy_deterministic,
        eval_seed_offset=args.eval_seed_offset,
        eval_milestone_profit_thresholds=parse_thresholds_csv(
            args.eval_milestone_profit_thresholds
        ),
        eval_milestone_return_thresholds=parse_thresholds_csv(
            args.eval_milestone_return_thresholds
        ),
        eval_milestone_survival_thresholds=parse_thresholds_csv(
            args.eval_milestone_survival_thresholds
        ),
        ppo_num_envs=args.ppo_num_envs,
        ppo_num_workers=args.ppo_num_workers,
        ppo_rollout_steps=args.ppo_rollout_steps,
        ppo_num_minibatches=args.ppo_num_minibatches,
        ppo_update_epochs=args.ppo_update_epochs,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_gamma=args.ppo_gamma,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_clip_coef=args.ppo_clip_coef,
        ppo_ent_coef=args.ppo_ent_coef,
        ppo_vf_coef=args.ppo_vf_coef,
        ppo_max_grad_norm=args.ppo_max_grad_norm,
        ppo_vector_backend=args.ppo_vector_backend,
    )


def main() -> int:
    cfg = _parse_args()
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
