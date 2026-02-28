from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from replay.index import append_replay_entry, load_replay_index
from replay.schema import frame_from_step, now_iso, validate_replay_frame

if __package__ is None or __package__ == "":
    from training.policy import (
        POLICY_ARCH,
        create_actor_critic,
        load_policy_state_dict,
        select_policy_action,
    )
else:
    from .policy import (
        POLICY_ARCH,
        create_actor_critic,
        load_policy_state_dict,
        select_policy_action,
    )


@dataclass(frozen=True)
class EvalReplayConfig:
    run_id: str
    run_dir: Path
    checkpoint_path: Path
    window_id: int
    trainer_backend: str
    env_time_max: float
    base_seed: int
    num_episodes: int = 1
    max_steps_per_episode: int = 512
    include_info: bool = True
    policy_deterministic: bool = True
    milestone_profit_thresholds: tuple[float, ...] = ()
    milestone_return_thresholds: tuple[float, ...] = ()
    milestone_survival_thresholds: tuple[float, ...] = ()


@dataclass(frozen=True)
class EvalReplayResult:
    replay_id: str
    replay_path: Path
    replay_path_relative: str
    replay_index_path: Path
    replay_index_path_relative: str
    replay_entry: dict[str, Any]


def _as_relative_posix(path: Path, *, start: Path) -> str:
    try:
        return path.relative_to(start).as_posix()
    except ValueError:
        return path.as_posix()


def _build_render_state(*, obs: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
    obs_list = [float(v) for v in np.asarray(obs, dtype=np.float32).tolist()]
    return {
        "observation": obs_list,
        "time_remaining": float(info.get("time_remaining", 0.0)),
        "credits": float(info.get("credits", 0.0)),
        "net_profit": float(info.get("net_profit", 0.0)),
        "survival": float(info.get("survival", 0.0)),
        "cargo_utilization_avg": float(info.get("cargo_utilization_avg", 0.0)),
        "node_context": str(info.get("node_context", "unknown")),
    }


def _derive_events(
    *,
    info: dict[str, Any],
    prev_info: dict[str, Any] | None,
    terminated: bool,
    truncated: bool,
) -> list[str]:
    events: list[str] = []

    if bool(info.get("invalid_action", False)):
        events.append("invalid_action")

    current_pirates = float(info.get("pirate_encounters", 0.0))
    prev_pirates = float((prev_info or {}).get("pirate_encounters", 0.0))
    if current_pirates > prev_pirates:
        events.append("pirate_encounter")

    current_overheat = float(info.get("overheat_ticks", 0.0))
    prev_overheat = float((prev_info or {}).get("overheat_ticks", 0.0))
    if current_overheat > prev_overheat:
        events.append("overheat_tick")

    if terminated:
        events.append("terminated")
    if truncated:
        events.append("truncated")

    return events


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load non-JSON checkpoint because torch is not installed"
        ) from exc

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload).__name__}")
    return payload


def _format_threshold_value(value: float) -> str:
    formatted = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return formatted if formatted != "" else "0"


def _compute_milestone_tags(
    *,
    return_total: float,
    profit: float,
    survival: float,
    profit_thresholds: tuple[float, ...],
    return_thresholds: tuple[float, ...],
    survival_thresholds: tuple[float, ...],
) -> list[str]:
    tags: list[str] = []

    for threshold in profit_thresholds:
        if profit >= threshold:
            tags.append(f"milestone:profit:{_format_threshold_value(threshold)}")

    for threshold in return_thresholds:
        if return_total >= threshold:
            tags.append(f"milestone:return:{_format_threshold_value(threshold)}")

    for threshold in survival_thresholds:
        if survival >= threshold:
            tags.append(f"milestone:survival:{_format_threshold_value(threshold)}")

    return tags


def _compute_replay_tags(
    *,
    replay_index_path: Path,
    run_id: str,
    return_total: float,
    profit: float,
    survival: float,
    profit_thresholds: tuple[float, ...],
    return_thresholds: tuple[float, ...],
    survival_thresholds: tuple[float, ...],
) -> list[str]:
    tags = ["every_window"]
    index_payload = load_replay_index(path=replay_index_path, run_id=run_id)
    prior_entries = index_payload.get("entries", [])

    prior_best_return = float("-inf")
    if isinstance(prior_entries, list):
        for entry in prior_entries:
            if not isinstance(entry, dict):
                continue
            maybe_return = entry.get("return_total")
            try:
                prior_best_return = max(prior_best_return, float(maybe_return))
            except (TypeError, ValueError):
                continue

    if return_total > prior_best_return:
        tags.append("best_so_far")

    tags.extend(
        _compute_milestone_tags(
            return_total=return_total,
            profit=profit,
            survival=survival,
            profit_thresholds=profit_thresholds,
            return_thresholds=return_thresholds,
            survival_thresholds=survival_thresholds,
        )
    )

    return list(dict.fromkeys(tags))


def _load_policy_for_eval(
    *,
    cfg: EvalReplayConfig,
    checkpoint_payload: dict[str, Any],
) -> Any | None:
    checkpoint_backend = str(checkpoint_payload.get("trainer_backend", cfg.trainer_backend))
    if checkpoint_backend != "puffer_ppo":
        return None

    model_state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError("puffer_ppo checkpoint is missing model_state_dict")

    policy_arch = str(checkpoint_payload.get("policy_arch", ""))
    if policy_arch != POLICY_ARCH:
        raise ValueError(
            f"Unsupported policy_arch in checkpoint: {policy_arch!r} (expected {POLICY_ARCH!r})"
        )

    raw_obs_shape = checkpoint_payload.get("obs_shape")
    if not isinstance(raw_obs_shape, (list, tuple)):
        raise ValueError("puffer_ppo checkpoint is missing obs_shape")

    obs_shape = tuple(int(v) for v in raw_obs_shape)
    n_actions = int(checkpoint_payload.get("n_actions", 0))
    if n_actions <= 0:
        raise ValueError("puffer_ppo checkpoint is missing n_actions")

    model = create_actor_critic(obs_shape=obs_shape, n_actions=n_actions, device="cpu")
    load_policy_state_dict(model, model_state_dict)
    model.eval()
    return model


def run_eval_and_record_replay(cfg: EvalReplayConfig) -> EvalReplayResult:
    if cfg.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if cfg.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive")

    from asteroid_prospector import N_ACTIONS, ProspectorReferenceEnv, ReferenceEnvConfig

    checkpoint_payload = _load_checkpoint_payload(cfg.checkpoint_path)
    checkpoint_backend = str(checkpoint_payload.get("trainer_backend", cfg.trainer_backend))
    policy_model = _load_policy_for_eval(cfg=cfg, checkpoint_payload=checkpoint_payload)

    best_frames: list[dict[str, Any]] | None = None
    best_return = float("-inf")
    best_steps = 0
    best_last_info: dict[str, Any] = {}
    best_terminated = False
    best_truncated = False

    for episode_idx in range(cfg.num_episodes):
        eval_seed = cfg.base_seed + cfg.window_id * 1000 + episode_idx
        rng = np.random.default_rng(eval_seed + 17)

        env = ProspectorReferenceEnv(
            config=ReferenceEnvConfig(time_max=cfg.env_time_max),
            seed=eval_seed,
        )
        obs, info = env.reset(seed=eval_seed)

        prev_info: dict[str, Any] | None = None
        frames: list[dict[str, Any]] = []
        return_total = 0.0
        t = 0
        terminated = False
        truncated = False

        for frame_idx in range(cfg.max_steps_per_episode):
            if checkpoint_backend == "puffer_ppo" and policy_model is not None:
                action = select_policy_action(
                    model=policy_model,
                    obs=np.asarray(obs, dtype=np.float32),
                    deterministic=cfg.policy_deterministic,
                )
            else:
                action = int(rng.integers(0, N_ACTIONS))

            if action < 0 or action >= N_ACTIONS:
                action = action % N_ACTIONS

            obs, reward, terminated, truncated, info = env.step(action)

            dt = int(info.get("dt", 1))
            if dt <= 0:
                dt = 1
            t += dt
            return_total += float(reward)

            frame = frame_from_step(
                frame_index=frame_idx,
                t=t,
                dt=dt,
                action=action,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                render_state=_build_render_state(obs=obs, info=info),
                events=_derive_events(
                    info=info,
                    prev_info=prev_info,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                ),
                info=info,
                include_info=cfg.include_info,
            )
            validate_replay_frame(frame)
            frames.append(frame)
            prev_info = info

            if terminated or truncated:
                break

        if return_total > best_return or best_frames is None:
            best_return = return_total
            best_frames = frames
            best_steps = len(frames)
            best_last_info = info
            best_terminated = bool(terminated)
            best_truncated = bool(truncated)

    if best_frames is None:
        raise RuntimeError("eval replay generation failed: no frames recorded")

    replay_id = f"replay-{cfg.window_id:06d}-{uuid4().hex[:8]}"
    replays_dir = cfg.run_dir / "replays"
    replay_path = replays_dir / f"{replay_id}.jsonl.gz"
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(replay_path, mode="wt", encoding="utf-8") as handle:
        for frame in best_frames:
            handle.write(json.dumps(frame, separators=(",", ":")))
            handle.write("\n")

    replay_index_path = cfg.run_dir / "replay_index.json"
    profit = float(best_last_info.get("net_profit", 0.0))
    survival = float(best_last_info.get("survival", 0.0))

    replay_tags = _compute_replay_tags(
        replay_index_path=replay_index_path,
        run_id=cfg.run_id,
        return_total=best_return,
        profit=profit,
        survival=survival,
        profit_thresholds=cfg.milestone_profit_thresholds,
        return_thresholds=cfg.milestone_return_thresholds,
        survival_thresholds=cfg.milestone_survival_thresholds,
    )

    replay_entry: dict[str, Any] = {
        "run_id": cfg.run_id,
        "window_id": int(cfg.window_id),
        "replay_id": replay_id,
        "replay_path": _as_relative_posix(replay_path, start=cfg.run_dir),
        "checkpoint_path": _as_relative_posix(cfg.checkpoint_path, start=cfg.run_dir),
        "tags": replay_tags,
        "trainer_backend": checkpoint_backend,
        "return_total": float(best_return),
        "steps": int(best_steps),
        "profit": profit,
        "survival": survival,
        "terminated": bool(best_terminated),
        "truncated": bool(best_truncated),
        "checkpoint_env_steps_total": int(checkpoint_payload.get("env_steps_total", 0)),
        "created_at": now_iso(),
    }

    append_replay_entry(path=replay_index_path, run_id=cfg.run_id, entry=replay_entry)

    return EvalReplayResult(
        replay_id=replay_id,
        replay_path=replay_path,
        replay_path_relative=_as_relative_posix(replay_path, start=cfg.run_dir),
        replay_index_path=replay_index_path,
        replay_index_path_relative=_as_relative_posix(replay_index_path, start=cfg.run_dir),
        replay_entry=replay_entry,
    )
