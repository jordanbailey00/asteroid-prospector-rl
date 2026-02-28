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
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_replay_tags(
    *,
    replay_index_path: Path,
    run_id: str,
    return_total: float,
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

    return tags


def run_eval_and_record_replay(cfg: EvalReplayConfig) -> EvalReplayResult:
    if cfg.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if cfg.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive")

    from asteroid_prospector import N_ACTIONS, ProspectorReferenceEnv, ReferenceEnvConfig

    checkpoint_payload = _load_checkpoint_payload(cfg.checkpoint_path)

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
            action = int(rng.integers(0, N_ACTIONS))
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
    replay_tags = _compute_replay_tags(
        replay_index_path=replay_index_path,
        run_id=cfg.run_id,
        return_total=best_return,
    )

    replay_entry: dict[str, Any] = {
        "run_id": cfg.run_id,
        "window_id": int(cfg.window_id),
        "replay_id": replay_id,
        "replay_path": _as_relative_posix(replay_path, start=cfg.run_dir),
        "checkpoint_path": _as_relative_posix(cfg.checkpoint_path, start=cfg.run_dir),
        "tags": replay_tags,
        "trainer_backend": cfg.trainer_backend,
        "return_total": float(best_return),
        "steps": int(best_steps),
        "profit": float(best_last_info.get("net_profit", 0.0)),
        "survival": float(best_last_info.get("survival", 0.0)),
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
