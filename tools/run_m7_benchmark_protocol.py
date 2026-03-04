from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from asteroid_prospector import N_ACTIONS, ProspectorReferenceEnv, ReferenceEnvConfig

from training.baseline_bots import BASELINE_BOT_NAMES, get_baseline_bot
from training.policy import (
    POLICY_ARCH,
    create_actor_critic,
    load_policy_state_dict,
    select_policy_action,
)
from training.train_puffer import TrainConfig, run_training

SUPPORTED_TRAINER_BACKENDS = ("random", "puffer_ppo")
SUPPORTED_WANDB_MODES = ("disabled", "offline", "online")
COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("net_profit_mean", "higher"),
    ("survival_rate", "higher"),
    ("profit_per_tick_mean", "higher"),
    ("overheat_ticks_mean", "lower"),
    ("pirate_encounters_mean", "lower"),
)

EpisodePolicy = Callable[[np.ndarray], int]
PolicyFactory = Callable[[int], EpisodePolicy]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _parse_seed_matrix_csv(raw: str) -> tuple[int, ...]:
    parts = [item.strip() for item in raw.split(",") if item.strip() != ""]
    if not parts:
        raise ValueError("seed-matrix must contain at least one integer seed.")

    deduped: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid seed value in --seed-matrix: {part!r}") from exc
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path.as_posix()}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Unable to load non-JSON checkpoint because torch is not installed."
        ) from exc

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload).__name__}")
    return payload


def _make_random_policy_factory() -> PolicyFactory:
    def factory(episode_seed: int) -> EpisodePolicy:
        rng = np.random.default_rng(int(episode_seed) + 17)

        def policy(_obs: np.ndarray) -> int:
            return int(rng.integers(0, N_ACTIONS))

        return policy

    return factory


def _make_ppo_policy_factory(
    *,
    checkpoint_path: Path,
    deterministic: bool,
) -> PolicyFactory:
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)

    trainer_backend = str(checkpoint_payload.get("trainer_backend", "")).strip()
    if trainer_backend != "puffer_ppo":
        raise ValueError(
            "Expected puffer_ppo checkpoint payload but found "
            f"trainer_backend={trainer_backend!r}."
        )

    model_state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError("puffer_ppo checkpoint is missing model_state_dict.")

    policy_arch = str(checkpoint_payload.get("policy_arch", "")).strip()
    if policy_arch != POLICY_ARCH:
        raise ValueError(
            f"Unsupported policy_arch in checkpoint: {policy_arch!r} (expected {POLICY_ARCH!r})."
        )

    raw_obs_shape = checkpoint_payload.get("obs_shape")
    if not isinstance(raw_obs_shape, list | tuple):
        raise ValueError("puffer_ppo checkpoint is missing obs_shape.")
    obs_shape = tuple(int(value) for value in raw_obs_shape)

    n_actions = int(checkpoint_payload.get("n_actions", 0))
    if n_actions <= 0:
        raise ValueError("puffer_ppo checkpoint is missing n_actions.")

    model = create_actor_critic(obs_shape=obs_shape, n_actions=n_actions, device="cpu")
    load_policy_state_dict(model, model_state_dict)
    model.eval()

    def factory(_episode_seed: int) -> EpisodePolicy:
        def policy(obs: np.ndarray) -> int:
            return select_policy_action(
                model=model,
                obs=np.asarray(obs, dtype=np.float32),
                deterministic=bool(deterministic),
            )

        return policy

    return factory


def _run_single_episode(
    *,
    policy_name: str,
    policy: EpisodePolicy,
    seed: int,
    env_time_max: float,
    max_steps_per_episode: int,
) -> dict[str, Any]:
    env = ProspectorReferenceEnv(
        config=ReferenceEnvConfig(time_max=float(env_time_max)),
        seed=int(seed),
    )
    obs, info = env.reset(seed=int(seed))

    steps = 0
    done = False
    return_total = 0.0
    invalid_actions = 0
    dt_total = 0
    last_info = dict(info)

    while not done and steps < int(max_steps_per_episode):
        action = int(policy(obs))
        if action < 0 or action >= N_ACTIONS:
            raise RuntimeError(
                f"Policy {policy_name!r} produced invalid action {action}; "
                f"expected [0, {N_ACTIONS - 1}]."
            )

        obs, reward, terminated, truncated, step_info = env.step(action)
        steps += 1
        return_total += float(reward)
        invalid_actions += int(bool(step_info.get("invalid_action", False)))
        dt_total += int(step_info.get("dt", 1))
        done = bool(terminated or truncated)
        last_info = dict(step_info)

    guard_truncated = not done

    return {
        "policy": policy_name,
        "seed": int(seed),
        "steps": int(steps),
        "dt_total": int(dt_total),
        "return_total": float(return_total),
        "terminated": bool(last_info.get("terminated", False)),
        "truncated": bool(last_info.get("truncated", False)),
        "guard_truncated": bool(guard_truncated),
        "invalid_actions": int(invalid_actions),
        "invalid_action_rate": float(invalid_actions / max(1, steps)),
        "net_profit": _as_float(last_info.get("net_profit", 0.0)),
        "profit_per_tick": _as_float(last_info.get("profit_per_tick", 0.0)),
        "survival": _as_float(last_info.get("survival", 0.0)),
        "overheat_ticks": _as_float(last_info.get("overheat_ticks", 0.0)),
        "pirate_encounters": _as_float(last_info.get("pirate_encounters", 0.0)),
        "value_lost_to_pirates": _as_float(last_info.get("value_lost_to_pirates", 0.0)),
        "scan_count": _as_float(last_info.get("scan_count", 0.0)),
        "mining_ticks": _as_float(last_info.get("mining_ticks", 0.0)),
        "time_remaining": _as_float(last_info.get("time_remaining", 0.0)),
    }


def _summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["return_total"]) for row in episodes]
    net_profit = [float(row["net_profit"]) for row in episodes]
    profit_per_tick = [float(row["profit_per_tick"]) for row in episodes]
    survival = [float(row["survival"]) for row in episodes]
    overheat = [float(row["overheat_ticks"]) for row in episodes]
    pirates = [float(row["pirate_encounters"]) for row in episodes]
    pirate_loss = [float(row["value_lost_to_pirates"]) for row in episodes]
    scan_count = [float(row["scan_count"]) for row in episodes]
    mining_ticks = [float(row["mining_ticks"]) for row in episodes]
    invalid_rates = [float(row["invalid_action_rate"]) for row in episodes]

    total_steps = int(sum(int(row["steps"]) for row in episodes))
    total_invalid = int(sum(int(row["invalid_actions"]) for row in episodes))

    return {
        "episode_count": int(len(episodes)),
        "guard_truncated_count": int(sum(1 for row in episodes if bool(row["guard_truncated"]))),
        "steps_total": total_steps,
        "dt_total": int(sum(int(row["dt_total"]) for row in episodes)),
        "return_mean": _mean(returns),
        "return_median": _median(returns),
        "net_profit_mean": _mean(net_profit),
        "net_profit_median": _median(net_profit),
        "survival_rate": _mean(survival),
        "profit_per_tick_mean": _mean(profit_per_tick),
        "overheat_ticks_mean": _mean(overheat),
        "pirate_encounters_mean": _mean(pirates),
        "value_lost_to_pirates_mean": _mean(pirate_loss),
        "scan_count_mean": _mean(scan_count),
        "mining_ticks_mean": _mean(mining_ticks),
        "invalid_action_rate_mean": _mean(invalid_rates),
        "invalid_action_rate_global": float(total_invalid / max(1, total_steps)),
    }


def _evaluate_policy_for_seed(
    *,
    policy_name: str,
    policy_factory: PolicyFactory,
    base_seed: int,
    episodes_per_seed: int,
    env_time_max: float,
    max_steps_per_episode: int,
    include_episode_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    for episode_idx in range(int(episodes_per_seed)):
        episode_seed = int(base_seed) + episode_idx
        policy = policy_factory(episode_seed)
        episodes.append(
            _run_single_episode(
                policy_name=policy_name,
                policy=policy,
                seed=episode_seed,
                env_time_max=env_time_max,
                max_steps_per_episode=max_steps_per_episode,
            )
        )

    seed_report: dict[str, Any] = {
        "seed": int(base_seed),
        "summary": _summarize_episodes(episodes),
    }
    if include_episode_rows:
        seed_report["episodes"] = episodes
    return seed_report, episodes


def _resolve_checkpoint_path(
    *,
    run_root: Path,
    train_run_id: str,
    train_summary: dict[str, Any],
) -> tuple[Path, str]:
    latest_checkpoint = train_summary.get("latest_checkpoint")
    if not isinstance(latest_checkpoint, dict):
        raise RuntimeError(
            "Training summary is missing latest_checkpoint; "
            "benchmark protocol requires a checkpoint."
        )

    checkpoint_rel = str(latest_checkpoint.get("path", "")).strip()
    if checkpoint_rel == "":
        raise RuntimeError(
            "Training summary latest_checkpoint.path is missing; cannot evaluate trained policy."
        )

    checkpoint_path = run_root / train_run_id / checkpoint_rel
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Resolved checkpoint path does not exist: "
            f"{checkpoint_path.as_posix()} (from latest_checkpoint.path={checkpoint_rel!r})"
        )
    return checkpoint_path, checkpoint_rel


def _comparison_metric_rows(
    *,
    reference_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, direction in COMPARISON_METRICS:
        reference_value = float(reference_summary.get(metric, 0.0))
        candidate_value = float(candidate_summary.get(metric, 0.0))

        if direction == "higher":
            better_or_equal = reference_value >= candidate_value
        else:
            better_or_equal = reference_value <= candidate_value

        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "reference_value": reference_value,
                "candidate_value": candidate_value,
                "delta_reference_minus_candidate": reference_value - candidate_value,
                "reference_better_or_equal": bool(better_or_equal),
            }
        )
    return rows


def _protocol_expectations(
    *,
    reference_policy: str,
    reference_summary: dict[str, Any],
    contenders: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if reference_policy != "ppo":
        return []

    checks: tuple[tuple[str, str, str, str], ...] = (
        ("ppo_vs_greedy_net_profit", "greedy_miner", "net_profit_mean", "higher"),
        ("ppo_vs_cautious_survival", "cautious_scanner", "survival_rate", "higher"),
        ("ppo_vs_market_timer_profit_per_tick", "market_timer", "profit_per_tick_mean", "higher"),
    )

    expectations: list[dict[str, Any]] = []
    for check_name, contender_name, metric, direction in checks:
        contender_summary = contenders.get(contender_name, {})
        reference_value = float(reference_summary.get(metric, 0.0))
        contender_value = float(contender_summary.get(metric, 0.0))

        if direction == "higher":
            passed = reference_value >= contender_value
        else:
            passed = reference_value <= contender_value

        expectations.append(
            {
                "name": check_name,
                "contender": contender_name,
                "metric": metric,
                "direction": direction,
                "reference_value": reference_value,
                "candidate_value": contender_value,
                "pass": bool(passed),
            }
        )
    return expectations


def _serialize_config(cfg: BenchmarkProtocolConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["run_root"] = cfg.run_root.as_posix()
    payload["output_path"] = cfg.output_path.as_posix() if cfg.output_path is not None else None
    payload["seed_matrix"] = list(cfg.seed_matrix)
    return payload


def _validate_config(cfg: BenchmarkProtocolConfig) -> None:
    if not cfg.seed_matrix:
        raise ValueError("seed_matrix must be non-empty.")
    if cfg.episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be positive.")
    if cfg.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive.")
    if cfg.env_time_max <= 0.0:
        raise ValueError("env_time_max must be positive.")
    if cfg.trainer_backend not in SUPPORTED_TRAINER_BACKENDS:
        raise ValueError(
            f"Unsupported trainer_backend {cfg.trainer_backend!r}; "
            f"supported: {', '.join(SUPPORTED_TRAINER_BACKENDS)}."
        )
    if cfg.wandb_mode not in SUPPORTED_WANDB_MODES:
        raise ValueError(
            f"Unsupported wandb_mode {cfg.wandb_mode!r}; "
            f"supported: {', '.join(SUPPORTED_WANDB_MODES)}."
        )
    if cfg.trainer_total_env_steps <= 0:
        raise ValueError("trainer_total_env_steps must be positive.")
    if cfg.trainer_window_env_steps <= 0:
        raise ValueError("trainer_window_env_steps must be positive.")
    if cfg.checkpoint_every_windows <= 0:
        raise ValueError("checkpoint_every_windows must be positive.")


@dataclass(frozen=True)
class BenchmarkProtocolConfig:
    run_root: Path = Path("artifacts/benchmarks/runs")
    output_path: Path | None = None
    run_id: str | None = None
    run_id_prefix: str = "m7-protocol"

    seed_matrix: tuple[int, ...] = (7, 19, 31)
    episodes_per_seed: int = 100
    env_time_max: float = 20000.0
    max_steps_per_episode: int = 30000
    include_episode_rows: bool = False

    market_timer_target_commodity: int = 3
    eval_policy_deterministic: bool = True
    enforce_protocol_expectations: bool = False

    trainer_backend: str = "puffer_ppo"
    trainer_total_env_steps: int = 20000
    trainer_window_env_steps: int = 2000
    checkpoint_every_windows: int = 1

    wandb_mode: str = "disabled"
    wandb_project: str = "asteroid-prospector"

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
    ppo_vector_backend: str = "multiprocessing"
    ppo_env_impl: str = "auto"


def run_m7_benchmark_protocol(cfg: BenchmarkProtocolConfig) -> dict[str, Any]:
    _validate_config(cfg)
    run_id = cfg.run_id or _default_run_id(cfg.run_id_prefix)
    trained_policy_name = "ppo" if cfg.trainer_backend == "puffer_ppo" else cfg.trainer_backend

    output_path = cfg.output_path
    if output_path is None:
        output_path = Path("artifacts/benchmarks") / f"{run_id}.json"

    contender_order = (trained_policy_name, *BASELINE_BOT_NAMES)
    contender_kind = {
        trained_policy_name: "trained_policy",
        "greedy_miner": "baseline_bot",
        "cautious_scanner": "baseline_bot",
        "market_timer": "baseline_bot",
    }
    contender_seed_reports: dict[str, list[dict[str, Any]]] = {name: [] for name in contender_order}
    contender_episodes: dict[str, list[dict[str, Any]]] = {name: [] for name in contender_order}
    training_runs: list[dict[str, Any]] = []

    for matrix_seed in cfg.seed_matrix:
        train_run_id = f"{run_id}-{trained_policy_name}-seed{int(matrix_seed)}"
        train_run_dir = cfg.run_root / train_run_id
        if train_run_dir.exists():
            raise ValueError(
                f"Training run directory already exists for seed {matrix_seed}: "
                f"{train_run_dir.as_posix()}"
            )

        train_summary = run_training(
            TrainConfig(
                run_root=cfg.run_root,
                run_id=train_run_id,
                total_env_steps=int(cfg.trainer_total_env_steps),
                window_env_steps=int(cfg.trainer_window_env_steps),
                checkpoint_every_windows=int(cfg.checkpoint_every_windows),
                seed=int(matrix_seed),
                env_time_max=float(cfg.env_time_max),
                trainer_backend=cfg.trainer_backend,
                wandb_mode=cfg.wandb_mode,
                wandb_project=cfg.wandb_project,
                eval_replays_per_window=0,
                ppo_num_envs=int(cfg.ppo_num_envs),
                ppo_num_workers=int(cfg.ppo_num_workers),
                ppo_rollout_steps=int(cfg.ppo_rollout_steps),
                ppo_num_minibatches=int(cfg.ppo_num_minibatches),
                ppo_update_epochs=int(cfg.ppo_update_epochs),
                ppo_learning_rate=float(cfg.ppo_learning_rate),
                ppo_gamma=float(cfg.ppo_gamma),
                ppo_gae_lambda=float(cfg.ppo_gae_lambda),
                ppo_clip_coef=float(cfg.ppo_clip_coef),
                ppo_ent_coef=float(cfg.ppo_ent_coef),
                ppo_vf_coef=float(cfg.ppo_vf_coef),
                ppo_max_grad_norm=float(cfg.ppo_max_grad_norm),
                ppo_vector_backend=cfg.ppo_vector_backend,
                ppo_env_impl=cfg.ppo_env_impl,
            )
        )

        checkpoint_path, checkpoint_rel = _resolve_checkpoint_path(
            run_root=cfg.run_root,
            train_run_id=train_run_id,
            train_summary=train_summary,
        )

        if cfg.trainer_backend == "puffer_ppo":
            trained_policy_factory = _make_ppo_policy_factory(
                checkpoint_path=checkpoint_path,
                deterministic=cfg.eval_policy_deterministic,
            )
        else:
            trained_policy_factory = _make_random_policy_factory()

        trained_seed_report, trained_seed_episodes = _evaluate_policy_for_seed(
            policy_name=trained_policy_name,
            policy_factory=trained_policy_factory,
            base_seed=int(matrix_seed),
            episodes_per_seed=int(cfg.episodes_per_seed),
            env_time_max=float(cfg.env_time_max),
            max_steps_per_episode=int(cfg.max_steps_per_episode),
            include_episode_rows=cfg.include_episode_rows,
        )
        contender_seed_reports[trained_policy_name].append(trained_seed_report)
        contender_episodes[trained_policy_name].extend(trained_seed_episodes)

        for bot_name in BASELINE_BOT_NAMES:

            def baseline_factory(_episode_seed: int, *, name: str = bot_name) -> EpisodePolicy:
                del _episode_seed
                return get_baseline_bot(
                    name,
                    target_commodity=int(cfg.market_timer_target_commodity),
                )

            bot_seed_report, bot_seed_episodes = _evaluate_policy_for_seed(
                policy_name=bot_name,
                policy_factory=baseline_factory,
                base_seed=int(matrix_seed),
                episodes_per_seed=int(cfg.episodes_per_seed),
                env_time_max=float(cfg.env_time_max),
                max_steps_per_episode=int(cfg.max_steps_per_episode),
                include_episode_rows=cfg.include_episode_rows,
            )
            contender_seed_reports[bot_name].append(bot_seed_report)
            contender_episodes[bot_name].extend(bot_seed_episodes)

        training_runs.append(
            {
                "seed": int(matrix_seed),
                "run_id": train_run_id,
                "status": str(train_summary.get("status", "")),
                "trainer_backend": cfg.trainer_backend,
                "env_steps_total": int(train_summary.get("env_steps_total", 0)),
                "windows_emitted": int(train_summary.get("windows_emitted", 0)),
                "checkpoints_written": int(train_summary.get("checkpoints_written", 0)),
                "latest_checkpoint_path": checkpoint_rel,
            }
        )

    contenders: list[dict[str, Any]] = []
    contender_aggregate: dict[str, dict[str, Any]] = {}
    for contender_name in contender_order:
        aggregate = _summarize_episodes(contender_episodes[contender_name])
        contender_aggregate[contender_name] = aggregate
        contender_payload: dict[str, Any] = {
            "name": contender_name,
            "kind": contender_kind[contender_name],
            "aggregate": aggregate,
            "seed_reports": contender_seed_reports[contender_name],
        }
        if cfg.include_episode_rows:
            contender_payload["episodes"] = contender_episodes[contender_name]
        contenders.append(contender_payload)

    reference_summary = contender_aggregate[trained_policy_name]
    comparison_rows: list[dict[str, Any]] = []
    for bot_name in BASELINE_BOT_NAMES:
        comparison_rows.append(
            {
                "contender": bot_name,
                "metrics": _comparison_metric_rows(
                    reference_summary=reference_summary,
                    candidate_summary=contender_aggregate[bot_name],
                ),
            }
        )

    expectations = _protocol_expectations(
        reference_policy=trained_policy_name,
        reference_summary=reference_summary,
        contenders={name: contender_aggregate[name] for name in BASELINE_BOT_NAMES},
    )
    expectations_pass = (
        all(bool(row.get("pass", False)) for row in expectations) if expectations else True
    )
    all_episodes_complete = all(
        int(contender_aggregate[name].get("guard_truncated_count", 0)) == 0
        for name in contender_order
    )
    protocol_pass = all_episodes_complete and (
        expectations_pass or not cfg.enforce_protocol_expectations
    )

    report = {
        "generated_at": now_iso(),
        "run_id": run_id,
        "config": _serialize_config(cfg),
        "training_runs": training_runs,
        "contenders": contenders,
        "comparison": {
            "reference_policy": trained_policy_name,
            "metrics": list(metric for metric, _ in COMPARISON_METRICS),
            "rows": comparison_rows,
            "expectations": expectations,
        },
        "summary": {
            "pass": bool(protocol_pass),
            "seed_count": int(len(cfg.seed_matrix)),
            "episodes_per_seed": int(cfg.episodes_per_seed),
            "episodes_per_contender": int(len(cfg.seed_matrix) * cfg.episodes_per_seed),
            "all_episodes_complete": bool(all_episodes_complete),
            "expectations_evaluated": int(len(expectations)),
            "expectations_pass": bool(expectations_pass),
            "enforce_protocol_expectations": bool(cfg.enforce_protocol_expectations),
        },
        "artifacts": {
            "run_root": cfg.run_root.as_posix(),
            "report_path": output_path.as_posix(),
            "training_run_ids": [row["run_id"] for row in training_runs],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run M7.2 benchmark protocol automation: train a seeded policy candidate "
            "per matrix seed, evaluate against baseline bots, and emit comparison report."
        )
    )
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/benchmarks/runs"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-id-prefix", type=str, default="m7-protocol")
    parser.add_argument("--seed-matrix", type=str, default="7,19,31")
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--env-time-max", type=float, default=20000.0)
    parser.add_argument("--max-steps-per-episode", type=int, default=30000)
    parser.add_argument("--include-episode-rows", action="store_true")
    parser.add_argument("--market-timer-target-commodity", type=int, default=3)
    parser.add_argument("--enforce-protocol-expectations", action="store_true")

    policy_mode = parser.add_mutually_exclusive_group()
    policy_mode.add_argument(
        "--eval-policy-deterministic",
        dest="eval_policy_deterministic",
        action="store_true",
        help="Use deterministic argmax action selection for PPO eval policy (default).",
    )
    policy_mode.add_argument(
        "--eval-policy-stochastic",
        dest="eval_policy_deterministic",
        action="store_false",
        help="Sample stochastic actions from PPO policy during evaluation.",
    )
    parser.set_defaults(eval_policy_deterministic=True)

    parser.add_argument(
        "--trainer-backend",
        choices=list(SUPPORTED_TRAINER_BACKENDS),
        default="puffer_ppo",
    )
    parser.add_argument("--trainer-total-env-steps", type=int, default=20000)
    parser.add_argument("--trainer-window-env-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every-windows", type=int, default=1)
    parser.add_argument(
        "--wandb-mode",
        choices=list(SUPPORTED_WANDB_MODES),
        default="disabled",
    )
    parser.add_argument("--wandb-project", type=str, default="asteroid-prospector")

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
    parser.add_argument(
        "--ppo-env-impl",
        choices=["reference", "native", "auto"],
        default="auto",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = BenchmarkProtocolConfig(
        run_root=args.run_root,
        output_path=args.output_path,
        run_id=args.run_id,
        run_id_prefix=args.run_id_prefix,
        seed_matrix=_parse_seed_matrix_csv(args.seed_matrix),
        episodes_per_seed=int(args.episodes_per_seed),
        env_time_max=float(args.env_time_max),
        max_steps_per_episode=int(args.max_steps_per_episode),
        include_episode_rows=bool(args.include_episode_rows),
        market_timer_target_commodity=int(args.market_timer_target_commodity),
        eval_policy_deterministic=bool(args.eval_policy_deterministic),
        enforce_protocol_expectations=bool(args.enforce_protocol_expectations),
        trainer_backend=args.trainer_backend,
        trainer_total_env_steps=int(args.trainer_total_env_steps),
        trainer_window_env_steps=int(args.trainer_window_env_steps),
        checkpoint_every_windows=int(args.checkpoint_every_windows),
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        ppo_num_envs=int(args.ppo_num_envs),
        ppo_num_workers=int(args.ppo_num_workers),
        ppo_rollout_steps=int(args.ppo_rollout_steps),
        ppo_num_minibatches=int(args.ppo_num_minibatches),
        ppo_update_epochs=int(args.ppo_update_epochs),
        ppo_learning_rate=float(args.ppo_learning_rate),
        ppo_gamma=float(args.ppo_gamma),
        ppo_gae_lambda=float(args.ppo_gae_lambda),
        ppo_clip_coef=float(args.ppo_clip_coef),
        ppo_ent_coef=float(args.ppo_ent_coef),
        ppo_vf_coef=float(args.ppo_vf_coef),
        ppo_max_grad_norm=float(args.ppo_max_grad_norm),
        ppo_vector_backend=args.ppo_vector_backend,
        ppo_env_impl=args.ppo_env_impl,
    )

    report = run_m7_benchmark_protocol(cfg)
    reference = str(report.get("comparison", {}).get("reference_policy", ""))
    print(
        "[m7-protocol] run_id={run_id} reference={reference} seeds={seeds} "
        "episodes_per_seed={episodes}".format(
            run_id=report["run_id"],
            reference=reference,
            seeds=report["summary"]["seed_count"],
            episodes=report["summary"]["episodes_per_seed"],
        )
    )

    for row in report["comparison"]["rows"]:
        contender = row["contender"]
        net_profit_metric = next(
            (metric for metric in row["metrics"] if metric["metric"] == "net_profit_mean"),
            None,
        )
        survival_metric = next(
            (metric for metric in row["metrics"] if metric["metric"] == "survival_rate"),
            None,
        )
        print(
            "[m7-protocol] vs={contender} net_profit_delta={profit_delta:.3f} "
            "survival_delta={survival_delta:.3f}".format(
                contender=contender,
                profit_delta=float(
                    (net_profit_metric or {}).get("delta_reference_minus_candidate", 0.0)
                ),
                survival_delta=float(
                    (survival_metric or {}).get("delta_reference_minus_candidate", 0.0)
                ),
            )
        )

    print(
        "[m7-protocol] expectations_pass={passed} enforced={enforced}".format(
            passed=bool(report["summary"]["expectations_pass"]),
            enforced=bool(report["summary"]["enforce_protocol_expectations"]),
        )
    )
    print("[m7-protocol] wrote report to " f"{report['artifacts']['report_path']}")
    return 0 if bool(report["summary"]["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
