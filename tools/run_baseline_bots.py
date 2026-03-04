from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from asteroid_prospector import N_ACTIONS, ProspectorReferenceEnv, ReferenceEnvConfig

from training.baseline_bots import BASELINE_BOT_NAMES, get_baseline_bot


@dataclass(frozen=True)
class BaselineRunConfig:
    bot_names: tuple[str, ...]
    episodes: int
    base_seed: int
    env_time_max: float
    max_steps_per_episode: int
    market_timer_target_commodity: int
    run_id: str | None = None
    output_path: Path | None = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("m7-baselines-%Y%m%dT%H%M%SZ")


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _config_payload(cfg: BaselineRunConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["bot_names"] = list(cfg.bot_names)
    payload["output_path"] = cfg.output_path.as_posix() if cfg.output_path is not None else None
    return payload


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _run_single_episode(
    *,
    bot_name: str,
    seed: int,
    env_time_max: float,
    max_steps_per_episode: int,
    market_timer_target_commodity: int,
) -> dict[str, Any]:
    policy = get_baseline_bot(bot_name, target_commodity=market_timer_target_commodity)

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
                f"Baseline bot {bot_name!r} produced invalid action {action}; "
                f"expected [0, {N_ACTIONS - 1}]"
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
        "bot": bot_name,
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


def run_baseline_bots(cfg: BaselineRunConfig) -> dict[str, Any]:
    if int(cfg.episodes) < 1:
        raise ValueError("episodes must be >= 1")
    if int(cfg.max_steps_per_episode) < 1:
        raise ValueError("max_steps_per_episode must be >= 1")

    run_id = cfg.run_id or _default_run_id()
    bot_reports: list[dict[str, Any]] = []

    for bot_name in cfg.bot_names:
        episodes: list[dict[str, Any]] = []
        for episode_idx in range(int(cfg.episodes)):
            seed = int(cfg.base_seed) + episode_idx
            episodes.append(
                _run_single_episode(
                    bot_name=bot_name,
                    seed=seed,
                    env_time_max=float(cfg.env_time_max),
                    max_steps_per_episode=int(cfg.max_steps_per_episode),
                    market_timer_target_commodity=int(cfg.market_timer_target_commodity),
                )
            )

        bot_reports.append(
            {
                "bot": bot_name,
                "episodes": episodes,
                "summary": _summarize_episodes(episodes),
            }
        )

    report = {
        "generated_at": now_iso(),
        "run_id": run_id,
        "config": _config_payload(cfg),
        "bots": bot_reports,
        "summary": {
            "bot_count": int(len(bot_reports)),
            "episodes_per_bot": int(cfg.episodes),
            "all_episodes_complete": all(
                int(row["summary"]["guard_truncated_count"]) == 0 for row in bot_reports
            ),
        },
    }

    output_path = cfg.output_path
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _resolve_bot_names(raw: list[str]) -> tuple[str, ...]:
    normalized = [item.strip().lower() for item in raw if item.strip() != ""]
    if not normalized:
        return BASELINE_BOT_NAMES
    if "all" in normalized:
        return BASELINE_BOT_NAMES

    unknown = sorted({item for item in normalized if item not in BASELINE_BOT_NAMES})
    if unknown:
        supported = ", ".join(BASELINE_BOT_NAMES)
        raise ValueError(
            f"Unsupported --bot value(s): {', '.join(unknown)}. " f"Supported: {supported}, all"
        )

    deduped: list[str] = []
    for item in normalized:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible baseline bots (greedy_miner, cautious_scanner, market_timer) "
            "for M7.1 reporting."
        )
    )
    parser.add_argument(
        "--bot",
        action="append",
        default=[],
        help=(
            "Bot name to run (repeatable). Supported: "
            "greedy_miner, cautious_scanner, market_timer, all"
        ),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--env-time-max", type=float, default=20000.0)
    parser.add_argument("--max-steps-per-episode", type=int, default=30000)
    parser.add_argument("--market-timer-target-commodity", type=int, default=3)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    bot_names = _resolve_bot_names(list(args.bot))

    cfg = BaselineRunConfig(
        bot_names=bot_names,
        episodes=int(args.episodes),
        base_seed=int(args.base_seed),
        env_time_max=float(args.env_time_max),
        max_steps_per_episode=int(args.max_steps_per_episode),
        market_timer_target_commodity=int(args.market_timer_target_commodity),
        run_id=args.run_id,
        output_path=args.output_path,
    )

    report = run_baseline_bots(cfg)

    print(f"[m7-baselines] run_id={report['run_id']} bots={len(report['bots'])}")
    for bot_row in report["bots"]:
        summary = bot_row["summary"]
        print(
            "[m7-baselines] bot={bot} episodes={episodes} net_profit_mean={profit:.2f} "
            "survival_rate={survival:.3f} pirate_mean={pirates:.3f}".format(
                bot=bot_row["bot"],
                episodes=summary["episode_count"],
                profit=float(summary["net_profit_mean"]),
                survival=float(summary["survival_rate"]),
                pirates=float(summary["pirate_encounters_mean"]),
            )
        )

    if cfg.output_path is not None:
        print(f"[m7-baselines] wrote report to {cfg.output_path.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
