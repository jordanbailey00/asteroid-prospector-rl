from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from server.app import create_app
from training import TrainConfig, run_training


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("m7-%Y%m%dT%H%M%SZ")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))

    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_stats(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    return {
        "min_ms": float(min(values_ms)),
        "max_ms": float(max(values_ms)),
        "mean_ms": float(statistics.fmean(values_ms)),
        "p50_ms": _percentile(values_ms, 0.50),
        "p95_ms": _percentile(values_ms, 0.95),
        "p99_ms": _percentile(values_ms, 0.99),
    }


def _request_frames(
    *,
    client: TestClient,
    run_id: str,
    replay_id: str,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    response = client.get(
        f"/api/runs/{run_id}/replays/{replay_id}/frames",
        params={"offset": int(offset), "limit": int(limit)},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Replay frame request failed ({response.status_code}): {response.text[:200]}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Replay frame response must be a JSON object.")
    return payload


def _bench_replay_latency(
    *,
    client: TestClient,
    run_id: str,
    replay_id: str,
    iterations: int,
    warmup_iterations: int,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    warmup = max(0, int(warmup_iterations))
    total = max(1, int(iterations))
    frame_counts: list[int] = []
    durations_ms: list[float] = []

    for _ in range(warmup):
        _request_frames(
            client=client, run_id=run_id, replay_id=replay_id, offset=offset, limit=limit
        )

    for _ in range(total):
        start = perf_counter()
        payload = _request_frames(
            client=client,
            run_id=run_id,
            replay_id=replay_id,
            offset=offset,
            limit=limit,
        )
        elapsed_ms = (perf_counter() - start) * 1000.0
        durations_ms.append(elapsed_ms)
        frame_counts.append(int(payload.get("count", 0)))

    count_mean = float(statistics.fmean(frame_counts)) if frame_counts else 0.0
    return {
        "samples": total,
        "warmup_samples": warmup,
        "frame_count_mean": count_mean,
        **_latency_stats(durations_ms),
    }


def _bench_memory_soak(
    *,
    client: TestClient,
    run_id: str,
    replay_id: str,
    iterations: int,
    offset: int,
    limit: int,
    growth_limit_mb: float,
) -> dict[str, Any]:
    total = max(1, int(iterations))
    growth_limit_bytes = int(float(growth_limit_mb) * 1024.0 * 1024.0)
    max_current_bytes = 0

    gc.collect()
    tracemalloc.start()
    try:
        start_current, _start_peak = tracemalloc.get_traced_memory()
        max_current_bytes = start_current
        for idx in range(total):
            _request_frames(
                client=client,
                run_id=run_id,
                replay_id=replay_id,
                offset=offset,
                limit=limit,
            )
            current, _peak = tracemalloc.get_traced_memory()
            if current > max_current_bytes:
                max_current_bytes = current
            if idx % 100 == 0:
                gc.collect()
        gc.collect()
        end_current, end_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    final_growth = int(end_current - start_current)
    max_growth = int(max_current_bytes - start_current)
    peak_growth = int(end_peak - start_current)

    return {
        "iterations": total,
        "growth_limit_mb": float(growth_limit_mb),
        "final_growth_mb": float(final_growth) / (1024.0 * 1024.0),
        "max_current_growth_mb": float(max_growth) / (1024.0 * 1024.0),
        "peak_growth_mb": float(peak_growth) / (1024.0 * 1024.0),
        "pass": final_growth <= growth_limit_bytes,
    }


def _serialize_config(cfg: BenchmarkConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["run_root"] = cfg.run_root.as_posix()
    payload["output_path"] = cfg.output_path.as_posix() if cfg.output_path is not None else None
    return payload


@dataclass(frozen=True)
class BenchmarkConfig:
    run_root: Path = Path("artifacts/benchmarks/runs")
    output_path: Path | None = None
    run_id: str | None = None
    seed: int = 7

    trainer_backend: str = "random"
    trainer_total_env_steps: int = 2000
    trainer_window_env_steps: int = 500
    eval_max_steps_per_episode: int = 256

    replay_offset: int = 0
    replay_limit: int = 128
    replay_latency_iterations: int = 30
    replay_latency_warmup_iterations: int = 3

    memory_soak_iterations: int = 400
    memory_growth_limit_mb: float = 16.0


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    run_id = cfg.run_id or _default_run_id()
    run_dir = cfg.run_root / run_id
    if run_dir.exists():
        raise ValueError(
            f"Benchmark run directory already exists for run_id {run_id!r}: {run_dir.as_posix()}"
        )

    train_cfg = TrainConfig(
        run_root=cfg.run_root,
        run_id=run_id,
        total_env_steps=int(cfg.trainer_total_env_steps),
        window_env_steps=int(cfg.trainer_window_env_steps),
        checkpoint_every_windows=1,
        seed=int(cfg.seed),
        trainer_backend=cfg.trainer_backend,
        wandb_mode="disabled",
        eval_replays_per_window=1,
        eval_max_steps_per_episode=int(cfg.eval_max_steps_per_episode),
        eval_include_info=False,
    )

    train_start = perf_counter()
    train_summary = run_training(train_cfg)
    trainer_elapsed_seconds = perf_counter() - train_start
    trainer_env_steps = int(train_summary.get("env_steps_total", 0))
    trainer_steps_per_sec = (
        float(trainer_env_steps) / trainer_elapsed_seconds if trainer_elapsed_seconds > 0.0 else 0.0
    )

    latest_replay = train_summary.get("latest_replay")
    if not isinstance(latest_replay, dict):
        raise RuntimeError(
            "Benchmark requires eval replay generation but latest_replay is missing."
        )
    replay_id = str(latest_replay.get("replay_id", "")).strip()
    if replay_id == "":
        raise RuntimeError("Benchmark requires a valid replay_id from latest_replay metadata.")

    app = create_app(runs_root=cfg.run_root)
    with TestClient(app) as client:
        latency_metrics = _bench_replay_latency(
            client=client,
            run_id=run_id,
            replay_id=replay_id,
            iterations=cfg.replay_latency_iterations,
            warmup_iterations=cfg.replay_latency_warmup_iterations,
            offset=cfg.replay_offset,
            limit=cfg.replay_limit,
        )
        memory_metrics = _bench_memory_soak(
            client=client,
            run_id=run_id,
            replay_id=replay_id,
            iterations=cfg.memory_soak_iterations,
            offset=cfg.replay_offset,
            limit=cfg.replay_limit,
            growth_limit_mb=cfg.memory_growth_limit_mb,
        )

    output_path = cfg.output_path
    if output_path is None:
        output_path = Path("artifacts/benchmarks") / f"{run_id}.json"

    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "run_id": run_id,
        "config": _serialize_config(cfg),
        "trainer": {
            "backend": cfg.trainer_backend,
            "elapsed_seconds": trainer_elapsed_seconds,
            "env_steps_total": trainer_env_steps,
            "steps_per_sec": trainer_steps_per_sec,
            "windows_emitted": int(train_summary.get("windows_emitted", 0)),
            "checkpoints_written": int(train_summary.get("checkpoints_written", 0)),
        },
        "replay": {
            "replay_id": replay_id,
            "estimated_steps": int(latest_replay.get("steps", 0)),
            "offset": int(cfg.replay_offset),
            "limit": int(cfg.replay_limit),
            "frames_endpoint": f"/api/runs/{run_id}/replays/{replay_id}/frames",
        },
        "replay_api_latency": latency_metrics,
        "memory_soak": memory_metrics,
        "artifacts": {
            "run_root": cfg.run_root.as_posix(),
            "run_dir": run_dir.as_posix(),
            "report_path": output_path.as_posix(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="M7 benchmark harness")
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/benchmarks/runs"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trainer-backend", choices=["random", "puffer_ppo"], default="random")
    parser.add_argument("--trainer-total-env-steps", type=int, default=2000)
    parser.add_argument("--trainer-window-env-steps", type=int, default=500)
    parser.add_argument("--eval-max-steps-per-episode", type=int, default=256)
    parser.add_argument("--replay-offset", type=int, default=0)
    parser.add_argument("--replay-limit", type=int, default=128)
    parser.add_argument("--replay-latency-iterations", type=int, default=30)
    parser.add_argument("--replay-latency-warmup-iterations", type=int, default=3)
    parser.add_argument("--memory-soak-iterations", type=int, default=400)
    parser.add_argument("--memory-growth-limit-mb", type=float, default=16.0)

    args = parser.parse_args()
    return BenchmarkConfig(
        run_root=args.run_root,
        output_path=args.output_path,
        run_id=args.run_id,
        seed=args.seed,
        trainer_backend=args.trainer_backend,
        trainer_total_env_steps=args.trainer_total_env_steps,
        trainer_window_env_steps=args.trainer_window_env_steps,
        eval_max_steps_per_episode=args.eval_max_steps_per_episode,
        replay_offset=args.replay_offset,
        replay_limit=args.replay_limit,
        replay_latency_iterations=args.replay_latency_iterations,
        replay_latency_warmup_iterations=args.replay_latency_warmup_iterations,
        memory_soak_iterations=args.memory_soak_iterations,
        memory_growth_limit_mb=args.memory_growth_limit_mb,
    )


def main() -> int:
    cfg = _parse_args()
    report = run_benchmark(cfg)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
