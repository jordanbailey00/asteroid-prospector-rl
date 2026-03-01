from __future__ import annotations

import argparse
import gc
import gzip
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

from replay.index import load_replay_index
from replay.schema import validate_replay_frame
from server.app import create_app
from training import TrainConfig, run_training


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp_suffix() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class StabilityConfig:
    run_root: Path = Path("artifacts/stability/runs")
    output_path: Path | None = None
    run_id_prefix: str = "m7-stability"
    seed_start: int = 41
    cycles: int = 3

    trainer_backend: str = "random"
    trainer_total_env_steps: int = 1200
    trainer_window_env_steps: int = 300
    eval_max_steps_per_episode: int = 256

    catalog_iterations: int = 200
    frame_iterations: int = 800
    index_reload_iterations: int = 400
    frame_limit: int = 128

    memory_growth_limit_mb: float = 24.0


def _serialize_config(cfg: StabilityConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["run_root"] = cfg.run_root.as_posix()
    payload["output_path"] = cfg.output_path.as_posix() if cfg.output_path is not None else None
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path.as_posix()}")
    return payload


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


def _validate_replay_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Replay file missing: {path.as_posix()}")

    frame_count = 0
    first_frame: dict[str, Any] | None = None
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text == "":
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid replay frame payload in {path.as_posix()}")
            validate_replay_frame(payload)
            frame_count += 1
            if first_frame is None:
                first_frame = payload

    if frame_count <= 0 or first_frame is None:
        raise RuntimeError(f"Replay file has no frames: {path.as_posix()}")

    return {
        "frame_count": int(frame_count),
        "first_frame_index": int(first_frame.get("frame_index", -1)),
    }


def _verify_replay_index(
    *,
    run_dir: Path,
    run_id: str,
    expected_replay_count: int,
) -> dict[str, Any]:
    index_path = run_dir / "replay_index.json"
    index_payload = load_replay_index(path=index_path, run_id=run_id)
    entries_raw = index_payload.get("entries", [])
    if not isinstance(entries_raw, list):
        raise RuntimeError("Replay index entries must be a list")

    entries: list[dict[str, Any]] = []
    for entry in entries_raw:
        if not isinstance(entry, dict):
            raise RuntimeError("Replay index entry must be an object")
        entries.append(entry)

    replay_count = len(entries)
    if replay_count <= 0:
        raise RuntimeError("Replay index has no entries")
    if expected_replay_count > 0 and replay_count != expected_replay_count:
        raise RuntimeError(
            f"Replay index count mismatch: expected {expected_replay_count}, got {replay_count}"
        )

    replay_ids = [str(entry.get("replay_id", "")) for entry in entries]
    if any(replay_id == "" for replay_id in replay_ids):
        raise RuntimeError("Replay index contains empty replay_id")
    if len(set(replay_ids)) != len(replay_ids):
        raise RuntimeError("Replay index contains duplicate replay_id values")

    window_ids: list[int] = []
    frame_counts: list[int] = []
    for entry in entries:
        if str(entry.get("run_id", "")) != run_id:
            raise RuntimeError("Replay index entry run_id mismatch")

        window_ids.append(int(entry.get("window_id", -1)))
        replay_path = run_dir / str(entry.get("replay_path", ""))
        replay_info = _validate_replay_file(replay_path)
        frame_counts.append(int(replay_info["frame_count"]))

    nondecreasing_windows = all(
        window_ids[idx] >= window_ids[idx - 1] for idx in range(1, len(window_ids))
    )
    if not nondecreasing_windows:
        raise RuntimeError("Replay index window ordering regressed")

    return {
        "index_path": index_path.as_posix(),
        "replay_count": replay_count,
        "window_id_min": int(min(window_ids)),
        "window_id_max": int(max(window_ids)),
        "frame_count_min": int(min(frame_counts)),
        "frame_count_max": int(max(frame_counts)),
        "frame_count_mean": float(statistics.fmean(frame_counts)),
        "replay_ids": replay_ids,
    }


def _request_json(
    client: TestClient, path: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    response = client.get(path, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"GET {path} failed ({response.status_code}): {response.text[:200]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"GET {path} returned non-object JSON payload")
    return payload


def _run_cycle_stability_checks(
    *,
    cfg: StabilityConfig,
    run_id: str,
    replay_ids: list[str],
    expected_replay_count: int,
) -> dict[str, Any]:
    catalog_latencies_ms: list[float] = []
    frame_latencies_ms: list[float] = []
    drift_errors: list[str] = []

    app = create_app(runs_root=cfg.run_root)
    gc.collect()
    tracemalloc.start()
    try:
        start_current, _ = tracemalloc.get_traced_memory()
        max_current = start_current

        with TestClient(app) as client:
            for _ in range(max(1, int(cfg.catalog_iterations))):
                start = perf_counter()
                payload = _request_json(
                    client, f"/api/runs/{run_id}/replays", params={"limit": 5000}
                )
                catalog_latencies_ms.append((perf_counter() - start) * 1000.0)

                count = int(payload.get("count", -1))
                if count != expected_replay_count:
                    drift_errors.append(
                        "Replay catalog count drifted to "
                        f"{count} (expected {expected_replay_count})"
                    )

            if replay_ids:
                for idx in range(max(1, int(cfg.frame_iterations))):
                    replay_id = replay_ids[idx % len(replay_ids)]
                    path = f"/api/runs/{run_id}/replays/{replay_id}/frames"
                    start = perf_counter()
                    payload = _request_json(
                        client,
                        path,
                        params={"offset": 0, "limit": int(cfg.frame_limit)},
                    )
                    frame_latencies_ms.append((perf_counter() - start) * 1000.0)
                    frame_count = int(payload.get("count", 0))
                    if frame_count <= 0:
                        drift_errors.append(
                            f"Frame endpoint returned empty payload for replay_id={replay_id}"
                        )

            for idx in range(max(1, int(cfg.index_reload_iterations))):
                payload = load_replay_index(
                    path=cfg.run_root / run_id / "replay_index.json", run_id=run_id
                )
                entries = payload.get("entries", [])
                size = len(entries) if isinstance(entries, list) else -1
                if size != expected_replay_count:
                    drift_errors.append(
                        "Replay index reload count drifted to "
                        f"{size} (expected {expected_replay_count})"
                    )
                if idx % 100 == 0:
                    gc.collect()

        gc.collect()
        end_current, end_peak = tracemalloc.get_traced_memory()
        if end_current > max_current:
            max_current = end_current
    finally:
        tracemalloc.stop()

    final_growth = int(end_current - start_current)
    max_growth = int(max_current - start_current)
    peak_growth = int(end_peak - start_current)
    memory_limit_bytes = int(float(cfg.memory_growth_limit_mb) * 1024.0 * 1024.0)

    cycle_pass = len(drift_errors) == 0 and final_growth <= memory_limit_bytes

    return {
        "pass": bool(cycle_pass),
        "drift_error_count": len(drift_errors),
        "drift_errors": drift_errors,
        "catalog": {
            "samples": len(catalog_latencies_ms),
            **_latency_stats(catalog_latencies_ms),
        },
        "frames": {
            "samples": len(frame_latencies_ms),
            **_latency_stats(frame_latencies_ms),
        },
        "memory": {
            "growth_limit_mb": float(cfg.memory_growth_limit_mb),
            "final_growth_mb": float(final_growth) / (1024.0 * 1024.0),
            "max_current_growth_mb": float(max_growth) / (1024.0 * 1024.0),
            "peak_growth_mb": float(peak_growth) / (1024.0 * 1024.0),
            "pass": final_growth <= memory_limit_bytes,
        },
    }


def run_stability_job(cfg: StabilityConfig) -> dict[str, Any]:
    if cfg.cycles <= 0:
        raise ValueError("cycles must be positive")
    if cfg.trainer_total_env_steps <= 0:
        raise ValueError("trainer_total_env_steps must be positive")
    if cfg.trainer_window_env_steps <= 0:
        raise ValueError("trainer_window_env_steps must be positive")

    job_id = f"{cfg.run_id_prefix}-{_timestamp_suffix()}"
    cycle_reports: list[dict[str, Any]] = []

    for cycle_idx in range(int(cfg.cycles)):
        run_id = f"{job_id}-c{cycle_idx:02d}"
        seed = int(cfg.seed_start + cycle_idx)

        train_cfg = TrainConfig(
            run_root=cfg.run_root,
            run_id=run_id,
            total_env_steps=int(cfg.trainer_total_env_steps),
            window_env_steps=int(cfg.trainer_window_env_steps),
            checkpoint_every_windows=1,
            seed=seed,
            trainer_backend=cfg.trainer_backend,
            wandb_mode="disabled",
            eval_replays_per_window=1,
            eval_max_steps_per_episode=int(cfg.eval_max_steps_per_episode),
            eval_include_info=False,
        )

        train_start = perf_counter()
        train_summary = run_training(train_cfg)
        train_elapsed_seconds = perf_counter() - train_start

        expected_replay_count = int(train_summary.get("windows_emitted", 0))
        run_dir = cfg.run_root / run_id
        metadata = _read_json(run_dir / "run_metadata.json")

        index_metrics = _verify_replay_index(
            run_dir=run_dir,
            run_id=run_id,
            expected_replay_count=expected_replay_count,
        )

        latest_replay = metadata.get("latest_replay")
        latest_replay_id = ""
        if isinstance(latest_replay, dict):
            latest_replay_id = str(latest_replay.get("replay_id", ""))

        replay_ids = index_metrics["replay_ids"]
        if latest_replay_id and latest_replay_id not in replay_ids:
            raise RuntimeError(
                f"latest_replay replay_id missing from index for run {run_id}: {latest_replay_id}"
            )

        stability_metrics = _run_cycle_stability_checks(
            cfg=cfg,
            run_id=run_id,
            replay_ids=replay_ids,
            expected_replay_count=expected_replay_count,
        )

        cycle_reports.append(
            {
                "cycle": cycle_idx,
                "run_id": run_id,
                "seed": seed,
                "train": {
                    "elapsed_seconds": train_elapsed_seconds,
                    "env_steps_total": int(train_summary.get("env_steps_total", 0)),
                    "windows_emitted": expected_replay_count,
                },
                "index": {
                    key: value for key, value in index_metrics.items() if key != "replay_ids"
                },
                "stability": stability_metrics,
            }
        )

    cycle_passes = [bool(report["stability"]["pass"]) for report in cycle_reports]
    memory_passes = [bool(report["stability"]["memory"]["pass"]) for report in cycle_reports]
    drift_error_total = sum(
        int(report["stability"]["drift_error_count"]) for report in cycle_reports
    )

    output_path = cfg.output_path
    if output_path is None:
        output_path = Path("artifacts/stability") / f"{job_id}.json"

    report = {
        "generated_at": now_iso(),
        "job_id": job_id,
        "config": _serialize_config(cfg),
        "summary": {
            "cycles": int(cfg.cycles),
            "pass": all(cycle_passes),
            "cycles_passed": int(sum(1 for item in cycle_passes if item)),
            "memory_pass": all(memory_passes),
            "drift_error_total": int(drift_error_total),
        },
        "cycles": cycle_reports,
        "artifacts": {
            "run_root": cfg.run_root.as_posix(),
            "report_path": output_path.as_posix(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> StabilityConfig:
    parser = argparse.ArgumentParser(
        description="M7 long-run stability job for replay index consistency and leak checks"
    )
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/stability/runs"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id-prefix", type=str, default="m7-stability")
    parser.add_argument("--seed-start", type=int, default=41)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--trainer-backend", choices=["random", "puffer_ppo"], default="random")
    parser.add_argument("--trainer-total-env-steps", type=int, default=1200)
    parser.add_argument("--trainer-window-env-steps", type=int, default=300)
    parser.add_argument("--eval-max-steps-per-episode", type=int, default=256)
    parser.add_argument("--catalog-iterations", type=int, default=200)
    parser.add_argument("--frame-iterations", type=int, default=800)
    parser.add_argument("--index-reload-iterations", type=int, default=400)
    parser.add_argument("--frame-limit", type=int, default=128)
    parser.add_argument("--memory-growth-limit-mb", type=float, default=24.0)

    args = parser.parse_args()
    return StabilityConfig(
        run_root=args.run_root,
        output_path=args.output_path,
        run_id_prefix=args.run_id_prefix,
        seed_start=args.seed_start,
        cycles=args.cycles,
        trainer_backend=args.trainer_backend,
        trainer_total_env_steps=args.trainer_total_env_steps,
        trainer_window_env_steps=args.trainer_window_env_steps,
        eval_max_steps_per_episode=args.eval_max_steps_per_episode,
        catalog_iterations=args.catalog_iterations,
        frame_iterations=args.frame_iterations,
        index_reload_iterations=args.index_reload_iterations,
        frame_limit=args.frame_limit,
        memory_growth_limit_mb=args.memory_growth_limit_mb,
    )


def main() -> int:
    cfg = _parse_args()
    report = run_stability_job(cfg)
    print(json.dumps(report, indent=2))
    return 0 if bool(report["summary"]["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
