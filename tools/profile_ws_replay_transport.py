from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
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
    return datetime.now(UTC).strftime("ws-profile-%Y%m%dT%H%M%SZ")


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    text = raw.strip()
    if text == "":
        return tuple()
    values: list[int] = []
    for item in text.split(","):
        part = item.strip()
        if part == "":
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"CSV integer values must be positive: {part!r}")
        values.append(value)
    return tuple(values)


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


def _stream_ws_once(
    *,
    client: TestClient,
    run_id: str,
    replay_id: str,
    limit: int,
    batch_size: int,
    max_chunk_bytes: int,
    yield_every_batches: int,
) -> dict[str, Any]:
    ws_url = (
        f"/ws/runs/{run_id}/replays/{replay_id}/frames"
        f"?offset=0&limit={int(limit)}&batch_size={int(batch_size)}"
        f"&max_chunk_bytes={int(max_chunk_bytes)}&yield_every_batches={int(yield_every_batches)}"
    )

    stream_start = perf_counter()
    frames_total = 0
    chunks_total = 0
    chunk_bytes: list[int] = []
    complete_payload: dict[str, Any] | None = None

    with client.websocket_connect(ws_url) as ws:
        while True:
            payload = ws.receive_json()
            if not isinstance(payload, dict):
                raise RuntimeError("WebSocket replay profile received non-object payload")

            message_type = str(payload.get("type", ""))
            if message_type == "error":
                status_code = int(payload.get("status_code", 500))
                detail = str(payload.get("detail", "websocket profile failed"))
                raise RuntimeError(f"Websocket replay profile failed ({status_code}): {detail}")

            if message_type == "frames":
                count = int(payload.get("count", 0))
                frames_total += count
                chunks_total += 1
                maybe_bytes = int(payload.get("chunk_bytes", 0))
                if maybe_bytes > 0:
                    chunk_bytes.append(maybe_bytes)
                continue

            if message_type == "complete":
                complete_payload = payload
                break

            raise RuntimeError(
                f"Unexpected websocket replay profile message type: {message_type!r}"
            )

    stream_seconds = perf_counter() - stream_start
    frames_per_second = float(frames_total) / stream_seconds if stream_seconds > 0.0 else 0.0

    return {
        "stream_seconds": stream_seconds,
        "stream_ms": stream_seconds * 1000.0,
        "frames_total": int(frames_total),
        "chunks_total": int(chunks_total),
        "frames_per_second": frames_per_second,
        "chunk_bytes_mean": float(statistics.fmean(chunk_bytes)) if chunk_bytes else 0.0,
        "chunk_bytes_max": int(max(chunk_bytes)) if chunk_bytes else 0,
        "complete": complete_payload or {},
    }


def _count_replay_frames(path: Path) -> int:
    count = 0
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() != "":
                count += 1
    return count


def _serialize_config(cfg: WsProfileConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["run_root"] = cfg.run_root.as_posix()
    payload["output_path"] = cfg.output_path.as_posix() if cfg.output_path is not None else None
    payload["batch_size_candidates"] = list(cfg.batch_size_candidates)
    payload["max_chunk_bytes_candidates"] = list(cfg.max_chunk_bytes_candidates)
    return payload


@dataclass(frozen=True)
class WsProfileConfig:
    run_root: Path = Path("artifacts/ws_profile/runs")
    output_path: Path | None = None
    run_id: str | None = None
    seed: int = 29

    trainer_backend: str = "random"
    trainer_total_env_steps: int = 2000
    trainer_window_env_steps: int = 500
    eval_max_steps_per_episode: int = 1024

    ws_limit: int = 5000
    min_replay_frames: int = 256
    iterations_per_config: int = 3
    batch_size_candidates: tuple[int, ...] = (64, 128, 256, 512)
    max_chunk_bytes_candidates: tuple[int, ...] = (65536, 131072, 262144, 524288)
    yield_every_batches: int = 8


def run_ws_profile(cfg: WsProfileConfig) -> dict[str, Any]:
    if cfg.iterations_per_config <= 0:
        raise ValueError("iterations_per_config must be positive")

    run_id = cfg.run_id or _default_run_id()
    run_dir = cfg.run_root / run_id
    if run_dir.exists():
        raise ValueError(f"run_id already exists under run_root: {run_id!r}")

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
        eval_include_info=True,
    )

    train_summary = run_training(train_cfg)
    latest_replay = train_summary.get("latest_replay")
    if not isinstance(latest_replay, dict):
        raise RuntimeError("latest_replay missing after training run")

    replay_id = str(latest_replay.get("replay_id", "")).strip()
    replay_path_rel = str(latest_replay.get("replay_path", "")).strip()
    if replay_id == "" or replay_path_rel == "":
        raise RuntimeError("latest_replay does not include replay_id/replay_path")

    replay_path = run_dir / replay_path_rel
    replay_frame_count = _count_replay_frames(replay_path)
    if replay_frame_count < int(cfg.min_replay_frames):
        raise RuntimeError(
            "Replay frame count "
            f"{replay_frame_count} below min_replay_frames={cfg.min_replay_frames}"
        )

    app = create_app(runs_root=cfg.run_root)
    config_reports: list[dict[str, Any]] = []

    with TestClient(app) as client:
        for batch_size in cfg.batch_size_candidates:
            for max_chunk_bytes in cfg.max_chunk_bytes_candidates:
                attempts: list[dict[str, Any]] = []
                for _ in range(cfg.iterations_per_config):
                    attempts.append(
                        _stream_ws_once(
                            client=client,
                            run_id=run_id,
                            replay_id=replay_id,
                            limit=cfg.ws_limit,
                            batch_size=int(batch_size),
                            max_chunk_bytes=int(max_chunk_bytes),
                            yield_every_batches=int(cfg.yield_every_batches),
                        )
                    )

                stream_ms = [float(item["stream_ms"]) for item in attempts]
                frames_per_second = [float(item["frames_per_second"]) for item in attempts]
                chunks_total = [int(item["chunks_total"]) for item in attempts]
                chunk_bytes_mean = [float(item["chunk_bytes_mean"]) for item in attempts]

                config_reports.append(
                    {
                        "batch_size": int(batch_size),
                        "max_chunk_bytes": int(max_chunk_bytes),
                        "yield_every_batches": int(cfg.yield_every_batches),
                        "iterations": int(cfg.iterations_per_config),
                        "stream": {
                            **_latency_stats(stream_ms),
                            "mean_frames_per_second": float(statistics.fmean(frames_per_second)),
                        },
                        "chunking": {
                            "mean_chunks_per_stream": float(statistics.fmean(chunks_total)),
                            "mean_chunk_bytes": float(statistics.fmean(chunk_bytes_mean)),
                        },
                        "frames_total_mean": float(
                            statistics.fmean([int(item["frames_total"]) for item in attempts])
                        ),
                    }
                )

    best = sorted(
        config_reports,
        key=lambda row: (
            -float(row["stream"]["mean_frames_per_second"]),
            float(row["stream"]["p95_ms"]),
            float(row["chunking"]["mean_chunks_per_stream"]),
        ),
    )[0]

    output_path = cfg.output_path
    if output_path is None:
        output_path = Path("artifacts/ws_profile") / f"{run_id}.json"

    report = {
        "generated_at": now_iso(),
        "run_id": run_id,
        "config": _serialize_config(cfg),
        "replay": {
            "replay_id": replay_id,
            "replay_path": replay_path_rel,
            "frame_count": int(replay_frame_count),
        },
        "summary": {
            "pass": True,
            "config_count": len(config_reports),
            "recommended": {
                "batch_size": int(best["batch_size"]),
                "max_chunk_bytes": int(best["max_chunk_bytes"]),
                "yield_every_batches": int(best["yield_every_batches"]),
                "mean_frames_per_second": float(best["stream"]["mean_frames_per_second"]),
                "p95_stream_ms": float(best["stream"]["p95_ms"]),
            },
        },
        "profiles": config_reports,
        "artifacts": {
            "run_root": cfg.run_root.as_posix(),
            "run_dir": run_dir.as_posix(),
            "report_path": output_path.as_posix(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> WsProfileConfig:
    parser = argparse.ArgumentParser(description="Profile websocket replay transport chunk tuning")
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/ws_profile/runs"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--trainer-backend", choices=["random", "puffer_ppo"], default="random")
    parser.add_argument("--trainer-total-env-steps", type=int, default=2000)
    parser.add_argument("--trainer-window-env-steps", type=int, default=500)
    parser.add_argument("--eval-max-steps-per-episode", type=int, default=1024)
    parser.add_argument("--ws-limit", type=int, default=5000)
    parser.add_argument("--min-replay-frames", type=int, default=256)
    parser.add_argument("--iterations-per-config", type=int, default=3)
    parser.add_argument("--batch-size-candidates", type=str, default="64,128,256,512")
    parser.add_argument(
        "--max-chunk-bytes-candidates",
        type=str,
        default="65536,131072,262144,524288",
    )
    parser.add_argument("--yield-every-batches", type=int, default=8)

    args = parser.parse_args()

    batch_size_candidates = _parse_int_csv(args.batch_size_candidates)
    max_chunk_bytes_candidates = _parse_int_csv(args.max_chunk_bytes_candidates)
    if not batch_size_candidates:
        raise ValueError("batch_size_candidates must contain at least one value")
    if not max_chunk_bytes_candidates:
        raise ValueError("max_chunk_bytes_candidates must contain at least one value")

    return WsProfileConfig(
        run_root=args.run_root,
        output_path=args.output_path,
        run_id=args.run_id,
        seed=args.seed,
        trainer_backend=args.trainer_backend,
        trainer_total_env_steps=args.trainer_total_env_steps,
        trainer_window_env_steps=args.trainer_window_env_steps,
        eval_max_steps_per_episode=args.eval_max_steps_per_episode,
        ws_limit=args.ws_limit,
        min_replay_frames=args.min_replay_frames,
        iterations_per_config=args.iterations_per_config,
        batch_size_candidates=batch_size_candidates,
        max_chunk_bytes_candidates=max_chunk_bytes_candidates,
        yield_every_batches=args.yield_every_batches,
    )


def main() -> int:
    cfg = _parse_args()
    report = run_ws_profile(cfg)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
