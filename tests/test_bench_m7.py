import json
from pathlib import Path

from tools.bench_m7 import BenchmarkConfig, run_benchmark


def test_m7_benchmark_harness_emits_report(tmp_path: Path) -> None:
    report_path = tmp_path / "bench-report.json"

    report = run_benchmark(
        BenchmarkConfig(
            run_root=tmp_path / "runs",
            output_path=report_path,
            run_id="m7-test",
            seed=5,
            trainer_total_env_steps=80,
            trainer_window_env_steps=40,
            eval_max_steps_per_episode=16,
            replay_limit=8,
            replay_latency_iterations=4,
            replay_latency_warmup_iterations=1,
            memory_soak_iterations=8,
            memory_growth_limit_mb=64.0,
        )
    )

    assert report["run_id"] == "m7-test"
    assert report["trainer"]["env_steps_total"] >= 80
    assert report["trainer"]["steps_per_sec"] > 0.0

    assert report["replay"]["replay_id"]
    assert report["replay_api_latency"]["samples"] == 4
    assert report["replay_api_latency"]["mean_ms"] >= 0.0

    assert report["memory_soak"]["iterations"] == 8
    assert report["memory_soak"]["pass"] is True
    assert report["summary"]["pass"] is True

    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "m7-test"


def test_m7_benchmark_threshold_failure_is_reported(tmp_path: Path) -> None:
    report = run_benchmark(
        BenchmarkConfig(
            run_root=tmp_path / "runs",
            output_path=tmp_path / "bench-threshold-fail.json",
            run_id="m7-threshold-fail",
            seed=3,
            trainer_total_env_steps=40,
            trainer_window_env_steps=20,
            eval_max_steps_per_episode=10,
            replay_limit=8,
            replay_latency_iterations=2,
            replay_latency_warmup_iterations=1,
            memory_soak_iterations=4,
            memory_growth_limit_mb=64.0,
            min_trainer_steps_per_sec=1e9,
        )
    )

    assert report["summary"]["pass"] is False
    failures = report["summary"]["threshold_failures"]
    assert any("trainer steps/sec" in message for message in failures)
