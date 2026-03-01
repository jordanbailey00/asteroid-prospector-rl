import json
from pathlib import Path

from tools.stability_replay_long_run import StabilityConfig, run_stability_job


def test_replay_stability_job_emits_pass_report(tmp_path: Path) -> None:
    output_path = tmp_path / "stability-report.json"

    report = run_stability_job(
        StabilityConfig(
            run_root=tmp_path / "runs",
            output_path=output_path,
            run_id_prefix="stability-test",
            seed_start=9,
            cycles=1,
            trainer_total_env_steps=80,
            trainer_window_env_steps=40,
            eval_max_steps_per_episode=16,
            catalog_iterations=4,
            frame_iterations=8,
            index_reload_iterations=8,
            frame_limit=8,
            memory_growth_limit_mb=64.0,
        )
    )

    assert report["summary"]["cycles"] == 1
    assert report["summary"]["pass"] is True
    assert report["summary"]["drift_error_total"] == 0

    assert len(report["cycles"]) == 1
    cycle = report["cycles"][0]
    assert cycle["train"]["windows_emitted"] >= 2
    assert cycle["index"]["replay_count"] >= 2
    assert cycle["stability"]["memory"]["pass"] is True
    assert cycle["stability"]["frames"]["samples"] == 8

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["summary"]["pass"] is True
