import json
from pathlib import Path

from tools.profile_training_throughput import (
    ENV_IMPL_REFERENCE,
    PROFILE_MODE_ENV_ONLY,
    PROFILE_MODE_TRAINER,
    ThroughputProfileConfig,
    run_training_throughput_profile,
)


def test_throughput_profile_env_only_emits_report(tmp_path: Path) -> None:
    output_path = tmp_path / "throughput-env.json"

    report = run_training_throughput_profile(
        ThroughputProfileConfig(
            run_root=tmp_path / "runs",
            output_path=output_path,
            run_id="throughput-env",
            seed=5,
            modes=(PROFILE_MODE_ENV_ONLY,),
            env_impl=ENV_IMPL_REFERENCE,
            env_duration_seconds=0.05,
            env_repeats=1,
            target_steps_per_sec=1.0,
            enforce_target=True,
        )
    )

    assert report["summary"]["pass"] is True
    assert len(report["modes"]) == 1

    mode = report["modes"][0]
    assert mode["mode"] == PROFILE_MODE_ENV_ONLY
    assert mode["samples"] == 1
    assert mode["steps_per_sec_stats"]["mean"] > 0.0

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "throughput-env"


def test_throughput_profile_threshold_failure_is_reported(tmp_path: Path) -> None:
    output_path = tmp_path / "throughput-trainer-fail.json"

    report = run_training_throughput_profile(
        ThroughputProfileConfig(
            run_root=tmp_path / "runs",
            output_path=output_path,
            run_id="throughput-trainer-fail",
            seed=9,
            modes=(PROFILE_MODE_TRAINER,),
            trainer_total_env_steps=40,
            trainer_window_env_steps=20,
            trainer_repeats=1,
            target_steps_per_sec=1.0e9,
            enforce_target=True,
        )
    )

    assert report["summary"]["pass"] is False
    failures = report["summary"]["threshold_failures"]
    assert any("trainer mean steps/sec" in message for message in failures)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "throughput-trainer-fail"
