import math
from pathlib import Path

from tools.profile_training_throughput import (
    ENV_IMPL_NATIVE,
    ENV_IMPL_REFERENCE,
    PROFILE_MODE_ENV_ONLY,
    PROFILE_MODE_TRAINER,
)
from tools.run_throughput_matrix import ThroughputMatrixConfig, run_throughput_matrix


def test_throughput_matrix_selects_best_candidate_and_floor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run_training_throughput_profile(cfg):
        mode = cfg.modes[0]
        if mode == PROFILE_MODE_ENV_ONLY:
            mean = 2400.0 if cfg.env_impl == ENV_IMPL_NATIVE else 300.0
            minimum = mean * 0.8
        else:
            mean = 500.0
            minimum = 450.0

        return {
            "summary": {"pass": True, "threshold_failures": []},
            "modes": [
                {
                    "mode": mode,
                    "steps_per_sec_stats": {
                        "min": minimum,
                        "max": mean * 1.1,
                        "mean": mean,
                        "p50": mean,
                        "p95": mean * 1.05,
                        "p99": mean * 1.08,
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "tools.run_throughput_matrix.run_training_throughput_profile",
        fake_run_training_throughput_profile,
    )

    report = run_throughput_matrix(
        ThroughputMatrixConfig(
            run_root=tmp_path / "runs",
            output_path=tmp_path / "throughput-matrix.json",
            run_id="throughput-matrix-test",
            modes=(PROFILE_MODE_ENV_ONLY, PROFILE_MODE_TRAINER),
            env_impls=(ENV_IMPL_NATIVE, ENV_IMPL_REFERENCE),
            trainer_backends=("random",),
            floor_safety_factor=0.9,
            enforce_target=False,
        )
    )

    assert report["summary"]["pass"] is True
    assert report["summary"]["total_candidates"] == 3
    assert report["summary"]["successful_candidates"] == 3

    env_summary = report["mode_summaries"][PROFILE_MODE_ENV_ONLY]
    assert env_summary["best_candidate_id"] == "env_only-env_native"
    assert env_summary["best_mean_steps_per_sec"] == 2400.0
    assert math.isclose(
        env_summary["recommended_floor_steps_per_sec"],
        2400.0 * 0.8 * 0.9,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )

    output_path = tmp_path / "throughput-matrix.json"
    assert output_path.exists()


def test_throughput_matrix_records_candidate_errors_without_crashing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run_training_throughput_profile(cfg):
        mode = cfg.modes[0]
        if mode == PROFILE_MODE_TRAINER:
            raise RuntimeError("simulated trainer failure")
        raise AssertionError("unexpected mode")

    monkeypatch.setattr(
        "tools.run_throughput_matrix.run_training_throughput_profile",
        fake_run_training_throughput_profile,
    )

    report = run_throughput_matrix(
        ThroughputMatrixConfig(
            run_root=tmp_path / "runs",
            output_path=tmp_path / "throughput-matrix-errors.json",
            run_id="throughput-matrix-errors",
            modes=(PROFILE_MODE_TRAINER,),
            trainer_backends=("random",),
            enforce_target=False,
        )
    )

    assert report["summary"]["pass"] is False
    assert report["summary"]["total_candidates"] == 1
    assert report["summary"]["error_candidates"] == 1
    assert report["summary"]["successful_candidates"] == 0

    candidate = report["candidates"][0]
    assert candidate["candidate_id"] == "trainer-backend_random"
    assert candidate["status"] == "error"
    assert "simulated trainer failure" in candidate["error"]
