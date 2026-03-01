import json
from pathlib import Path

from tools.gate_throughput_floors import ThroughputFloorGateConfig, run_throughput_floor_gate


def _write_matrix_report(path: Path) -> None:
    payload = {
        "run_id": "throughput-matrix-test",
        "generated_at": "2026-03-01T00:00:00+00:00",
        "config": {"target_steps_per_sec": 100000.0},
        "mode_summaries": {
            "env_only": {
                "best_candidate_id": "env_only-env_native",
                "recommended_floor_steps_per_sec": 19520.6826,
            },
            "trainer": {
                "best_candidate_id": "trainer-backend_random",
                "recommended_floor_steps_per_sec": 204.9169,
            },
        },
        "candidates": [
            {
                "candidate_id": "env_only-env_native",
                "mode": "env_only",
                "status": "ok",
                "env_impl": "native",
                "trainer_backend": "random",
                "ppo_num_envs": 8,
                "ppo_num_workers": 4,
                "ppo_rollout_steps": 128,
                "ppo_num_minibatches": 4,
                "ppo_update_epochs": 4,
                "ppo_vector_backend": "multiprocessing",
                "ppo_env_impl": "auto",
            },
            {
                "candidate_id": "trainer-backend_random",
                "mode": "trainer",
                "status": "ok",
                "env_impl": "auto",
                "trainer_backend": "random",
                "ppo_num_envs": 8,
                "ppo_num_workers": 4,
                "ppo_rollout_steps": 128,
                "ppo_num_minibatches": 4,
                "ppo_update_epochs": 4,
                "ppo_vector_backend": "multiprocessing",
                "ppo_env_impl": "auto",
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_throughput_floor_gate_uses_matrix_recommended_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    _write_matrix_report(matrix_path)

    calls = []

    def fake_run_training_throughput_profile(cfg):
        calls.append(cfg)
        mode = cfg.modes[0]
        return {
            "summary": {"pass": True, "threshold_failures": []},
            "modes": [
                {
                    "mode": mode,
                    "steps_per_sec_stats": {
                        "min": cfg.target_steps_per_sec + 1.0,
                        "mean": cfg.target_steps_per_sec + 2.0,
                        "p50": cfg.target_steps_per_sec + 2.0,
                        "p95": cfg.target_steps_per_sec + 3.0,
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "tools.gate_throughput_floors.run_training_throughput_profile",
        fake_run_training_throughput_profile,
    )

    report = run_throughput_floor_gate(
        ThroughputFloorGateConfig(
            matrix_report_path=matrix_path,
            run_root=tmp_path / "runs",
            output_path=tmp_path / "floor-gate.json",
            run_id="floor-gate-test",
            modes=("env_only", "trainer"),
        )
    )

    assert report["summary"]["pass"] is True
    assert len(calls) == 2

    call_by_mode = {call.modes[0]: call for call in calls}
    assert call_by_mode["env_only"].target_steps_per_sec == 19520.6826
    assert call_by_mode["env_only"].env_impl == "native"
    assert call_by_mode["trainer"].target_steps_per_sec == 204.9169
    assert call_by_mode["trainer"].trainer_backend == "random"


def test_throughput_floor_gate_fails_when_mode_is_below_floor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    _write_matrix_report(matrix_path)

    def fake_run_training_throughput_profile(cfg):
        mode = cfg.modes[0]
        if mode == "trainer":
            return {
                "summary": {
                    "pass": False,
                    "threshold_failures": ["trainer mean below floor"],
                },
                "modes": [
                    {
                        "mode": mode,
                        "steps_per_sec_stats": {
                            "min": cfg.target_steps_per_sec - 10.0,
                            "mean": cfg.target_steps_per_sec - 5.0,
                            "p50": cfg.target_steps_per_sec - 5.0,
                            "p95": cfg.target_steps_per_sec - 4.0,
                        },
                    }
                ],
            }

        return {
            "summary": {"pass": True, "threshold_failures": []},
            "modes": [
                {
                    "mode": mode,
                    "steps_per_sec_stats": {
                        "min": cfg.target_steps_per_sec + 1.0,
                        "mean": cfg.target_steps_per_sec + 2.0,
                        "p50": cfg.target_steps_per_sec + 2.0,
                        "p95": cfg.target_steps_per_sec + 3.0,
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "tools.gate_throughput_floors.run_training_throughput_profile",
        fake_run_training_throughput_profile,
    )

    report = run_throughput_floor_gate(
        ThroughputFloorGateConfig(
            matrix_report_path=matrix_path,
            run_root=tmp_path / "runs",
            output_path=tmp_path / "floor-gate-fail.json",
            run_id="floor-gate-fail-test",
            modes=("env_only", "trainer"),
        )
    )

    assert report["summary"]["pass"] is False
    mode_status = report["summary"]["mode_status"]
    assert mode_status["env_only"] == "pass"
    assert mode_status["trainer"] == "fail"
