import json
from pathlib import Path
from typing import Any

from tools.log_m7_benchmark_wandb import M7BenchmarkWandbLogConfig, log_m7_benchmark_to_wandb


class _FakeBenchmarkLogger:
    last_instance: "_FakeBenchmarkLogger | None" = None

    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] = {}
        self.logged_payload: dict[str, Any] = {}
        self.logged_step: int | None = None
        self.report_call: dict[str, Any] = {}
        self.finished_summary: dict[str, Any] = {}
        self.run_url = "https://example.test/wandb/m7"

    @classmethod
    def create(cls, **kwargs: Any) -> "_FakeBenchmarkLogger":
        instance = cls()
        instance.create_kwargs = dict(kwargs)
        cls.last_instance = instance
        return instance

    def log_metrics(self, payload: dict[str, Any], *, step: int = 0) -> None:
        self.logged_payload = dict(payload)
        self.logged_step = int(step)

    def log_benchmark_report(
        self,
        *,
        report_path: Path,
        run_id: str,
        report: dict[str, Any],
        lineage_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        self.report_call = {
            "report_path": report_path,
            "run_id": run_id,
            "report": report,
            "lineage_paths": list(lineage_paths or []),
        }
        return {
            "artifact_name": f"benchmark-{run_id}",
            "artifact_aliases": ["latest", run_id, "job-eval"],
            "lineage_file_count": len(lineage_paths or []),
        }

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        self.finished_summary = dict(summary or {})


def _write_report(path: Path, *, run_root: Path) -> None:
    payload = {
        "generated_at": "2026-03-04T00:00:00Z",
        "run_id": "m7-protocol-test",
        "config": {
            "run_root": run_root.as_posix(),
            "seed_matrix": [7, 11],
            "episodes_per_seed": 2,
        },
        "training_runs": [
            {
                "seed": 7,
                "run_id": "m7-protocol-test-ppo-seed7",
                "latest_checkpoint_path": "checkpoints/ckpt_000001.pt",
            },
            {
                "seed": 11,
                "run_id": "m7-protocol-test-ppo-seed11",
                "latest_checkpoint_path": "checkpoints/ckpt_000001.pt",
            },
        ],
        "contenders": [
            {
                "name": "ppo",
                "aggregate": {
                    "net_profit_mean": 10.0,
                    "survival_rate": 1.0,
                    "profit_per_tick_mean": 1.2,
                    "overheat_ticks_mean": 2.0,
                    "pirate_encounters_mean": 0.3,
                },
            }
        ],
        "comparison": {
            "reference_policy": "ppo",
            "rows": [
                {
                    "contender": "greedy_miner",
                    "metrics": [
                        {
                            "metric": "net_profit_mean",
                            "delta_reference_minus_candidate": 2.5,
                            "reference_better_or_equal": True,
                        }
                    ],
                }
            ],
            "expectations": [
                {
                    "name": "ppo_vs_greedy_net_profit",
                    "pass": True,
                }
            ],
        },
        "summary": {
            "pass": True,
            "seed_count": 2,
            "episodes_per_seed": 2,
            "episodes_per_contender": 4,
            "expectations_pass": True,
        },
        "artifacts": {
            "training_run_ids": [
                "m7-protocol-test-ppo-seed7",
                "m7-protocol-test-ppo-seed11",
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_log_m7_benchmark_to_wandb_updates_report_and_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "runs"
    ckpt_a = run_root / "m7-protocol-test-ppo-seed7" / "checkpoints" / "ckpt_000001.pt"
    ckpt_b = run_root / "m7-protocol-test-ppo-seed11" / "checkpoints" / "ckpt_000001.pt"
    ckpt_a.parent.mkdir(parents=True, exist_ok=True)
    ckpt_b.parent.mkdir(parents=True, exist_ok=True)
    ckpt_a.write_text("a", encoding="utf-8")
    ckpt_b.write_text("b", encoding="utf-8")

    report_path = tmp_path / "bench-report.json"
    _write_report(report_path, run_root=run_root)

    monkeypatch.setattr(
        "tools.log_m7_benchmark_wandb.WandbBenchmarkLogger",
        _FakeBenchmarkLogger,
    )

    cfg = M7BenchmarkWandbLogConfig(
        report_path=report_path,
        wandb_mode="offline",
        wandb_project="bench-proj",
        wandb_entity="team-x",
        wandb_job_type="eval",
        wandb_tags=("m7", "benchmark", "eval"),
    )

    report = log_m7_benchmark_to_wandb(cfg)

    fake = _FakeBenchmarkLogger.last_instance
    assert fake is not None
    assert fake.create_kwargs["project"] == "bench-proj"
    assert fake.create_kwargs["entity"] == "team-x"
    assert fake.create_kwargs["job_type"] == "eval"
    assert fake.logged_step == 0
    assert "benchmark/pass" in fake.logged_payload

    lineage_paths = fake.report_call["lineage_paths"]
    assert len(lineage_paths) == 2
    assert ckpt_a in lineage_paths
    assert ckpt_b in lineage_paths

    wandb_block = report["wandb_benchmark"]
    assert wandb_block["enabled"] is True
    assert wandb_block["mode"] == "offline"
    assert wandb_block["project"] == "bench-proj"
    assert wandb_block["job_type"] == "eval"
    assert wandb_block["artifact_name"] == "benchmark-m7-protocol-test"
    assert wandb_block["lineage_file_count"] == 2

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["wandb_benchmark"]["enabled"] is True


def test_log_m7_benchmark_to_wandb_disabled_mode_noop(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    report_path = tmp_path / "bench-report-disabled.json"
    _write_report(report_path, run_root=run_root)

    cfg = M7BenchmarkWandbLogConfig(
        report_path=report_path,
        wandb_mode="disabled",
    )

    report = log_m7_benchmark_to_wandb(cfg)
    assert report["wandb_benchmark"]["enabled"] is False
    assert report["wandb_benchmark"]["mode"] == "disabled"

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["wandb_benchmark"]["enabled"] is False
