import json
from pathlib import Path

import pytest

from tools.run_m7_benchmark_protocol import (
    BenchmarkProtocolConfig,
    _parse_seed_matrix_csv,
    run_m7_benchmark_protocol,
)


def test_parse_seed_matrix_csv_dedupes_and_validates() -> None:
    assert _parse_seed_matrix_csv("7, 7, 9") == (7, 9)

    with pytest.raises(ValueError, match="seed-matrix"):
        _parse_seed_matrix_csv("  ")


def test_run_m7_benchmark_protocol_emits_comparison_report(tmp_path: Path) -> None:
    output_path = tmp_path / "m7-protocol-report.json"

    cfg = BenchmarkProtocolConfig(
        run_root=tmp_path / "runs",
        output_path=output_path,
        run_id="m7-protocol-test",
        seed_matrix=(5, 11),
        episodes_per_seed=2,
        env_time_max=4000.0,
        max_steps_per_episode=12000,
        trainer_backend="random",
        trainer_total_env_steps=80,
        trainer_window_env_steps=40,
        checkpoint_every_windows=1,
        wandb_mode="disabled",
        include_episode_rows=False,
    )

    report = run_m7_benchmark_protocol(cfg)

    assert report["run_id"] == "m7-protocol-test"
    assert report["summary"]["seed_count"] == 2
    assert report["summary"]["episodes_per_seed"] == 2
    assert report["summary"]["episodes_per_contender"] == 4
    assert report["comparison"]["reference_policy"] == "random"

    contenders = {row["name"]: row for row in report["contenders"]}
    assert set(contenders) == {"random", "greedy_miner", "cautious_scanner", "market_timer"}
    assert contenders["random"]["kind"] == "trained_policy"
    assert contenders["random"]["aggregate"]["episode_count"] == 4

    rows = {row["contender"]: row for row in report["comparison"]["rows"]}
    assert set(rows) == {"greedy_miner", "cautious_scanner", "market_timer"}
    for row in rows.values():
        metric_names = {metric["metric"] for metric in row["metrics"]}
        assert metric_names == {
            "net_profit_mean",
            "survival_rate",
            "profit_per_tick_mean",
            "overheat_ticks_mean",
            "pirate_encounters_mean",
        }

    assert len(report["training_runs"]) == 2
    assert output_path.exists()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "m7-protocol-test"


def test_run_m7_benchmark_protocol_rejects_existing_train_run_dir(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    existing = run_root / "duplicate-random-seed7"
    existing.mkdir(parents=True, exist_ok=True)

    cfg = BenchmarkProtocolConfig(
        run_root=run_root,
        output_path=tmp_path / "unused.json",
        run_id="duplicate",
        seed_matrix=(7,),
        episodes_per_seed=1,
        trainer_backend="random",
        trainer_total_env_steps=40,
        trainer_window_env_steps=20,
        checkpoint_every_windows=1,
        wandb_mode="disabled",
    )

    with pytest.raises(ValueError, match="already exists"):
        run_m7_benchmark_protocol(cfg)
