import json
from pathlib import Path

import pytest

from tools.run_baseline_bots import BaselineRunConfig, run_baseline_bots


def _cfg(*, bot_names: tuple[str, ...], output_path: Path | None = None) -> BaselineRunConfig:
    return BaselineRunConfig(
        bot_names=bot_names,
        episodes=2,
        base_seed=11,
        env_time_max=600.0,
        max_steps_per_episode=1200,
        market_timer_target_commodity=3,
        run_id="m7-baseline-test",
        output_path=output_path,
    )


def test_run_baseline_bots_emits_report_and_artifact(tmp_path: Path) -> None:
    report_path = tmp_path / "baseline-report.json"

    report = run_baseline_bots(_cfg(bot_names=("greedy_miner",), output_path=report_path))

    assert report["run_id"] == "m7-baseline-test"
    assert report["summary"]["bot_count"] == 1
    assert report["summary"]["episodes_per_bot"] == 2

    bot_row = report["bots"][0]
    assert bot_row["bot"] == "greedy_miner"
    assert bot_row["summary"]["episode_count"] == 2
    assert len(bot_row["episodes"]) == 2

    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "m7-baseline-test"


def test_run_baseline_bots_is_reproducible_for_fixed_config() -> None:
    cfg = _cfg(bot_names=("cautious_scanner", "market_timer"), output_path=None)

    report_a = run_baseline_bots(cfg)
    report_b = run_baseline_bots(cfg)

    assert report_a["bots"] == report_b["bots"]
    assert report_a["summary"] == report_b["summary"]


def test_run_baseline_bots_rejects_invalid_bot_name() -> None:
    with pytest.raises(ValueError):
        run_baseline_bots(_cfg(bot_names=("unknown",), output_path=None))
