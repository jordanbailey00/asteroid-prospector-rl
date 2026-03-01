import json
from pathlib import Path

from tools.profile_ws_replay_transport import WsProfileConfig, run_ws_profile


def test_ws_profile_emits_recommendation_report(tmp_path: Path) -> None:
    output_path = tmp_path / "ws-profile.json"

    report = run_ws_profile(
        WsProfileConfig(
            run_root=tmp_path / "runs",
            output_path=output_path,
            run_id="ws-profile-test",
            seed=7,
            trainer_total_env_steps=80,
            trainer_window_env_steps=40,
            eval_max_steps_per_episode=32,
            ws_limit=32,
            min_replay_frames=1,
            iterations_per_config=1,
            batch_size_candidates=(8,),
            max_chunk_bytes_candidates=(4096,),
            yield_every_batches=1,
        )
    )

    assert report["summary"]["pass"] is True
    assert report["summary"]["config_count"] == 1
    assert report["summary"]["recommended"]["batch_size"] == 8
    assert report["summary"]["recommended"]["max_chunk_bytes"] == 4096

    assert len(report["profiles"]) == 1
    profile = report["profiles"][0]
    assert profile["frames_total_mean"] > 0
    assert profile["stream"]["mean_frames_per_second"] > 0.0

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "ws-profile-test"
