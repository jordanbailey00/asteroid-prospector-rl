import numpy as np

from training.windowing import INFO_METRIC_KEYS, WindowMetricsAggregator


def _info(dt: int, *, invalid: bool = False) -> dict[str, float | int | bool]:
    return {
        "dt": dt,
        "invalid_action": invalid,
        "credits": 10.0,
        "net_profit": 5.0,
        "profit_per_tick": 1.0,
        "survival": 1.0,
        "overheat_ticks": 0.0,
        "pirate_encounters": 0.0,
        "value_lost_to_pirates": 0.0,
        "fuel_used": 2.0,
        "hull_damage": 0.5,
        "tool_wear": 0.5,
        "scan_count": 1.0,
        "mining_ticks": 1.0,
        "cargo_utilization_avg": 0.2,
    }


def test_window_rollover_with_macro_dt() -> None:
    agg = WindowMetricsAggregator(run_id="run-a", window_env_steps=10)

    assert agg.record_step(reward=1.0, info=_info(3), terminated=False, truncated=False) == []
    assert agg.record_step(reward=2.0, info=_info(4), terminated=False, truncated=False) == []

    emitted = agg.record_step(reward=3.0, info=_info(5), terminated=False, truncated=False)
    assert len(emitted) == 1

    first = emitted[0]
    assert first.window_id == 0
    assert first.window_complete is True
    assert first.env_steps_in_window == 10
    assert first.env_steps_end == 10
    assert first.env_steps_total == 10

    assert agg.env_steps_total == 12
    assert agg.current_window_id == 1

    partial = agg.flush_partial()
    assert partial is not None
    assert partial.window_id == 1
    assert partial.window_complete is False
    assert partial.env_steps_in_window == 2
    assert partial.env_steps_start == 10
    assert partial.env_steps_end == 12


def test_episode_completion_assigned_to_window_with_step_end() -> None:
    agg = WindowMetricsAggregator(run_id="run-b", window_env_steps=10)

    assert agg.record_step(reward=1.0, info=_info(9), terminated=False, truncated=False) == []

    emitted = agg.record_step(reward=2.0, info=_info(3), terminated=True, truncated=False)
    assert len(emitted) == 1

    first = emitted[0]
    assert first.window_id == 0
    assert first.episodes_completed == 0

    partial = agg.flush_partial()
    assert partial is not None
    assert partial.window_id == 1
    assert partial.episodes_completed == 1
    assert partial.terminated_episodes == 1
    assert partial.return_mean == 3.0


def test_invalid_action_rate_is_step_weighted() -> None:
    agg = WindowMetricsAggregator(run_id="run-c", window_env_steps=6)

    assert (
        agg.record_step(reward=0.0, info=_info(2, invalid=True), terminated=False, truncated=False)
        == []
    )
    emitted = agg.record_step(
        reward=0.0,
        info=_info(4, invalid=False),
        terminated=False,
        truncated=False,
    )

    assert len(emitted) == 1
    record = emitted[0]
    assert record.invalid_action_rate == 2.0 / 6.0


def test_record_step_batch_matches_scalar_path() -> None:
    infos_rows = [
        _info(3, invalid=False),
        _info(2, invalid=True),
        _info(5, invalid=False),
    ]
    rewards = np.array([1.25, -0.5, 0.75], dtype=np.float32)
    terminated = np.array([False, False, True], dtype=bool)
    truncated = np.array([False, False, False], dtype=bool)

    scalar = WindowMetricsAggregator(run_id="same", window_env_steps=8)
    scalar_emitted = []
    for i in range(3):
        scalar_emitted.extend(
            scalar.record_step(
                reward=float(rewards[i]),
                info=infos_rows[i],
                terminated=bool(terminated[i]),
                truncated=bool(truncated[i]),
            )
        )

    batched = WindowMetricsAggregator(run_id="same", window_env_steps=8)
    batched_emitted = batched.record_step_batch(
        rewards=rewards,
        infos=infos_rows,
        terminated=terminated,
        truncated=truncated,
    )

    assert len(batched_emitted) == len(scalar_emitted)
    for i, scalar_record in enumerate(scalar_emitted):
        assert batched_emitted[i].to_dict() == scalar_record.to_dict()

    scalar_partial = scalar.flush_partial()
    batched_partial = batched.flush_partial()
    assert scalar_partial is not None
    assert batched_partial is not None
    assert batched_partial.to_dict() == scalar_partial.to_dict()


def test_record_step_batch_supports_dict_of_arrays_infos() -> None:
    rewards = np.array([0.2, 0.4], dtype=np.float32)
    terminated = np.array([False, True], dtype=bool)
    truncated = np.array([False, False], dtype=bool)

    infos = {
        "dt": np.array([3, 2], dtype=np.int32),
        "invalid_action": np.array([1, 0], dtype=np.uint8),
    }
    for key in INFO_METRIC_KEYS:
        infos[key] = np.array([1.0, 2.0], dtype=np.float32)

    agg = WindowMetricsAggregator(run_id="dict-infos", window_env_steps=5)
    emitted = agg.record_step_batch(
        rewards=rewards,
        infos=infos,
        terminated=terminated,
        truncated=truncated,
    )

    assert len(emitted) == 1
    record = emitted[0]
    assert record.window_id == 0
    assert record.env_steps_in_window == 5
    assert record.invalid_action_rate == 3.0 / 5.0
    assert record.episodes_completed == 1
    assert record.terminated_episodes == 1
