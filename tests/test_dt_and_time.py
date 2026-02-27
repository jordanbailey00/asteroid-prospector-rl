import numpy as np
from asteroid_prospector import ProspectorReferenceEnv


def _first_valid_travel_action(env: ProspectorReferenceEnv) -> int:
    for slot in range(6):
        if int(env.neighbors[env.current_node, slot]) >= 0:
            return slot
    raise AssertionError("Expected a valid travel action")


def test_dt_reduces_time_remaining_exactly() -> None:
    env = ProspectorReferenceEnv(seed=17)
    env.reset(seed=17)

    travel_action = _first_valid_travel_action(env)
    time_before = env.time_remaining

    _, _, _, _, info = env.step(travel_action)
    dt = int(info["dt"])

    assert np.isclose(env.time_remaining, time_before - dt)


def test_invalid_action_matches_hold_with_penalty() -> None:
    env_invalid = ProspectorReferenceEnv(seed=123)
    env_hold = ProspectorReferenceEnv(seed=123)

    env_invalid.reset(seed=123)
    env_hold.reset(seed=123)

    obs_invalid, reward_invalid, _, _, info_invalid = env_invalid.step(999)
    obs_hold, reward_hold, _, _, info_hold = env_hold.step(6)

    np.testing.assert_allclose(obs_invalid, obs_hold, atol=1e-7)
    assert info_invalid["invalid_action"] is True
    assert info_hold["invalid_action"] is False

    expected_delta = env_invalid.config.reward_cfg.invalid_action_pen
    assert np.isclose(reward_invalid, reward_hold - expected_delta)
