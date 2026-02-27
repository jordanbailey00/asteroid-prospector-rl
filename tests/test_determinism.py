import math

import numpy as np
from asteroid_prospector import ProspectorReferenceEnv

NUMERIC_INFO_KEYS = (
    "credits",
    "net_profit",
    "profit_per_tick",
    "survival",
    "overheat_ticks",
    "pirate_encounters",
    "value_lost_to_pirates",
    "fuel_used",
    "hull_damage",
    "tool_wear",
    "scan_count",
    "mining_ticks",
    "cargo_utilization_avg",
    "time_remaining",
)


def test_fixed_seed_and_action_sequence_is_deterministic() -> None:
    seed = 31415
    actions = np.random.default_rng(7).integers(-3, 74, size=300)

    env_a = ProspectorReferenceEnv(seed=seed)
    env_b = ProspectorReferenceEnv(seed=seed)

    obs_a, _ = env_a.reset(seed=seed)
    obs_b, _ = env_b.reset(seed=seed)
    np.testing.assert_allclose(obs_a, obs_b, atol=0.0, rtol=0.0)

    episode_seed = seed
    for action in actions:
        obs_a, rew_a, term_a, trunc_a, info_a = env_a.step(int(action))
        obs_b, rew_b, term_b, trunc_b, info_b = env_b.step(int(action))

        np.testing.assert_allclose(obs_a, obs_b, atol=0.0, rtol=0.0)
        assert math.isclose(rew_a, rew_b, rel_tol=0.0, abs_tol=0.0)
        assert term_a is term_b
        assert trunc_a is trunc_b

        for key in NUMERIC_INFO_KEYS:
            assert math.isclose(info_a[key], info_b[key], rel_tol=0.0, abs_tol=0.0)

        assert info_a["invalid_action"] is info_b["invalid_action"]
        assert info_a["dt"] == info_b["dt"]

        if term_a or trunc_a:
            episode_seed += 1
            obs_a, _ = env_a.reset(seed=episode_seed)
            obs_b, _ = env_b.reset(seed=episode_seed)
            np.testing.assert_allclose(obs_a, obs_b, atol=0.0, rtol=0.0)


def test_different_seeds_produce_different_initial_world_observations() -> None:
    env_a = ProspectorReferenceEnv(seed=100)
    env_b = ProspectorReferenceEnv(seed=101)

    obs_a, _ = env_a.reset(seed=100)
    obs_b, _ = env_b.reset(seed=101)

    assert not np.array_equal(obs_a, obs_b)
