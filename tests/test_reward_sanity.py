import numpy as np
from asteroid_prospector import ProspectorReferenceEnv


def test_reward_is_finite_across_random_rollout() -> None:
    env = ProspectorReferenceEnv(seed=101)
    env.reset(seed=101)

    rng = np.random.default_rng(101)
    for _ in range(300):
        action = int(rng.integers(-5, 75))
        _, reward, terminated, truncated, _ = env.step(action)

        assert np.isfinite(reward)
        if terminated or truncated:
            env.reset(seed=102)


def test_sell_action_produces_positive_reward_when_cargo_exists() -> None:
    env = ProspectorReferenceEnv(seed=5)
    env.reset(seed=5)

    env.cargo[0] = np.float32(80.0)
    credits_before = env.credits

    _, reward, terminated, truncated, info = env.step(45)  # SELL[commodity0,100%]

    assert terminated is False
    assert truncated is False
    assert info["invalid_action"] is False
    assert env.credits > credits_before
    assert reward > 0.0


def test_fuel_burn_is_worse_than_hold_from_same_state() -> None:
    env_hold = ProspectorReferenceEnv(seed=77)
    env_burn = ProspectorReferenceEnv(seed=77)

    env_hold.reset(seed=77)
    env_burn.reset(seed=77)

    _, reward_hold, _, _, _ = env_hold.step(6)  # HOLD_DRIFT
    _, reward_burn, _, _, _ = env_burn.step(7)  # EMERGENCY_BURN

    assert env_burn.fuel < env_hold.fuel
    assert reward_burn < reward_hold


def test_high_heat_state_gets_stronger_penalty() -> None:
    env_low = ProspectorReferenceEnv(seed=88)
    env_high = ProspectorReferenceEnv(seed=88)

    env_low.reset(seed=88)
    env_high.reset(seed=88)

    env_low.heat = 60.0
    env_high.heat = 95.0

    _, reward_low, _, _, _ = env_low.step(6)
    _, reward_high, _, _, _ = env_high.step(6)

    assert reward_high < reward_low
