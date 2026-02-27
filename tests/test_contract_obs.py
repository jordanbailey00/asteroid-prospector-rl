import numpy as np
from asteroid_prospector import OBS_DIM, HelloProspectorEnv


def test_reset_obs_shape_and_dtype() -> None:
    env = HelloProspectorEnv()
    obs, info = env.reset(seed=7)

    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert info["seed"] == 7


def test_step_returns_gymnasium_five_tuple() -> None:
    env = HelloProspectorEnv()
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)


def test_reset_is_deterministic_under_fixed_seed() -> None:
    env = HelloProspectorEnv()

    obs_a, _ = env.reset(seed=1234)
    obs_b, _ = env.reset(seed=1234)

    np.testing.assert_array_equal(obs_a, obs_b)
