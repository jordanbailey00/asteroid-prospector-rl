import numpy as np
from asteroid_prospector import ProspectorReferenceEnv
from asteroid_prospector.constants import MAX_ASTEROIDS, MAX_NEIGHBORS, OBS_DIM


def test_reference_obs_contract_layout_and_ranges() -> None:
    env = ProspectorReferenceEnv(seed=44)
    obs, _ = env.reset(seed=44)

    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32

    ship_and_market_ranges = np.r_[obs[0:24], obs[244:250], obs[256:260]]
    assert np.all(ship_and_market_ranges >= 0.0)
    assert np.all(ship_and_market_ranges <= 1.0)

    assert np.all(obs[250:256] >= -1.0)
    assert np.all(obs[250:256] <= 1.0)

    node_onehot = obs[19:22]
    assert np.isclose(float(np.sum(node_onehot)), 1.0, atol=1e-6)

    for slot in range(MAX_NEIGHBORS):
        base = 24 + 7 * slot
        valid = obs[base] > 0.5
        type_sum = float(np.sum(obs[base + 1 : base + 4]))
        if valid:
            assert np.isclose(type_sum, 1.0, atol=1e-6)
        else:
            assert np.isclose(type_sum, 0.0, atol=1e-6)

    selected_count = 0
    for asteroid in range(MAX_ASTEROIDS):
        base = 68 + 11 * asteroid
        valid = obs[base] > 0.5
        comp = obs[base + 1 : base + 7]

        if valid:
            assert np.isclose(float(np.sum(comp)), 1.0, atol=1e-4)
        else:
            assert np.isclose(float(np.sum(comp)), 0.0, atol=1e-6)

        if obs[base + 10] > 0.5:
            selected_count += 1

    assert selected_count <= 1


def test_reference_action_space_matches_frozen_contract() -> None:
    env = ProspectorReferenceEnv(seed=1)
    assert env.action_space.n == 69
