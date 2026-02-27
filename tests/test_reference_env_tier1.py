import numpy as np
from asteroid_prospector import ProspectorReferenceEnv
from asteroid_prospector.constants import (
    ALERT_MAX,
    CARGO_MAX,
    FUEL_MAX,
    HEAT_MAX,
    HULL_MAX,
    TOOL_MAX,
)


def _first_valid_travel_action(env: ProspectorReferenceEnv) -> int:
    for slot in range(6):
        if int(env.neighbors[env.current_node, slot]) >= 0:
            return slot
    raise AssertionError("Expected at least one valid neighbor from current node")


def test_resource_clamps_hold_across_rollout() -> None:
    env = ProspectorReferenceEnv(seed=123)
    obs, _ = env.reset(seed=123)
    assert obs.shape == (260,)

    rng = np.random.default_rng(123)
    for _ in range(300):
        action = int(rng.integers(0, 69))
        _, _, terminated, truncated, _ = env.step(action)

        assert 0.0 <= env.fuel <= FUEL_MAX
        assert 0.0 <= env.hull <= HULL_MAX
        assert 0.0 <= env.heat <= HEAT_MAX
        assert 0.0 <= env.tool_condition <= TOOL_MAX
        assert 0.0 <= env.alert <= ALERT_MAX
        assert 0.0 <= float(np.sum(env.cargo)) <= CARGO_MAX + 1e-6

        if terminated or truncated:
            env.reset(seed=456)


def test_scan_updates_keep_composition_normalized() -> None:
    env = ProspectorReferenceEnv(seed=7)
    env.reset(seed=7)

    travel_action = _first_valid_travel_action(env)
    env.step(travel_action)

    valid_asteroids = np.where(env.ast_valid[env.current_node] > 0)[0]
    assert valid_asteroids.size > 0
    asteroid = int(valid_asteroids[0])

    _, _, _, _, select_info = env.step(12 + asteroid)
    assert select_info["invalid_action"] is False

    conf_before = float(env.scan_conf[env.current_node, asteroid])
    env.step(10)  # DEEP_SCAN_SELECTED
    conf_after = float(env.scan_conf[env.current_node, asteroid])

    comp = env.comp_est[env.current_node, asteroid]
    assert conf_after >= conf_before
    assert np.isclose(float(np.sum(comp)), 1.0, atol=1e-4)


def test_market_prices_stay_finite_and_bounded() -> None:
    env = ProspectorReferenceEnv(seed=99)
    env.reset(seed=99)

    rng = np.random.default_rng(99)
    price_min = np.array(env.config.price_min)
    price_max = np.array(env.config.price_max)

    for _ in range(300):
        action = int(rng.integers(0, 69))
        _, _, terminated, truncated, _ = env.step(action)

        assert np.isfinite(env.price).all()
        assert np.all(env.price >= price_min - 1e-6)
        assert np.all(env.price <= price_max + 1e-6)

        if terminated or truncated:
            env.reset(seed=100)
