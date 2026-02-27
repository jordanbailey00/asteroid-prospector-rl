from asteroid_prospector import ProspectorReferenceEnv


def _first_valid_travel_action(env: ProspectorReferenceEnv) -> int:
    for slot in range(6):
        if int(env.neighbors[env.current_node, slot]) >= 0:
            return slot
    raise AssertionError("Expected a valid travel action")


def test_station_only_actions_invalid_in_field() -> None:
    env = ProspectorReferenceEnv(seed=11)
    env.reset(seed=11)

    travel_action = _first_valid_travel_action(env)
    env.step(travel_action)
    assert not env._is_at_station()  # noqa: SLF001

    credits_before = env.credits
    _, _, _, _, sell_info = env.step(43)  # SELL[commodity0,25%]
    assert sell_info["invalid_action"] is True
    assert env.credits <= credits_before + 1e-6

    _, _, _, _, buy_info = env.step(61)  # BUY_FUEL_SMALL
    assert buy_info["invalid_action"] is True

    _, _, _, _, overhaul_info = env.step(67)
    assert overhaul_info["invalid_action"] is True
