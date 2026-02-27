from asteroid_prospector import N_ACTIONS, HelloProspectorEnv


def test_action_space_n_matches_frozen_contract() -> None:
    env = HelloProspectorEnv()
    assert env.action_space.n == N_ACTIONS == 69


def test_all_actions_0_to_68_are_accepted() -> None:
    env = HelloProspectorEnv()
    env.reset(seed=0)

    for action in range(N_ACTIONS):
        _, reward, terminated, truncated, info = env.step(action)
        assert terminated is False
        assert truncated is False
        assert reward == 0.0
        assert info["invalid_action"] is False


def test_invalid_actions_are_flagged_and_penalized() -> None:
    env = HelloProspectorEnv()
    env.reset(seed=0)

    for invalid_action in (-1, N_ACTIONS, 999):
        _, reward, terminated, truncated, info = env.step(invalid_action)
        assert terminated is False
        assert truncated is False
        assert reward < 0.0
        assert info["invalid_action"] is True
