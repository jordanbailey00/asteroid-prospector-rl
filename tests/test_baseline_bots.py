import pytest
from asteroid_prospector import N_ACTIONS, ProspectorReferenceEnv
from asteroid_prospector.constants import N_COMMODITIES

from training.baseline_bots import BASELINE_BOT_NAMES, get_baseline_bot, make_market_timer_policy


def test_baseline_bots_produce_valid_actions_across_rollout() -> None:
    for bot_name in BASELINE_BOT_NAMES:
        policy = get_baseline_bot(bot_name, target_commodity=3)
        env = ProspectorReferenceEnv(seed=17)
        obs, _info = env.reset(seed=17)

        next_seed = 18
        for _ in range(256):
            action = int(policy(obs))
            assert 0 <= action < N_ACTIONS

            obs, _reward, terminated, truncated, _step_info = env.step(action)
            if terminated or truncated:
                obs, _info = env.reset(seed=next_seed)
                next_seed += 1


def test_baseline_bots_decisions_are_deterministic_for_fixed_obs() -> None:
    env = ProspectorReferenceEnv(seed=23)
    obs, _info = env.reset(seed=23)

    for bot_name in BASELINE_BOT_NAMES:
        policy = get_baseline_bot(bot_name, target_commodity=2)
        action_a = int(policy(obs.copy()))
        action_b = int(policy(obs.copy()))
        assert action_a == action_b


def test_market_timer_policy_validates_target_commodity() -> None:
    with pytest.raises(ValueError):
        make_market_timer_policy(target_commodity=-1)

    with pytest.raises(ValueError):
        make_market_timer_policy(target_commodity=N_COMMODITIES)


def test_get_baseline_bot_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_baseline_bot("not-a-bot")
