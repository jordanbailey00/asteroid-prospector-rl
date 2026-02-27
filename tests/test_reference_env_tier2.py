import numpy as np
from asteroid_prospector import ProspectorReferenceEnv
from asteroid_prospector.constants import OBS_DIM

REQUIRED_INFO_KEYS = {
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
}


def test_short_rollouts_across_seeds_do_not_crash() -> None:
    for seed in range(5):
        env = ProspectorReferenceEnv(seed=seed)
        obs, _ = env.reset(seed=seed)

        rng = np.random.default_rng(seed)
        for step_idx in range(200):
            assert obs.shape == (OBS_DIM,)
            assert obs.dtype == np.float32

            action = int(rng.integers(0, 69))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + step_idx + 1)


def test_info_metrics_keys_always_present() -> None:
    env = ProspectorReferenceEnv(seed=222)
    env.reset(seed=222)

    rng = np.random.default_rng(222)
    for _ in range(120):
        action = int(rng.integers(0, 69))
        _, _, terminated, truncated, info = env.step(action)

        assert REQUIRED_INFO_KEYS.issubset(set(info.keys()))

        if terminated or truncated:
            env.reset(seed=333)
