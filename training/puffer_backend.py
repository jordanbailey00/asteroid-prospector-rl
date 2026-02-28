from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    from training.policy import POLICY_ARCH, create_actor_critic, export_policy_state_dict_cpu
else:
    from .policy import POLICY_ARCH, create_actor_critic, export_policy_state_dict_cpu


@dataclass(frozen=True)
class PpoConfig:
    total_env_steps: int
    seed: int
    env_time_max: float
    num_envs: int = 8
    num_workers: int = 4
    rollout_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    vector_backend: str = "multiprocessing"  # serial|multiprocessing


StepCallback = Callable[[float, dict[str, Any], bool, bool], bool]
StateGetter = Callable[[], dict[str, Any]]
RegisterStateGetter = Callable[[StateGetter], None]


def _validate_config(cfg: PpoConfig) -> None:
    if cfg.total_env_steps <= 0:
        raise ValueError("total_env_steps must be positive")
    if cfg.num_envs <= 0:
        raise ValueError("ppo_num_envs must be positive")
    if cfg.num_workers <= 0:
        raise ValueError("ppo_num_workers must be positive")
    if cfg.rollout_steps <= 0:
        raise ValueError("ppo_rollout_steps must be positive")
    if cfg.num_minibatches <= 0:
        raise ValueError("ppo_num_minibatches must be positive")
    if cfg.update_epochs <= 0:
        raise ValueError("ppo_update_epochs must be positive")
    if cfg.learning_rate <= 0.0:
        raise ValueError("ppo_learning_rate must be positive")
    if cfg.max_grad_norm <= 0.0:
        raise ValueError("ppo_max_grad_norm must be positive")
    if cfg.vector_backend not in {"serial", "multiprocessing"}:
        raise ValueError("ppo_vector_backend must be one of: serial, multiprocessing")
    if cfg.vector_backend == "multiprocessing" and cfg.num_envs % cfg.num_workers != 0:
        raise ValueError("ppo_num_envs must be divisible by ppo_num_workers")


class _ProspectorGymEnv:
    metadata = {"render_modes": []}

    def __init__(self, *, time_max: float, seed: int | None = None) -> None:
        import gymnasium as gym
        from asteroid_prospector import (
            N_ACTIONS,
            OBS_DIM,
            ProspectorReferenceEnv,
            ReferenceEnvConfig,
        )

        self._env = ProspectorReferenceEnv(config=ReferenceEnvConfig(time_max=time_max), seed=seed)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.render_mode = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        return None


def run_puffer_ppo_training(
    *,
    cfg: PpoConfig,
    on_step: StepCallback,
    register_checkpoint_state_getter: RegisterStateGetter | None = None,
) -> dict[str, Any]:
    _validate_config(cfg)

    import pufferlib.emulation
    import pufferlib.vector
    import torch
    import torch.nn as nn
    import torch.optim as optim

    backend = {
        "serial": pufferlib.vector.Serial,
        "multiprocessing": pufferlib.vector.Multiprocessing,
    }[cfg.vector_backend]

    seed_counter = {"value": 0}

    def make_env(*, buf: Any | None = None) -> Any:
        seed = cfg.seed + seed_counter["value"]
        seed_counter["value"] += 1
        return pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=_ProspectorGymEnv,
            env_kwargs={"time_max": cfg.env_time_max, "seed": seed},
            buf=buf,
        )

    vec_kwargs: dict[str, Any] = {
        "backend": backend,
        "num_envs": cfg.num_envs,
    }
    if backend is pufferlib.vector.Multiprocessing:
        vec_kwargs["num_workers"] = cfg.num_workers

    envs = pufferlib.vector.make(make_env, **vec_kwargs)

    try:
        obs_shape = tuple(int(v) for v in envs.single_observation_space.shape)
        n_actions = int(envs.single_action_space.n)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = create_actor_critic(obs_shape=obs_shape, n_actions=n_actions, device=device)
        optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

        if register_checkpoint_state_getter is not None:

            def snapshot_state() -> dict[str, Any]:
                return {
                    "policy_arch": POLICY_ARCH,
                    "obs_shape": list(obs_shape),
                    "n_actions": int(n_actions),
                    "model_state_dict": export_policy_state_dict_cpu(agent),
                }

            register_checkpoint_state_getter(snapshot_state)

        rollout_obs = torch.zeros((cfg.rollout_steps, cfg.num_envs) + obs_shape, device=device)
        rollout_actions = torch.zeros(
            (cfg.rollout_steps, cfg.num_envs), dtype=torch.long, device=device
        )
        rollout_logprobs = torch.zeros((cfg.rollout_steps, cfg.num_envs), device=device)
        rollout_rewards = torch.zeros((cfg.rollout_steps, cfg.num_envs), device=device)
        rollout_dones = torch.zeros((cfg.rollout_steps, cfg.num_envs), device=device)
        rollout_values = torch.zeros((cfg.rollout_steps, cfg.num_envs), device=device)

        next_obs_np, _ = envs.reset(seed=cfg.seed)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        next_done = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

        policy_updates = 0
        stop_requested = False

        while not stop_requested:
            collected_steps = 0

            for step in range(cfg.rollout_steps):
                collected_steps = step + 1

                rollout_obs[step] = next_obs
                rollout_dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)

                rollout_actions[step] = action
                rollout_logprobs[step] = logprob
                rollout_values[step] = value

                next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(action.cpu().numpy())
                done_np = np.logical_or(term_np, trunc_np)

                rollout_rewards[step] = torch.as_tensor(
                    reward_np, dtype=torch.float32, device=device
                )
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

                info_rows = list(infos) if isinstance(infos, list) else []
                for i in range(cfg.num_envs):
                    info = (
                        info_rows[i]
                        if i < len(info_rows) and isinstance(info_rows[i], dict)
                        else {}
                    )
                    stop_requested = on_step(
                        float(reward_np[i]),
                        info,
                        bool(term_np[i]),
                        bool(trunc_np[i]),
                    )
                    if stop_requested:
                        break
                if stop_requested:
                    break

            if collected_steps <= 0:
                break

            with torch.no_grad():
                next_value = agent.get_value(next_obs)
                advantages = torch.zeros((collected_steps, cfg.num_envs), device=device)
                last_gae = torch.zeros(cfg.num_envs, device=device)
                for t in reversed(range(collected_steps)):
                    if t == collected_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - rollout_dones[t + 1]
                        next_values = rollout_values[t + 1]

                    delta = (
                        rollout_rewards[t]
                        + cfg.gamma * next_values * next_nonterminal
                        - rollout_values[t]
                    )
                    last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + rollout_values[:collected_steps]

            b_obs = rollout_obs[:collected_steps].reshape((-1,) + obs_shape)
            b_actions = rollout_actions[:collected_steps].reshape(-1)
            b_logprobs = rollout_logprobs[:collected_steps].reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = rollout_values[:collected_steps].reshape(-1)

            batch_size = collected_steps * cfg.num_envs
            minibatch_size = max(1, batch_size // cfg.num_minibatches)
            indices = np.arange(batch_size)

            for _ in range(cfg.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = indices[start : start + minibatch_size]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds],
                        action=b_actions[mb_inds],
                    )

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = torch.exp(logratio)

                    mb_adv = b_advantages[mb_inds]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

            policy_updates += 1

        return {
            "ppo_device": device,
            "ppo_num_envs": cfg.num_envs,
            "ppo_num_workers": cfg.num_workers,
            "ppo_rollout_steps": cfg.rollout_steps,
            "ppo_policy_updates": policy_updates,
            "ppo_vector_backend": cfg.vector_backend,
            "ppo_policy_arch": POLICY_ARCH,
        }
    finally:
        envs.close()
