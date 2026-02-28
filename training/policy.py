from __future__ import annotations

from typing import Any

import numpy as np

POLICY_ARCH = "mlp-256x256-tanh-v1"


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - exercised in Linux trainer runtime
        raise RuntimeError(
            "torch is required for puffer_ppo policy operations. "
            "Install torch or use trainer_backend='random'."
        ) from exc
    return torch, nn


def obs_dim_from_shape(obs_shape: tuple[int, ...]) -> int:
    return int(np.prod(tuple(int(v) for v in obs_shape)))


def create_actor_critic(*, obs_shape: tuple[int, ...], n_actions: int, device: str) -> Any:
    torch, nn = _require_torch()

    obs_dim = obs_dim_from_shape(obs_shape)

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
            )
            self.actor = nn.Linear(256, int(n_actions))
            self.critic = nn.Linear(256, 1)

        def _features(self, obs: Any) -> Any:
            return self.encoder(obs.view(obs.shape[0], -1))

        def policy_logits(self, obs: Any) -> Any:
            return self.actor(self._features(obs))

        def get_value(self, obs: Any) -> Any:
            return self.critic(self._features(obs)).squeeze(-1)

        def get_action_and_value(
            self,
            obs: Any,
            action: Any | None = None,
        ) -> tuple[Any, Any, Any, Any]:
            from torch.distributions.categorical import Categorical

            logits = self.policy_logits(obs)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return (
                action,
                dist.log_prob(action),
                dist.entropy(),
                self.critic(self._features(obs)).squeeze(-1),
            )

    return ActorCritic().to(device)


def export_policy_state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def load_policy_state_dict(model: Any, state_dict: dict[str, Any]) -> None:
    model.load_state_dict(state_dict)


def select_policy_action(*, model: Any, obs: np.ndarray, deterministic: bool) -> int:
    torch, _ = _require_torch()

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        logits = model.policy_logits(obs_tensor)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            from torch.distributions.categorical import Categorical

            dist = Categorical(logits=logits)
            action = dist.sample()
    return int(action.item())
