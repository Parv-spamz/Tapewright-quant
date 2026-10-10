"""
PyTorch neural network architectures for reinforcement learning agents.

Contains:
- MLP building block.
- ActorCritic supporting discrete (Categorical) and continuous (Gaussian) action spaces.
- QNetwork for deep Q-learning.
"""

from __future__ import annotations

from typing import Sequence, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    """Builds a Multi-Layer Perceptron (MLP)."""
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO supporting Discrete and Box action spaces."""

    def __init__(
        self,
        obs_dim: int,
        action_space: Union[spaces.Discrete, spaces.Box],
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space

        # Critic value network
        self.critic = build_mlp(obs_dim, hidden_dims, 1)

        # Actor policy network
        if isinstance(action_space, spaces.Discrete):
            self.is_continuous = False
            self.n_actions = int(action_space.n)
            self.actor = build_mlp(obs_dim, hidden_dims, self.n_actions)
        elif isinstance(action_space, spaces.Box):
            self.is_continuous = True
            self.action_dim = int(np.prod(action_space.shape))
            self.actor_mean = build_mlp(obs_dim, hidden_dims, self.action_dim, output_activation=nn.Tanh)
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
        else:
            raise TypeError(f"Unsupported action space type: {type(action_space)}")

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes critic state value V(s)."""
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples or evaluates action, returning (action, log_prob, entropy, value)."""
        val = self.get_value(obs)

        if not self.is_continuous:
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            if action is None:
                if deterministic:
                    action_out = torch.argmax(logits, dim=-1)
                else:
                    action_out = dist.sample()
            else:
                action_out = action
            log_prob = dist.log_prob(action_out)
            entropy = dist.entropy()
            return action_out, log_prob, entropy, val
        else:
            mean = self.actor_mean(obs)
            std = torch.exp(self.actor_logstd).expand_as(mean)
            dist = Normal(mean, std)
            if action is None:
                if deterministic:
                    action_out = mean
                else:
                    action_out = dist.sample()
            else:
                action_out = action
            log_prob = dist.log_prob(action_out).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return action_out, log_prob, entropy, val


class QNetwork(nn.Module):
    """Deep Q-Network for discrete action-value estimation."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.network = build_mlp(obs_dim, hidden_dims, n_actions, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Q(s, a) for all actions."""
        return self.network(obs)
