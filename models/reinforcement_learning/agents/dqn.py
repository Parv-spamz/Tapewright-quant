"""
Deep Q-Network (DQN) agent written in pure PyTorch.

Features:
- Fixed-capacity circular experience replay buffer.
- Epsilon-greedy exploration schedule with linear decay.
- Hard target network updates.
- Double-DQN option (Hasselt et al., 2015) to mitigate Q-value overestimation.
- Huber (Smooth L1) loss function.
"""

from __future__ import annotations

import random
from typing import Any, Sequence, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.reinforcement_learning.agents.networks import QNetwork


class ReplayBuffer:
    """Fixed-capacity FIFO circular replay buffer for off-policy transition storage."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.pos: int = 0
        self.size: int = 0

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Appends a transition, overwriting oldest records when capacity is reached."""
        self.obs[self.pos] = obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = 1.0 if done else 0.0

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Samples a uniform random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[indices], dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions[indices], dtype=torch.long, device=device),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones[indices], dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return self.size


class EpsilonSchedule:
    """Linear epsilon decay exploration schedule."""

    def __init__(
        self,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        decay_steps: int = 1000,
    ) -> None:
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.decay_steps = max(1, int(decay_steps))

    def get_epsilon(self, step: int) -> float:
        """Returns the exploration rate for a given step index."""
        if step <= 0:
            return self.eps_start
        if step >= self.decay_steps:
            return self.eps_end
        frac = float(step) / float(self.decay_steps)
        return float(self.eps_start + frac * (self.eps_end - self.eps_start))


class DQNAgent:
    """Deep Q-Network agent supporting Double-DQN and Huber loss.

    Parameters
    ----------
    env : gym.Env
        Discrete Gymnasium environment.
    lr : float, default 1e-3
        Adam optimizer learning rate.
    gamma : float, default 0.99
        Discount factor.
    buffer_capacity : int, default 10_000
        Maximum replay buffer storage size.
    batch_size : int, default 64
        Batch size per gradient descent update.
    target_update_freq : int, default 100
        Environment steps between hard target network syncs.
    double_dqn : bool, default True
        Enables Double-DQN decoupled action selection and evaluation.
    eps_start : float, default 1.0
        Initial epsilon exploration rate.
    eps_end : float, default 0.05
        Terminal epsilon exploration floor.
    eps_decay_steps : int, default 1000
        Number of steps over which epsilon decays linearly.
    hidden_dims : sequence of int, default (64, 64)
        Hidden dimensions for QNetwork MLP layers.
    seed : int, default 42
        Reproducibility seed.
    device : str, default 'cpu'
        Compute device.
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        double_dqn: bool = True,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 1000,
        hidden_dims: Sequence[int] = (64, 64),
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError("DQNAgent requires a gymnasium.spaces.Discrete action space.")

        self.env = env
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.double_dqn = bool(double_dqn)
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.n_actions = int(env.action_space.n)

        # Q-Networks: online policy network and target network
        self.policy_net = QNetwork(self.obs_dim, self.n_actions, hidden_dims=hidden_dims).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.n_actions, hidden_dims=hidden_dims).to(self.device)
        self.update_target_network()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.buffer = ReplayBuffer(buffer_capacity, self.obs_dim)
        self.epsilon_schedule = EpsilonSchedule(
            eps_start=eps_start,
            eps_end=eps_end,
            decay_steps=eps_decay_steps,
        )
        self.total_steps: int = 0

    def update_target_network(self) -> None:
        """Hard-synchronizes target network weights with online policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Selects action via greedy or epsilon-greedy exploration."""
        eps = 0.0 if deterministic else self.epsilon_schedule.get_epsilon(self.total_steps)

        if not deterministic and random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        self.policy_net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs_t)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action

    def train_step(self) -> float:
        """Samples a mini-batch from the buffer and optimizes the Q-network with Huber loss."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        self.policy_net.train()
        # Q(s, a) predicted by policy network
        current_q = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Decoupled action selection via policy network, evaluation via target network
                next_actions = self.policy_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values

            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train(self, total_timesteps: int) -> list[float]:
        """Trains the DQN agent on the environment for total_timesteps."""
        obs, _ = self.env.reset()
        episode_rewards: list[float] = []
        current_ep_reward = 0.0

        for _ in range(total_timesteps):
            self.total_steps += 1
            action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            current_ep_reward += float(reward)

            self.buffer.push(obs=obs, action=action, reward=float(reward), next_obs=next_obs, done=done)
            self.train_step()

            if self.total_steps % self.target_update_freq == 0:
                self.update_target_network()

            obs = next_obs
            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                obs, _ = self.env.reset()

        return episode_rewards
