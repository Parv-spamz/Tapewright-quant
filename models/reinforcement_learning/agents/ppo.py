"""
Proximal Policy Optimization (PPO) agent written in pure PyTorch.

Features:
- Generalized Advantage Estimation GAE(lambda).
- Clipped surrogate policy objective.
- Value function squared error loss.
- Policy entropy exploration bonus.
- Advantage normalization.
- Gradient clipping.
- Minibatch SGD/Adam epochs.
- Supports both discrete (Categorical) and continuous (Gaussian) action spaces.
"""

from __future__ import annotations

from typing import Any, Generator, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.reinforcement_learning.agents.networks import ActorCritic


class RolloutBuffer:
    """In-memory rollout storage for on-policy trajectories."""

    def __init__(self, capacity: int, obs_dim: int, is_continuous: bool, action_dim: int = 1) -> None:
        self.capacity = capacity
        self.is_continuous = is_continuous
        self.action_dim = action_dim

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        if is_continuous:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros((capacity,), dtype=np.int64)

        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float64)
        self.dones = np.zeros((capacity,), dtype=np.float64)
        self.values = np.zeros((capacity,), dtype=np.float64)

        self.advantages = np.zeros((capacity,), dtype=np.float64)
        self.returns = np.zeros((capacity,), dtype=np.float64)
        self.ptr: int = 0

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, float, np.ndarray],
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Stores a transition into the buffer."""
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.log_probs[idx] = float(log_prob)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0
        self.values[idx] = float(value)
        self.ptr += 1

    def compute_gae(self, last_val: float, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """Computes GAE advantages and target returns backwards from rollout end.

        Formula:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
            A_t = delta_t + gamma * lambda * (1 - d_t) * A_{t+1}
            R_t = A_t + V(s_t)
        """
        last_adv = 0.0
        g = float(gamma)
        lam = float(gae_lambda)
        lv = float(last_val)

        for t in reversed(range(self.ptr)):
            next_non_terminal = float(1.0 - self.dones[t])
            if t == self.ptr - 1:
                next_value = lv
            else:
                next_value = float(self.values[t + 1])

            delta = float(self.rewards[t]) + g * next_value * next_non_terminal - float(self.values[t])
            last_adv = delta + g * lam * next_non_terminal * last_adv
            self.advantages[t] = last_adv

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_minibatches(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Generates random minibatches for policy optimization."""
        indices = np.random.permutation(self.ptr)
        for start_idx in range(0, self.ptr, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield {
                "obs": torch.tensor(self.obs[batch_indices], dtype=torch.float32, device=device),
                "actions": (
                    torch.tensor(self.actions[batch_indices], dtype=torch.float32, device=device)
                    if self.is_continuous
                    else torch.tensor(self.actions[batch_indices], dtype=torch.long, device=device)
                ),
                "log_probs": torch.tensor(self.log_probs[batch_indices], dtype=torch.float32, device=device),
                "advantages": torch.tensor(self.advantages[batch_indices], dtype=torch.float32, device=device),
                "returns": torch.tensor(self.returns[batch_indices], dtype=torch.float32, device=device),
            }


class PPOAgent:
    """Proximal Policy Optimization agent.

    Parameters
    ----------
    env : gym.Env
        Gymnasium training environment.
    lr : float, default 3e-4
        Adam learning rate.
    gamma : float, default 0.99
        Discount factor.
    gae_lambda : float, default 0.95
        Generalized advantage estimation factor.
    clip_coef : float, default 0.2
        PPO surrogate clipping parameter epsilon.
    ent_coef : float, default 0.01
        Entropy bonus coefficient.
    vf_coef : float, default 0.5
        Value function loss coefficient.
    max_grad_norm : float, default 0.5
        Gradient norm clipping maximum.
    update_epochs : int, default 4
        Number of optimization epochs per rollout.
    minibatch_size : int, default 64
        Batch size per gradient step.
    hidden_dims : sequence of int, default (64, 64)
        MLP hidden layers for actor and critic.
    seed : int, default 42
        Random seed for reproducibility.
    device : str, default 'cpu'
        Target compute device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        hidden_dims: Sequence[int] = (64, 64),
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_coef = float(clip_coef)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            action_space=env.action_space,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.is_continuous = self.actor_critic.is_continuous
        self.action_dim = getattr(self.actor_critic, "action_dim", 1)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Any:
        """Returns action for given observation in numpy format."""
        self.actor_critic.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, _, _ = self.actor_critic.get_action_and_value(obs_t, deterministic=deterministic)
            action_np = action.cpu().numpy()[0]
        return action_np

    def train_step(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Runs update epochs on transitions stored in rollout buffer."""
        self.actor_critic.train()

        pg_losses: list[float] = []
        vf_losses: list[float] = []
        entropy_losses: list[float] = []

        for _ in range(self.update_epochs):
            for batch in buffer.get_minibatches(self.minibatch_size, self.device):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalize advantages
                adv_std = torch.std(advantages)
                adv_mean = torch.mean(advantages)
                norm_advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                # Re-evaluate action probabilities and values
                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(obs, actions)

                # Clipped surrogate policy objective
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * norm_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * norm_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value function squared loss
                vf_loss = 0.5 * ((new_values - returns) ** 2).mean()

                # Entropy exploration loss
                ent_loss = -entropy.mean()

                total_loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                entropy_losses.append(ent_loss.item())

        return {
            "policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "value_loss": float(np.mean(vf_losses)) if vf_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        }

    def train(self, total_timesteps: int, rollout_steps: int = 128) -> list[float]:
        """Trains the agent for total_timesteps."""
        obs_dim = int(np.prod(self.env.observation_space.shape))
        buffer = RolloutBuffer(rollout_steps, obs_dim, self.is_continuous, self.action_dim)

        obs, _ = self.env.reset()
        timesteps = 0
        episode_rewards: list[float] = []
        current_ep_reward = 0.0

        while timesteps < total_timesteps:
            buffer.ptr = 0
            for _ in range(rollout_steps):
                timesteps += 1
                self.actor_critic.eval()
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_t, log_prob_t, _, val_t = self.actor_critic.get_action_and_value(obs_t)

                if self.is_continuous:
                    action_env = action_t.cpu().numpy()[0]
                else:
                    action_env = int(action_t.cpu().numpy()[0])

                next_obs, reward, terminated, truncated, _ = self.env.step(action_env)
                done = terminated or truncated
                current_ep_reward += float(reward)

                buffer.add(
                    obs=obs,
                    action=action_env,
                    log_prob=float(log_prob_t.cpu().item()),
                    reward=float(reward),
                    done=done,
                    value=float(val_t.cpu().item()),
                )

                obs = next_obs
                if done:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0
                    obs, _ = self.env.reset()

            # End of rollout GAE computation
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_val = float(self.actor_critic.get_value(obs_t).cpu().item())

            buffer.compute_gae(last_val=last_val, gamma=self.gamma, gae_lambda=self.gae_lambda)
            self.train_step(buffer)

        return episode_rewards
