"""
Agent evaluation harness computing financial tearsheet risk and performance metrics.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from backtesting.metrics.risk_measures import summary


def evaluate_agent(
    agent: Any,
    env: gym.Env,
    n_episodes: int = 5,
) -> dict[str, Any]:
    """Evaluates an RL agent on an environment over n_episodes.

    Parameters
    ----------
    agent : Any
        Trained RL agent exposing .predict(obs, deterministic=True).
    env : gym.Env
        Gymnasium evaluation environment.
    n_episodes : int, default 5
        Number of evaluation episodes to run.

    Returns
    -------
    dict[str, Any]
        Dictionary with episode metrics, rewards, equity trajectories,
        and Prompt 11 summary risk measures (Sharpe, Sortino, VaR, CVaR, etc.).
    """
    episode_rewards: list[float] = []
    episode_returns: list[float] = []
    all_period_returns: list[float] = []
    final_equities: list[float] = []
    equity_curves: list[list[float]] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_equities = [float(info.get("equity", 100_000.0))]
        initial_eq = ep_equities[0]

        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            ep_reward += float(reward)
            ret_val = float(step_info.get("return", 0.0))
            all_period_returns.append(ret_val)
            ep_equities.append(float(step_info.get("equity", ep_equities[-1])))

        episode_rewards.append(ep_reward)
        final_eq = ep_equities[-1]
        final_equities.append(final_eq)
        ep_return = (final_eq - initial_eq) / initial_eq if initial_eq > 0 else 0.0
        episode_returns.append(ep_return)
        equity_curves.append(ep_equities)

    # Compute risk metrics across all evaluated returns
    returns_arr = np.asarray(all_period_returns, dtype=np.float64)
    if len(returns_arr) > 1 and np.std(returns_arr) > 1e-12:
        risk_summary = summary(returns_arr)
    else:
        risk_summary = {
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "cvar_95": 0.0,
            "max_drawdown": 0.0,
        }

    return {
        "n_episodes": n_episodes,
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_episode_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_final_equity": float(np.mean(final_equities)) if final_equities else 0.0,
        "episode_rewards": episode_rewards,
        "episode_returns": episode_returns,
        "equity_curves": equity_curves,
        "risk_summary": risk_summary,
    }
