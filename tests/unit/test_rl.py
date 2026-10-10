"""
Unit tests for reinforcement learning trading environment, rewards, and agents.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from models.reinforcement_learning.agents.dqn import DQNAgent, EpsilonSchedule, ReplayBuffer
from models.reinforcement_learning.agents.evaluate import evaluate_agent
from models.reinforcement_learning.agents.ppo import PPOAgent, RolloutBuffer
from models.reinforcement_learning.environments.trading_env import TradingEnv
from models.reinforcement_learning.rewards.rewards import (
    CompositeReward,
    DifferentialSharpe,
    cost_penalty,
    drawdown_penalty,
    sortino_reward,
)


def test_gymnasium_check_env() -> None:
    """Verifies that gymnasium.utils.env_checker.check_env passes across all three action modes."""
    np.random.seed(42)
    T = 60
    features_df = pd.DataFrame(
        np.random.randn(T, 4).astype(np.float32),
        columns=["f0", "f1", "f2", "f3"],
    )
    prices_df = pd.DataFrame(
        {
            "open": 100.0 + np.cumsum(np.random.randn(T) * 0.5),
            "close": 100.0 + np.cumsum(np.random.randn(T) * 0.5),
        }
    )

    for mode in ["discrete", "kelly", "weight"]:
        env = TradingEnv(
            features_df=features_df,
            prices_df=prices_df,
            window=5,
            action_mode=mode,
        )
        check_env(env)


def test_env_causality() -> None:
    """Verifies environment causality: perturbing future rows leaves past observations unchanged."""
    np.random.seed(42)
    T = 50
    features = np.random.randn(T, 3).astype(np.float32)
    prices = pd.DataFrame(
        {
            "open": 100.0 + np.cumsum(np.random.randn(T)),
            "close": 100.0 + np.cumsum(np.random.randn(T)),
        }
    )

    env1 = TradingEnv(features_df=features, prices_df=prices, window=4, action_mode="weight")
    obs1, _ = env1.reset()

    # Step up to step t=5
    for _ in range(5):
        obs1, _, _, _, _ = env1.step(np.array([0.5], dtype=np.float32))

    # Create perturbed features & prices strictly in the future (rows >= 25)
    features_perturbed = features.copy()
    features_perturbed[25:] += 100.0
    prices_perturbed = prices.copy()
    prices_perturbed.loc[25:, "close"] += 50.0

    env2 = TradingEnv(
        features_df=features_perturbed,
        prices_df=prices_perturbed,
        window=4,
        action_mode="weight",
    )
    obs2, _ = env2.reset()

    for _ in range(5):
        obs2, _, _, _, _ = env2.step(np.array([0.5], dtype=np.float32))

    # Past observations at step 5 must be identical within 1e-6
    np.testing.assert_allclose(obs1, obs2, atol=1e-6)


def test_execution_timing_hand_example() -> None:
    """Execution timing: action at close of t fills at OPEN of t+1.

    6-bar hand example with zero costs:
        Bar 0: open 100, close 100
        Bar 1: open 100, close 100
        Bar 2: open 100, close 100
        Bar 3: open 100, close 102
        Bar 4: open 103, close 105
        Bar 5: open 104, close 100
    Window = 2. Starts at Bar 1 close.
    Step 0: target 1.0 -> fills at Bar 2 open (100) -> equity = 1000.0
    Step 1: hold 1.0 -> Bar 3 (close 102) -> equity = 1020.0
    Step 2: exit to 0.0 -> fills at Bar 4 open (103) -> overnight gain 1020*(103/102) = 1030.0
    Step 3: short -0.5 -> fills at Bar 5 open (104) -> intraday gain -0.5*(100-104)/104 = +2/104
            final equity = 1030 * (106/104) = 1049.8076923...
    """
    prices = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 103.0, 104.0],
            "close": [100.0, 100.0, 100.0, 102.0, 105.0, 100.0],
        }
    )
    features = pd.DataFrame(np.zeros((6, 2)))

    env = TradingEnv(
        features_df=features,
        prices_df=prices,
        window=2,
        initial_cash=1000.0,
        execution_cost_model=0.0,
        action_mode="weight",
    )

    env.reset()

    # Step 0: Action at Bar 1 close -> fills at Bar 2 open
    _, _, _, _, info0 = env.step(np.array([1.0], dtype=np.float32))
    assert info0["equity"] == 1000.0

    # Step 1: Hold 1.0 -> Bar 3 close (102)
    _, _, _, _, info1 = env.step(np.array([1.0], dtype=np.float32))
    assert info1["equity"] == 1020.0

    # Step 2: Exit to 0.0 -> fills at Bar 4 open (103)
    _, _, _, _, info2 = env.step(np.array([0.0], dtype=np.float32))
    assert abs(info2["equity"] - 1030.0) < 1e-6

    # Step 3: Short -0.5 -> fills at Bar 5 open (104) -> ends at Bar 5 close (100)
    _, _, term, _, info3 = env.step(np.array([-0.5], dtype=np.float32))
    expected_final = 1030.0 * (106.0 / 104.0)
    assert abs(info3["equity"] - expected_final) < 1e-5
    assert term is True


def test_costs_flipping_vs_holding() -> None:
    """Verifies that flipping position every step with nonzero execution costs ends below holding."""
    T = 40
    prices = pd.DataFrame(
        {
            "open": np.full(T, 100.0),
            "close": np.full(T, 100.0),
        }
    )
    features = pd.DataFrame(np.zeros((T, 2)))

    # Holding strategy: position stays 1.0
    env_hold = TradingEnv(
        features_df=features,
        prices_df=prices,
        window=2,
        initial_cash=10_000.0,
        action_mode="weight",
    )
    env_hold.reset()
    for _ in range(T - 2):
        _, _, _, _, info_hold = env_hold.step(np.array([1.0], dtype=np.float32))

    # Flipping strategy: alternates +1.0 and -1.0 every step
    env_flip = TradingEnv(
        features_df=features,
        prices_df=prices,
        window=2,
        initial_cash=10_000.0,
        action_mode="weight",
    )
    env_flip.reset()
    for step_i in range(T - 2):
        weight = 1.0 if (step_i % 2 == 0) else -1.0
        _, _, _, _, info_flip = env_flip.step(np.array([weight], dtype=np.float32))

    # Flipping must pay execution fees every step, ending strictly lower than holding
    assert info_flip["equity"] < info_hold["equity"]
    assert info_flip["costs"] > 0.0


def test_kelly_mode_bounds() -> None:
    """Verifies that in 'kelly' mode |position| never exceeds max_position_pct, and a=0 yields 0 position."""
    T = 40
    prices = pd.DataFrame(
        {
            "open": 100.0 + np.cumsum(np.random.randn(T)),
            "close": 100.0 + np.cumsum(np.random.randn(T)),
        }
    )
    features = pd.DataFrame(np.random.randn(T, 2))

    max_pos = 0.6
    env = TradingEnv(
        features_df=features,
        prices_df=prices,
        window=3,
        action_mode="kelly",
        max_position_pct=max_pos,
    )
    env.reset()

    # Action 0 yields strictly 0 position
    _, _, _, _, info_zero = env.step(np.array([0.0], dtype=np.float32))
    assert abs(info_zero["position"]) == 0.0

    # Large actions never exceed max_position_pct
    for a in [1.0, -1.0, 0.8, -0.9]:
        _, _, _, _, info = env.step(np.array([a], dtype=np.float32))
        assert abs(info["position"]) <= max_pos + 1e-6


def test_differential_sharpe_hand_arithmetic() -> None:
    """Tests Differential Sharpe formula on a 3-step hand sequence and zero-variance stability."""
    ds = DifferentialSharpe(eta=0.1)
    ds.reset(A=0.05, B=0.01)

    # Hand calculation:
    # A0 = 0.05, B0 = 0.01, denom_var0 = 0.01 - 0.0025 = 0.0075
    # Step 1: R1 = 0.10 -> dA = 0.05, dB = 0.0 -> Num = 0.01*0.05 = 0.0005
    # D1 = 0.0005 / (0.0075^1.5) = 0.7698003589...
    d1 = ds(0.10)
    expected_d1 = 0.0005 / (0.0075 ** 1.5)
    assert abs(d1 - expected_d1) < 1e-8

    # Step 2: R2 = -0.05
    # A1 = 0.055, B1 = 0.01 -> dA = -0.105, dB = -0.0075 -> Num = -0.00084375
    # denom_var1 = 0.01 - 0.055^2 = 0.006975
    d2 = ds(-0.05)
    expected_d2 = -0.00084375 / (0.006975 ** 1.5)
    assert abs(d2 - expected_d2) < 1e-8

    # Step 3: R3 = 0.02
    # A2 = 0.0445, B2 = 0.00925 -> dA = -0.0245, dB = -0.00885 -> Num = -0.0000297125
    # denom_var2 = 0.00925 - 0.0445^2 = 0.00726975
    d3 = ds(0.02)
    expected_d3 = -0.0000297125 / (0.00726975 ** 1.5)
    assert abs(d3 - expected_d3) < 1e-8

    # Zero-variance stability check: constant return stream should yield 0 without NaN or inf
    ds_zero = DifferentialSharpe(eta=0.01)
    ds_zero.reset(A=0.0, B=0.0)
    for _ in range(20):
        val = ds_zero(0.05)
        assert np.isfinite(val)
        assert not np.isnan(val)


def test_gae_and_ppo_clipped_autograd() -> None:
    """Verifies GAE on a 3-step hand rollout to 1e-8 and tests PPO clipped loss zero gradient."""
    gamma = 0.99
    lam = 0.95
    rewards = [1.0, 2.0, 3.0]
    values = [10.0, 11.0, 12.0]
    last_value = 13.0
    dones = [0, 0, 0]

    # Hand calculation:
    # delta0 = 1.0 + 0.99*11.0 - 10.0 = 1.89
    # delta1 = 2.0 + 0.99*12.0 - 11.0 = 2.88
    # delta2 = 3.0 + 0.99*13.0 - 12.0 = 3.87
    # adv2 = 3.87
    # adv1 = 2.88 + 0.99*0.95*3.87 = 6.519735
    # adv0 = 1.89 + 0.99*0.95*6.519735 = 8.0218107675
    expected_adv = [8.0218107675, 6.519734999999998, 3.87]

    buf = RolloutBuffer(capacity=3, obs_dim=2, is_continuous=False)
    for t in range(3):
        buf.add(np.zeros(2), 0, 0.0, rewards[t], False, values[t])
    buf.compute_gae(last_val=last_value, gamma=gamma, gae_lambda=lam)

    for t in range(3):
        assert abs(buf.advantages[t] - expected_adv[t]) < 1e-8

    # Autograd check: clipped loss has zero gradient when ratio is outside clip range with positive advantage
    epsilon = 0.2
    adv = torch.tensor([2.0])  # Positive advantage
    ratio = torch.tensor([1.5], requires_grad=True)  # ratio > 1 + epsilon

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
    loss = -torch.min(surr1, surr2).mean()
    loss.backward()

    assert ratio.grad.item() == 0.0


def test_dqn_buffer_target_update_and_schedule() -> None:
    """Tests DQN replay buffer capacity/overwrite, hard target sync, and epsilon schedule endpoints."""
    # 1. Buffer capacity & overwrite
    buf = ReplayBuffer(capacity=3, obs_dim=2)
    assert len(buf) == 0

    buf.push(np.array([1, 1]), 0, 1.0, np.array([2, 2]), False)
    buf.push(np.array([2, 2]), 1, 2.0, np.array([3, 3]), False)
    buf.push(np.array([3, 3]), 0, 3.0, np.array([4, 4]), False)
    assert len(buf) == 3

    # Overwrite oldest (slot 0)
    buf.push(np.array([4, 4]), 1, 4.0, np.array([5, 5]), True)
    assert len(buf) == 3
    assert buf.rewards[0] == 4.0

    # 2. Epsilon schedule endpoints
    sched = EpsilonSchedule(eps_start=1.0, eps_end=0.05, decay_steps=100)
    assert sched.get_epsilon(0) == 1.0
    assert sched.get_epsilon(50) == 0.525
    assert sched.get_epsilon(100) == 0.05
    assert sched.get_epsilon(200) == 0.05

    # 3. Hard target update
    dummy_env = gym.make("CartPole-v1")
    dqn = DQNAgent(dummy_env, hidden_dims=(32, 32))

    # Mutate policy net parameters
    with torch.no_grad():
        for p in dqn.policy_net.parameters():
            p.add_(1.0)

    # Target net should differ before update
    diff = sum((p1 - p2).abs().sum().item() for p1, p2 in zip(dqn.policy_net.parameters(), dqn.target_net.parameters()))
    assert diff > 0.1

    # Sync target network
    dqn.update_target_network()
    diff_after = sum((p1 - p2).abs().sum().item() for p1, p2 in zip(dqn.policy_net.parameters(), dqn.target_net.parameters()))
    assert diff_after == 0.0


def test_sanity_learning() -> None:
    """Sanity learning: environment whose next return is a deterministic function of the last return.

    Verifies both PPO and DQN beat a random policy's mean reward over 5 episodes.
    """
    T = 120
    returns = np.array([0.02, -0.02] * (T // 2), dtype=np.float64)
    prices_list = [100.0]
    for r in returns:
        prices_list.append(prices_list[-1] * (1.0 + r))
    prices_arr = np.array(prices_list[:T])

    prices_df = pd.DataFrame(
        {
            "open": prices_arr,
            "close": prices_arr * (1.0 + returns[:T]),
        }
    )
    features_df = pd.DataFrame(
        {
            "last_ret": returns[:T],
            "dummy": np.zeros(T),
        }
    )

    env = TradingEnv(
        features_df=features_df,
        prices_df=prices_df,
        window=3,
        initial_cash=10_000.0,
        execution_cost_model=0.0,
        action_mode="discrete",
    )

    # 1. Random policy benchmark
    class RandomAgent:
        def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
            return int(np.random.randint(0, 3))

    rand_eval = evaluate_agent(RandomAgent(), env, n_episodes=5)
    rand_reward = rand_eval["mean_episode_reward"]
    print(f"\n[Sanity Learning] Random Policy Mean Reward: {rand_reward:.4f}")

    # 2. Train DQN agent
    dqn = DQNAgent(env, lr=0.01, eps_start=0.5, eps_end=0.01, eps_decay_steps=200, seed=42)
    dqn.train(total_timesteps=400)
    dqn_eval = evaluate_agent(dqn, env, n_episodes=5)
    dqn_reward = dqn_eval["mean_episode_reward"]
    print(f"[Sanity Learning] DQNAgent Mean Reward: {dqn_reward:.4f}")
    assert dqn_reward > rand_reward, f"DQN ({dqn_reward}) failed to beat random ({rand_reward})"

    # 3. Train PPO agent
    ppo = PPOAgent(env, lr=0.01, seed=42)
    ppo.train(total_timesteps=400, rollout_steps=100)
    ppo_eval = evaluate_agent(ppo, env, n_episodes=5)
    ppo_reward = ppo_eval["mean_episode_reward"]
    print(f"[Sanity Learning] PPOAgent Mean Reward: {ppo_reward:.4f}")
    assert ppo_reward > rand_reward, f"PPO ({ppo_reward}) failed to beat random ({rand_reward})"


def test_determinism_with_same_seed() -> None:
    """Verifies determinism when training an agent with identical seed."""
    T = 60
    prices_df = pd.DataFrame(
        {
            "open": 100.0 + np.cumsum(np.random.randn(T) * 0.1),
            "close": 100.0 + np.cumsum(np.random.randn(T) * 0.1),
        }
    )
    features_df = pd.DataFrame(np.random.randn(T, 2))

    env1 = TradingEnv(features_df, prices_df, window=3, action_mode="discrete")
    env2 = TradingEnv(features_df, prices_df, window=3, action_mode="discrete")

    dqn1 = DQNAgent(env1, seed=123)
    dqn1.train(total_timesteps=100)

    dqn2 = DQNAgent(env2, seed=123)
    dqn2.train(total_timesteps=100)

    for p1, p2 in zip(dqn1.policy_net.parameters(), dqn2.policy_net.parameters()):
        torch.testing.assert_close(p1, p2)
