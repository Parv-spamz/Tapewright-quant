"""
OpenAI Gymnasium trading environment with microstructure-realistic execution.

Simulates discrete-time multi-bar execution where:
- Action decided at the close of bar t fills at the OPEN of bar t+1.
- Execution incorporates slippage, market impact, and transaction commissions.
- Observation includes causal standardized historical features, current position,
  unrealized PnL fraction, and trailing return volatility.
- Supports three action spaces: 'discrete', 'kelly', and 'weight'.
- Compatible with gymnasium.utils.env_checker.check_env.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from backtesting.metrics.risk_measures.sizing import half_kelly_fraction
from models.reinforcement_learning.rewards.rewards import (
    CompositeReward,
    DifferentialSharpe,
)


class TradingEnv(gym.Env):
    """Custom Gymnasium trading environment for quantitative execution & RL.

    Parameters
    ----------
    features_df : pd.DataFrame or np.ndarray
        Historical feature matrix of shape [T, F].
    prices_df : pd.DataFrame
        Asset prices DataFrame with 'open' and 'close' columns of length T.
    window : int, default 10
        Lookback window length for feature history in observations.
    initial_cash : float, default 100_000.0
        Starting portfolio capital in currency units.
    execution_cost_model : Any, optional
        Cost model or numeric fee. If None, uses default 5 bps commission + 5 bps impact.
        If 0 or 0.0, zero costs are incurred.
    reward : Any, optional
        Reward calculator or callable. Defaults to DifferentialSharpe(eta=0.01).
    action_mode : str, default 'discrete'
        Action mode: 'discrete', 'kelly', or 'weight'.
    max_position_pct : float, default 1.0
        Maximum allowable absolute position leverage / weight.
    max_drawdown_limit : float, default 0.5
        Terminates episode if equity drops below (1 - max_drawdown_limit) * peak_equity.
    half_kelly_window : int, default 20
        Lookback window of past asset returns for trailing Half-Kelly estimation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features_df: Union[pd.DataFrame, np.ndarray],
        prices_df: pd.DataFrame,
        window: int = 10,
        initial_cash: float = 100_000.0,
        execution_cost_model: Any = None,
        reward: Any = None,
        action_mode: str = "discrete",
        max_position_pct: float = 1.0,
        max_drawdown_limit: float = 0.5,
        half_kelly_window: int = 20,
    ) -> None:
        super().__init__()

        if isinstance(features_df, pd.DataFrame):
            self.features = features_df.values.astype(np.float32)
        else:
            self.features = np.asarray(features_df, dtype=np.float32)

        if not isinstance(prices_df, pd.DataFrame):
            raise TypeError("prices_df must be a pandas DataFrame with 'open' and 'close' columns.")

        cols_lower = {str(c).lower(): c for c in prices_df.columns}
        if "open" not in cols_lower or "close" not in cols_lower:
            raise ValueError("prices_df must contain 'open' and 'close' columns.")

        self.opens = prices_df[cols_lower["open"]].values.astype(np.float64)
        self.closes = prices_df[cols_lower["close"]].values.astype(np.float64)
        self.T_bars = len(self.closes)

        if len(self.features) != self.T_bars:
            raise ValueError(
                f"Features length ({len(self.features)}) != prices length ({self.T_bars})"
            )

        self.window = int(window)
        if self.T_bars <= self.window:
            raise ValueError(f"Total bars ({self.T_bars}) must exceed window ({self.window})")

        self.initial_cash = float(initial_cash)
        self.execution_cost_model = execution_cost_model
        self.max_position_pct = float(max_position_pct)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.half_kelly_window = int(half_kelly_window)
        self.action_mode = str(action_mode).lower()

        # Set up reward function
        if reward is None:
            self.reward_fn = DifferentialSharpe(eta=0.01)
        else:
            self.reward_fn = reward

        # Setup action space
        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(3)  # 0: -1, 1: 0, 2: +1
        elif self.action_mode in ("kelly", "weight"):
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        # Setup observation space: flattened window features + position + unrealized_pnl + trailing_vol
        self.n_features = self.features.shape[1]
        self.obs_dim = self.window * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Dynamic state variables
        self.current_step: int = self.window - 1
        self.equity: float = self.initial_cash
        self.peak_equity: float = self.initial_cash
        self.position: float = 0.0
        self.entry_price: float = float(self.closes[self.current_step])

    def _estimate_half_kelly(self) -> float:
        """Estimates Half-Kelly optimal leverage strictly from past historical returns."""
        tau = self.current_step
        start = max(1, tau - self.half_kelly_window)
        if tau <= start:
            return self.max_position_pct

        past_closes = self.closes[start : tau + 1]
        past_returns = np.diff(past_closes) / past_closes[:-1]
        if len(past_returns) < 2:
            return self.max_position_pct

        mu = float(np.mean(past_returns))
        sigma = float(np.std(past_returns, ddof=1))
        if sigma <= 1e-8:
            return self.max_position_pct

        k = half_kelly_fraction(mu=mu, sigma=sigma, cap=self.max_position_pct, signed=False)
        return float(k) if k > 0.0 else 0.0

    def _calculate_costs(self, notional: float, shares: float, price: float) -> float:
        """Calculates transaction costs, commission, and slippage for a rebalance trade."""
        if notional <= 1e-8 or shares <= 1e-8:
            return 0.0

        if self.execution_cost_model is not None and self.execution_cost_model == 0:
            return 0.0

        if self.execution_cost_model is not None:
            # Custom cost model
            model = self.execution_cost_model
            cost = 0.0
            if hasattr(model, "calculate"):
                cost += float(model.calculate(side="BUY", quantity=shares, price=price))
            if hasattr(model, "calculate_impact"):
                impact = float(model.calculate_impact(side="BUY", quantity=shares, price=price))
                cost += float(abs(impact) * shares)
            if callable(model) and not hasattr(model, "calculate"):
                cost += float(model(notional=notional, shares=shares, price=price))
            if cost > 0.0:
                return cost

        # Default realistic institutional cost: 5 bps commission + 5 bps square root impact
        commission = notional * 0.0005
        # Square-root Kyle impact: c * sqrt(shares / 100_000) * price * shares
        adv = 1_000_000.0
        impact_frac = 0.05 * np.sqrt(shares / adv)
        impact_cost = impact_frac * notional
        return float(commission + impact_cost)

    def _get_observation(self) -> np.ndarray:
        """Generates observation vector strictly using expanding past data up to current bar."""
        tau = self.current_step

        # Causal expanding standardizer: mean and std strictly computed from rows 0..tau
        hist_features = self.features[: tau + 1]
        mean = np.mean(hist_features, axis=0, keepdims=True)
        std = np.std(hist_features, axis=0, keepdims=True) + 1e-8

        # Extract current lookback slice
        feature_slice = self.features[tau - self.window + 1 : tau + 1]
        norm_slice = (feature_slice - mean) / std
        flat_feats = norm_slice.flatten().astype(np.float32)

        # Unrealized PnL fraction
        curr_price = float(self.closes[tau])
        if abs(self.position) > 1e-8 and self.entry_price > 0.0:
            unrealized_pnl = float(self.position * (curr_price - self.entry_price) / self.entry_price)
        else:
            unrealized_pnl = 0.0

        # Trailing volatility of returns
        vol_start = max(1, tau - self.window)
        if tau > vol_start:
            window_closes = self.closes[vol_start : tau + 1]
            rets = np.diff(window_closes) / window_closes[:-1]
            trailing_vol = float(np.std(rets)) if len(rets) > 0 else 0.0
        else:
            trailing_vol = 0.0

        state_extras = np.array(
            [self.position, unrealized_pnl, trailing_vol],
            dtype=np.float32,
        )
        obs = np.concatenate([flat_feats, state_extras]).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment to the beginning of the lookback period."""
        super().reset(seed=seed)

        self.current_step = self.window - 1
        self.equity = float(self.initial_cash)
        self.peak_equity = float(self.initial_cash)
        self.position = 0.0
        self.entry_price = float(self.closes[self.current_step])

        if hasattr(self.reward_fn, "reset") and callable(self.reward_fn.reset):
            self.reward_fn.reset()

        obs = self._get_observation()
        info = {
            "equity": self.equity,
            "position": self.position,
            "costs": 0.0,
        }
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Executes one step in the environment.

        The action taken at the close of bar tau executes at the OPEN of bar tau+1.
        """
        tau = self.current_step
        if tau >= self.T_bars - 1:
            obs = self._get_observation()
            return obs, 0.0, True, False, {"equity": self.equity, "position": self.position, "costs": 0.0}

        # 1. Resolve action into target position fraction w_target
        if self.action_mode == "discrete":
            act = int(action)
            direction = float(act - 1)  # 0 -> -1, 1 -> 0, 2 -> +1
            kelly_frac = self._estimate_half_kelly()
            w_target = direction * min(kelly_frac, self.max_position_pct)
        elif self.action_mode == "kelly":
            act_val = float(action[0] if isinstance(action, (np.ndarray, list)) else action)
            act_val = float(np.clip(act_val, -1.0, 1.0))
            if abs(act_val) <= 1e-8:
                w_target = 0.0
            else:
                sign = float(np.sign(act_val))
                conf = abs(act_val)
                kelly_frac = self._estimate_half_kelly()
                w_target = sign * min(conf * kelly_frac, self.max_position_pct)
        elif self.action_mode == "weight":
            act_val = float(action[0] if isinstance(action, (np.ndarray, list)) else action)
            w_target = float(np.clip(act_val, -self.max_position_pct, self.max_position_pct))
        else:
            w_target = 0.0

        # 2. Execution timing: action at close of bar tau executes at OPEN of bar tau+1
        next_open = float(self.opens[tau + 1])
        next_close = float(self.closes[tau + 1])
        curr_close = float(self.closes[tau])

        # Gap return overnight from close of tau to open of tau+1 on prior position
        r_gap = (next_open - curr_close) / curr_close
        equity_open = self.equity * (1.0 + self.position * r_gap)

        # Rebalance at Open of tau+1
        delta_w = w_target - self.position
        if abs(delta_w) > 1e-8:
            trade_notional = abs(delta_w) * equity_open
            shares_traded = trade_notional / next_open
            costs = self._calculate_costs(notional=trade_notional, shares=shares_traded, price=next_open)
            if abs(self.position) <= 1e-8 and abs(w_target) > 1e-8:
                self.entry_price = next_open
            elif abs(w_target) <= 1e-8:
                self.entry_price = next_open
        else:
            costs = 0.0

        # Equity after cost at Open of tau+1
        equity_after_cost = max(1e-8, equity_open - costs)

        # Intraday return from Open to Close of tau+1 on new position w_target
        r_intraday = (next_close - next_open) / next_open
        equity_close = equity_after_cost * (1.0 + w_target * r_intraday)

        # Step net return for portfolio
        step_net_return = (equity_close - self.equity) / self.equity

        # Update environment state
        self.equity = max(1e-8, float(equity_close))
        self.peak_equity = max(self.peak_equity, self.equity)
        self.position = float(w_target)
        self.current_step += 1

        # 3. Reward calculation
        if isinstance(self.reward_fn, CompositeReward):
            reward_val = self.reward_fn(
                return_t=step_net_return,
                equity=self.equity,
                peak=self.peak_equity,
                cost_paid=costs,
            )
        elif callable(self.reward_fn):
            try:
                reward_val = float(self.reward_fn(step_net_return))
            except TypeError:
                reward_val = float(self.reward_fn(step_net_return, self.equity, self.peak_equity, costs))
        else:
            reward_val = float(step_net_return)

        # 4. Termination checks
        terminated = False
        truncated = False

        if self.current_step >= self.T_bars - 1:
            terminated = True

        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd >= self.max_drawdown_limit:
            terminated = True

        obs = self._get_observation()
        info = {
            "equity": self.equity,
            "position": self.position,
            "costs": costs,
            "return": step_net_return,
        }

        return obs, float(reward_val), bool(terminated), bool(truncated), info
