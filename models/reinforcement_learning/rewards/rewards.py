"""
Reinforcement learning reward functions for algorithmic trading.

Includes:
- Differential Sharpe Ratio (Moody & Saffell, 2001) with exponential moving variance.
- Sortino Ratio reward over trailing windows.
- Maximum drawdown penalty.
- Transaction cost and slippage execution penalty.
- Composite reward aggregator.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import numpy as np


class DifferentialSharpe:
    """Online recursive Differential Sharpe Ratio reward (Moody & Saffell, 2001).

    Maintains exponential moving averages of the first and second moments:
        A_t = A_{t-1} + eta * dA_t, where dA_t = R_t - A_{t-1}
        B_t = B_{t-1} + eta * dB_t, where dB_t = R_t^2 - B_{t-1}

    Differential Sharpe gradient:
        D_t = (B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t) / (B_{t-1} - A_{t-1}^2)^{1.5}

    Parameters
    ----------
    eta : float, default 0.01
        Exponential moving average decay factor in (0, 1].
    eps : float, default 1e-12
        Small tolerance for denominator variance check to avoid division by zero.
    """

    def __init__(self, eta: float = 0.01, eps: float = 1e-12) -> None:
        if not (0.0 < eta <= 1.0):
            raise ValueError(f"eta must be in (0, 1], got {eta}")
        self.eta = float(eta)
        self.eps = float(eps)
        self.A: float = 0.0
        self.B: float = 0.0

    def reset(self, A: float = 0.0, B: float = 0.0) -> None:
        """Resets the online first and second moment estimators."""
        self.A = float(A)
        self.B = float(B)

    def calculate(self, R_t: float) -> float:
        """Calculates D_t for current return R_t and updates online state (A, B).

        Parameters
        ----------
        R_t : float
            Period portfolio return.

        Returns
        -------
        float
            Differential Sharpe ratio reward.
        """
        r = float(R_t)
        dA = r - self.A
        dB = (r ** 2) - self.B

        denom_var = self.B - (self.A ** 2)
        if denom_var <= self.eps:
            D_t = 0.0
        else:
            denom = denom_var ** 1.5
            D_t = (self.B * dA - 0.5 * self.A * dB) / denom

        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(D_t)

    def __call__(self, R_t: float) -> float:
        return self.calculate(R_t)


def sortino_reward(
    returns_window: Union[Sequence[float], np.ndarray],
    mar: float = 0.0,
    eps: float = 1e-8,
) -> float:
    """Calculates Sortino ratio reward over a trailing returns window.

    Sortino = (mean(R) - mar) / (downside_std + eps)

    Parameters
    ----------
    returns_window : sequence or np.ndarray
        Historical trailing returns.
    mar : float, default 0.0
        Minimum acceptable return.
    eps : float, default 1e-8
        Numerical stability epsilon.

    Returns
    -------
    float
        Sortino ratio reward.
    """
    arr = np.asarray(returns_window, dtype=np.float64)
    if len(arr) == 0:
        return 0.0

    mean_ret = float(np.mean(arr))
    excess = arr - mar
    downside = np.minimum(excess, 0.0)
    downside_var = float(np.mean(downside ** 2))
    downside_std = np.sqrt(downside_var)

    return float((mean_ret - mar) / (downside_std + eps))


def drawdown_penalty(peak: float, equity: float, lam: float = 1.0) -> float:
    """Calculates penalty for current equity drawdown from peak.

    Penalty = -lam * max(0, (peak - equity) / peak)

    Parameters
    ----------
    peak : float
        Peak portfolio equity.
    equity : float
        Current portfolio equity.
    lam : float, default 1.0
        Penalty scaling coefficient.

    Returns
    -------
    float
        Non-positive drawdown penalty value.
    """
    if peak <= 0.0:
        return 0.0
    dd = max(0.0, (peak - equity) / peak)
    return float(-lam * dd)


def cost_penalty(cost_paid: float, equity: float, lam: float = 1.0) -> float:
    """Calculates explicit transaction cost penalty normalized by equity.

    Penalty = -lam * (cost_paid / equity)

    Parameters
    ----------
    cost_paid : float
        Total transaction fees, commissions, and price impact paid in currency units.
    equity : float
        Current portfolio equity.
    lam : float, default 1.0
        Penalty scaling coefficient.

    Returns
    -------
    float
        Non-positive cost penalty value.
    """
    if equity <= 0.0 or cost_paid <= 0.0:
        return 0.0
    return float(-lam * (cost_paid / equity))


class CompositeReward:
    """Combines a base return/Sharpe reward with drawdown and execution cost penalties.

    Total Reward = base_reward + drawdown_penalty + cost_penalty
    """

    def __init__(
        self,
        base_reward_fn: Optional[Callable[..., float]] = None,
        lam_drawdown: float = 1.0,
        lam_cost: float = 1.0,
    ) -> None:
        self.base_reward_fn = base_reward_fn
        self.lam_drawdown = float(lam_drawdown)
        self.lam_cost = float(lam_cost)

    def reset(self) -> None:
        """Resets the inner base reward function if it maintains state."""
        if self.base_reward_fn is not None and hasattr(self.base_reward_fn, "reset"):
            self.base_reward_fn.reset()

    def calculate(
        self,
        return_t: float,
        equity: float,
        peak: float,
        cost_paid: float,
    ) -> float:
        """Calculates aggregate composite reward.

        Parameters
        ----------
        return_t : float
            Period portfolio return.
        equity : float
            Current portfolio equity.
        peak : float
            Peak historical equity.
        cost_paid : float
            Execution fees and impact paid in currency units.

        Returns
        -------
        float
            Total combined reward.
        """
        if self.base_reward_fn is not None:
            base = float(self.base_reward_fn(return_t))
        else:
            base = float(return_t)

        dd_pen = drawdown_penalty(peak=peak, equity=equity, lam=self.lam_drawdown)
        c_pen = cost_penalty(cost_paid=cost_paid, equity=equity, lam=self.lam_cost)

        return float(base + dd_pen + c_pen)

    def __call__(
        self,
        return_t: float,
        equity: float,
        peak: float,
        cost_paid: float,
    ) -> float:
        return self.calculate(return_t=return_t, equity=equity, peak=peak, cost_paid=cost_paid)
