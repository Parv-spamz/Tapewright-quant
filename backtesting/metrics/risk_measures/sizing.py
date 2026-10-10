from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from backtesting.engine.event_loop.events import SignalEvent, TickEvent
from backtesting.engine.event_loop.strategy_base import Strategy
from backtesting.metrics.risk_measures.measures import cvar
from config.trading.risk_limits import RiskLimits, vol_scaling_factor


def half_kelly_fraction(
    mu: float,
    sigma: float,
    rf: float = 0.0,
    kelly_fraction: float = 0.5,
    cap: float = 1.0,
    signed: bool = False,
) -> float:
    """Calculate Half-Kelly optimal position fraction.

    Formula:
        f* = kelly_fraction * (mu - rf) / sigma^2

    In long-only mode (signed=False), non-positive fractions are clipped to 0.0,
    and positive fractions are bounded by cap.
    In shorting variant (signed=True), returns negative fractions bounded by -cap.

    Args:
        mu: Expected per-period return.
        sigma: Standard deviation of per-period returns.
        rf: Risk-free rate per period. Default is 0.0.
        kelly_fraction: Kelly multiplier (0.5 for Half-Kelly). Default is 0.5.
        cap: Maximum position fraction cap. Default is 1.0.
        signed: If True, allows short positions for negative excess returns.

    Returns:
        float: Sized position fraction.
    """
    if sigma <= 1e-12:
        return 0.0

    raw = kelly_fraction * (mu - rf) / (sigma ** 2)

    if not signed:
        if raw <= 0.0:
            return 0.0
        return float(min(raw, cap))
    else:
        return float(np.clip(raw, -cap, cap))


def cvar_scaled_fraction(
    fraction: float,
    returns_window: pd.Series | np.ndarray,
    alpha: float = 0.99,
    budget: float = 0.03,
) -> float:
    """Scale position fraction down if trailing portfolio CVaR exceeds risk budget.

    Scale fraction down by min(1, budget / (fraction * CVaR_alpha of the unit-leverage window)).
    CVaR is measured strictly from the past trailing window supplied.

    Args:
        fraction: Candidate position fraction (leverage).
        returns_window: Trailing historical periodic returns.
        alpha: Confidence level for CVaR (e.g. 0.99).
        budget: Maximum permitted loss at alpha level as a fraction of equity (e.g. 0.03 = 3%).

    Returns:
        float: CVaR-budgeted position fraction.
    """
    if abs(fraction) <= 1e-12:
        return 0.0

    cvar_val = cvar(returns_window, alpha=alpha)
    if cvar_val <= 1e-12:
        return float(fraction)

    # Unit-leverage loss * candidate fraction
    portfolio_cvar = abs(fraction) * cvar_val
    if portfolio_cvar > budget:
        scale = budget / portfolio_cvar
        return float(fraction * scale)

    return float(fraction)


def position_size(
    mu: float,
    sigma: float,
    returns_window: pd.Series | np.ndarray,
    limits: RiskLimits,
    current_vol: float | None = None,
    rf: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    """Determine multi-stage risk-budgeted position size and return constraint diagnostics.

    Stages:
        1. Half-Kelly fraction
        2. Volatility scaling (using limits.target_annual_vol / current_vol)
        3. CVaR budget scaling
        4. Bounds enforcement: min(limits.max_position_pct, limits.max_leverage)

    Args:
        mu: Expected per-period return.
        sigma: Return volatility per period.
        returns_window: Trailing window of historical returns.
        limits: Trading RiskLimits configuration.
        current_vol: Current annualized volatility (optional).
        rf: Risk-free rate per period.

    Returns:
        tuple[float, dict[str, Any]]: (final_fraction, diagnostics_dict)
    """
    # Stage 1: Half-Kelly
    f_kelly = half_kelly_fraction(
        mu=mu,
        sigma=sigma,
        rf=rf,
        kelly_fraction=limits.kelly_fraction,
        cap=limits.kelly_cap,
        signed=False,
    )

    # Stage 2: Volatility scaling
    if current_vol is not None and current_vol > 0.0 and f_kelly > 0.0:
        vol_factor = vol_scaling_factor(limits, current_vol)
        f_vol = float(f_kelly * vol_factor)
    else:
        f_vol = f_kelly

    # Stage 3: CVaR budget scaling
    if len(returns_window) > 0 and f_vol > 0.0:
        f_cvar = cvar_scaled_fraction(
            fraction=f_vol,
            returns_window=returns_window,
            alpha=limits.cvar_alpha,
            budget=limits.cvar_budget_pct,
        )
    else:
        f_cvar = f_vol

    # Stage 4: Risk limits ceiling
    limit_bound = min(limits.max_position_pct, limits.max_leverage)
    f_final = float(min(f_cvar, limit_bound))

    # Diagnose binding constraint
    if f_final <= 0.0:
        binding = "kelly"
    elif np.isclose(f_final, limits.max_position_pct) and (f_cvar > limits.max_position_pct):
        binding = "max_position_pct"
    elif np.isclose(f_final, limits.max_leverage) and (f_cvar > limits.max_leverage):
        binding = "max_leverage"
    elif f_cvar < f_vol and np.isclose(f_final, f_cvar):
        binding = "cvar_scaled"
    elif f_vol < f_kelly and np.isclose(f_final, f_vol):
        binding = "vol_scaled"
    else:
        binding = "kelly"

    diagnostics = {
        "kelly": f_kelly,
        "vol_scaled": f_vol,
        "cvar_scaled": f_cvar,
        "max_position_pct": limits.max_position_pct,
        "max_leverage": limits.max_leverage,
        "final": f_final,
        "binding_constraint": binding,
    }

    return f_final, diagnostics


class KellySizedStrategyWrapper(Strategy):
    """Wraps an underlying strategy to rewrite SignalEvent target_pct using causal position sizing."""

    def __init__(
        self,
        strategy: Strategy,
        window: int,
        limits: RiskLimits,
        rf: float = 0.0,
    ) -> None:
        self.strategy = strategy
        self.window = int(window)
        self.limits = limits
        self.rf = rf

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        raw_signals = self.strategy.on_bar(tick, data_handler)
        if not raw_signals:
            return []

        out_signals: list[SignalEvent] = []
        for sig in raw_signals:
            if sig.direction in ("LONG", "SHORT"):
                # Fetch only past bars (strictly no-lookahead)
                bars_df = data_handler.get_latest_bars(sig.symbol, n=self.window + 1)
                if len(bars_df) > 2:
                    closes = bars_df["close"].values
                    rets = np.diff(closes) / closes[:-1]
                    mu = float(np.mean(rets))
                    sigma = float(np.std(rets, ddof=1))
                    curr_vol = float(sigma * np.sqrt(252))

                    sized_fraction, _ = position_size(
                        mu=mu,
                        sigma=sigma,
                        returns_window=rets,
                        limits=self.limits,
                        current_vol=curr_vol,
                        rf=self.rf,
                    )
                    sig.target_pct = sized_fraction
            out_signals.append(sig)

        return out_signals
