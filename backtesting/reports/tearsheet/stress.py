"""
Macro stress testing and scenario analysis module.

Provides:
- Scenario dataclass and BUILTIN_SCENARIOS for historical tail events.
- Portfolio beta estimation via OLS.
- Multi-factor scenario stress testing (PnL shock, correlation override covariance,
  stressed 1-day 99% parametric VaR/CVaR, liquidation costs with haircuts).
- Historical window replay with strict DataUnavailableError enforcement.
- Stress summary table generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats


class DataUnavailableError(Exception):
    """Raised when historical market data for a specified replay window is missing."""


@dataclass
class Scenario:
    """Macroeconomic or market shock stress scenario specification."""

    name: str
    description: str
    equity_shock: float
    vol_multiplier: float
    correlation_override: Optional[float]
    spread_multiplier: float
    adv_haircut: float
    rate_shock_bp: float
    source_note: str


BUILTIN_SCENARIOS: dict[str, Scenario] = {
    "2008 Lehman liquidity freeze": Scenario(
        name="2008 Lehman liquidity freeze",
        description="Global financial crisis liquidity freeze and equity market selloff.",
        equity_shock=-0.45,
        vol_multiplier=3.0,
        correlation_override=0.85,
        spread_multiplier=5.0,
        adv_haircut=0.6,
        rate_shock_bp=0.0,
        source_note="approximate, owner-supplied, verify against your own data before citing",
    ),
    "2020 COVID crash": Scenario(
        name="2020 COVID crash",
        description="Rapid global market shutdown and extreme volatility spike.",
        equity_shock=-0.34,
        vol_multiplier=4.0,
        correlation_override=0.80,
        spread_multiplier=4.0,
        adv_haircut=0.5,
        rate_shock_bp=0.0,
        source_note="approximate, owner-supplied, verify against your own data before citing",
    ),
    "2022 rate-hike shock": Scenario(
        name="2022 rate-hike shock",
        description="Rapid aggressive central bank monetary tightening and duration selloff.",
        equity_shock=-0.25,
        vol_multiplier=1.8,
        correlation_override=0.60,
        spread_multiplier=1.5,
        adv_haircut=0.2,
        rate_shock_bp=250.0,
        source_note="approximate, owner-supplied, verify against your own data before citing",
    ),
}


@dataclass
class StressResult:
    """Quantitative output diagnostics from a scenario stress test."""

    scenario_name: str
    pnl_pct: float
    pnl_dollar: float
    stressed_vol: float
    stressed_var_99_1d: float
    stressed_cvar_99_1d: float
    liquidation_cost: float
    days_to_liquidate: float
    source_note: str


def estimate_betas(
    returns_df: pd.DataFrame,
    benchmark_returns: pd.Series | np.ndarray,
) -> pd.Series:
    """Estimates OLS market beta for each asset against the benchmark return stream.

    beta_i = Cov(R_i, R_m) / Var(R_m)
    """
    bm = pd.Series(benchmark_returns, index=returns_df.index).dropna()
    bm_var = float(bm.var(ddof=1))
    if bm_var <= 1e-12:
        return pd.Series(1.0, index=returns_df.columns)

    betas = {}
    for col in returns_df.columns:
        s = returns_df[col].dropna()
        common = s.index.intersection(bm.index)
        if len(common) > 2:
            cov = float(np.cov(s.loc[common], bm.loc[common])[0, 1])
            betas[col] = cov / bm_var
        else:
            betas[col] = 1.0

    return pd.Series(betas)


def apply_scenario(
    weights: Union[Sequence[float], np.ndarray, pd.Series],
    betas: Union[Sequence[float], np.ndarray, pd.Series],
    scenario: Scenario,
    equity: float = 100_000.0,
    adv: Union[float, Sequence[float], np.ndarray] = 1_000_000.0,
    positions_value: Optional[Union[Sequence[float], np.ndarray]] = None,
    asset_vols: Optional[Union[Sequence[float], np.ndarray]] = None,
    baseline_vol: float = 0.15,
    half_spread: float = 0.0005,
    price: float = 100.0,
) -> StressResult:
    """Evaluates macro stress test impact on a portfolio.

    Parameters
    ----------
    weights : array-like
        Portfolio asset allocation weights [N].
    betas : array-like
        Per-asset market betas [N].
    scenario : Scenario
        Target stress scenario specifications.
    equity : float, default 100_000.0
        Total portfolio equity value.
    adv : float or array-like, default 1_000_000.0
        Average daily volume in shares.
    positions_value : array-like, optional
        Dollar value per asset position. If None, derived as |weights| * equity.
    asset_vols : array-like, optional
        Per-asset annualized volatilities [N]. Defaults to baseline_vol.
    baseline_vol : float, default 0.15
        Baseline annualized portfolio volatility.
    half_spread : float, default 0.0005
        Baseline fractional half-spread (5 bps).
    price : float, default 100.0
        Reference asset share price.

    Returns
    -------
    StressResult
        Stress impact metrics including PnL, stressed covariance VaR/CVaR, and liquidation costs.
    """
    w_arr = np.asarray(weights, dtype=np.float64)
    b_arr = np.asarray(betas, dtype=np.float64)
    N = len(w_arr)

    # 1. Direct equity factor PnL shock
    pnl_pct = float(np.sum(w_arr * b_arr * scenario.equity_shock))
    pnl_dollar = float(pnl_pct * equity)

    # 2. Stressed Covariance & Volatility under correlation override
    if asset_vols is not None:
        vols_arr = np.asarray(asset_vols, dtype=np.float64)
    else:
        vols_arr = np.full(N, baseline_vol, dtype=np.float64)

    # Apply volatility multiplier
    stressed_asset_vols = vols_arr * scenario.vol_multiplier

    # Build constant-correlation covariance matrix
    corr = (
        float(scenario.correlation_override)
        if scenario.correlation_override is not None
        else 0.50
    )
    cov_matrix = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                cov_matrix[i, j] = stressed_asset_vols[i] ** 2
            else:
                cov_matrix[i, j] = corr * stressed_asset_vols[i] * stressed_asset_vols[j]

    port_var = float(w_arr @ cov_matrix @ w_arr)
    stressed_vol = float(np.sqrt(max(1e-12, port_var)))

    # Stressed 1-day 99% parametric VaR and CVaR
    daily_stressed_vol = stressed_vol / np.sqrt(252.0)
    z_99 = 2.3263478740408408  # stats.norm.ppf(0.99)
    stressed_var_99_1d = float(z_99 * daily_stressed_vol)
    # Expected shortfall for standard normal: phi(z) / (1 - alpha)
    phi_z = stats.norm.pdf(z_99)
    stressed_cvar_99_1d = float((phi_z / 0.01) * daily_stressed_vol)

    # 3. Stressed Liquidation Costs & Days to Liquidate
    if positions_value is not None:
        pos_vals = np.asarray(positions_value, dtype=np.float64)
    else:
        pos_vals = np.abs(w_arr) * equity

    if np.isscalar(adv):
        adv_arr = np.full(N, float(adv), dtype=np.float64)
    else:
        adv_arr = np.asarray(adv, dtype=np.float64)

    # Stressed ADV after liquidity haircut
    stressed_adv = np.maximum(1.0, adv_arr * (1.0 - scenario.adv_haircut))
    shares = pos_vals / max(1.0, price)

    # Stressed spread cost
    stressed_spread_cost = np.sum(pos_vals * (half_spread * scenario.spread_multiplier))

    # Stressed square-root market impact: c * sqrt(Q / ADV_stressed) * pos_value
    impact_frac = 0.10 * np.sqrt(shares / stressed_adv)
    stressed_impact_cost = np.sum(impact_frac * pos_vals)
    total_liquidation_cost = float(stressed_spread_cost + stressed_impact_cost)

    # Days to liquidate at 20% participation of stressed ADV
    days_per_asset = shares / (0.20 * stressed_adv)
    days_to_liquidate = float(np.max(days_per_asset)) if len(days_per_asset) > 0 else 0.0

    return StressResult(
        scenario_name=scenario.name,
        pnl_pct=pnl_pct,
        pnl_dollar=pnl_dollar,
        stressed_vol=stressed_vol,
        stressed_var_99_1d=stressed_var_99_1d,
        stressed_cvar_99_1d=stressed_cvar_99_1d,
        liquidation_cost=total_liquidation_cost,
        days_to_liquidate=days_to_liquidate,
        source_note=scenario.source_note,
    )


def replay_historical_window(
    portfolio_weights: Union[Sequence[float], np.ndarray, pd.Series],
    panel_returns: pd.DataFrame,
    start: Any,
    end: Any,
) -> pd.Series:
    """Replays historical realized returns over an exact crisis window.

    Raises DataUnavailableError if panel_returns does not contain valid observations.
    """
    if panel_returns.empty:
        raise DataUnavailableError(f"Historical returns for window {start} to {end} are not available.")

    s_ts = pd.to_datetime(start)
    e_ts = pd.to_datetime(end)

    window_df = panel_returns.loc[(panel_returns.index >= s_ts) & (panel_returns.index <= e_ts)]
    if window_df.empty or len(window_df) < 2:
        raise DataUnavailableError(
            f"No market data found in cache for historical window {start} to {end}."
        )

    w = np.asarray(portfolio_weights, dtype=np.float64)
    # Asset returns weighted sum
    port_daily_ret = window_df.values @ w
    return pd.Series(port_daily_ret, index=window_df.index, name="portfolio_return")


def stress_table(
    weights: Union[Sequence[float], np.ndarray, pd.Series],
    betas: Union[Sequence[float], np.ndarray, pd.Series],
    equity: float = 100_000.0,
    scenarios: Optional[dict[str, Scenario]] = None,
    adv: float = 1_000_000.0,
    baseline_vol: float = 0.15,
) -> pd.DataFrame:
    """Builds a formatted summary table across all stress scenarios."""
    scen_dict = scenarios or BUILTIN_SCENARIOS
    rows = []

    for name, scen in scen_dict.items():
        res = apply_scenario(
            weights=weights,
            betas=betas,
            scenario=scen,
            equity=equity,
            adv=adv,
            baseline_vol=baseline_vol,
        )
        rows.append(
            {
                "Scenario": res.scenario_name,
                "PnL (%)": f"{res.pnl_pct * 100.0:.2f}%",
                "PnL ($)": f"${res.pnl_dollar:,.2f}",
                "Stressed Vol": f"{res.stressed_vol * 100.0:.2f}%",
                "1d 99% VaR": f"{res.stressed_var_99_1d * 100.0:.2f}%",
                "1d 99% CVaR": f"{res.stressed_cvar_99_1d * 100.0:.2f}%",
                "Liquidation Cost": f"${res.liquidation_cost:,.2f}",
                "Days to Liquidate": f"{res.days_to_liquidate:.1f}",
                "Source Note": res.source_note,
            }
        )

    return pd.DataFrame(rows)
