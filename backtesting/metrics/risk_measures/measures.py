from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


def annualized_return(returns: pd.Series | np.ndarray, ppy: int = 252) -> float:
    """Calculate annualized geometric return: (prod(1 + r))^(ppy / n) - 1."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return 0.0
    compounded = np.prod(1.0 + arr)
    if compounded <= 0.0:
        return -1.0
    return float(compounded ** (ppy / n) - 1.0)


def annualized_vol(returns: pd.Series | np.ndarray, ppy: int = 252) -> float:
    """Calculate annualized sample standard deviation with ddof=1: std * sqrt(ppy)."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) * np.sqrt(ppy))


def sharpe(
    returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    ppy: int = 252,
    annualized: bool = True,
) -> float:
    """Calculate Sharpe ratio. Returns 0.0 for zero variance / constant return series."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    vol = np.std(arr, ddof=1)
    if vol <= 1e-14:
        # Constant-return series handled without exception
        return 0.0

    rf_per_period = rf / ppy if annualized else rf
    mean_excess = np.mean(arr) - rf_per_period
    factor = np.sqrt(ppy) if annualized else 1.0
    return float((mean_excess / vol) * factor)


def sortino(
    returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    mar: float = 0.0,
    ppy: int = 252,
    annualized: bool = True,
) -> float:
    """Calculate Sortino ratio with downside deviation sqrt(mean(min(r - mar, 0)^2))."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0

    mar_per_period = mar / ppy if annualized else mar
    downside_diff = np.minimum(arr - mar_per_period, 0.0)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2))
    if downside_dev <= 1e-14:
        return 0.0

    rf_per_period = rf / ppy if annualized else rf
    mean_excess = np.mean(arr) - rf_per_period
    factor = np.sqrt(ppy) if annualized else 1.0
    return float((mean_excess / downside_dev) * factor)


def calmar(
    returns: pd.Series | np.ndarray,
    max_dd: float | None = None,
    ppy: int = 252,
) -> float:
    """Calculate Calmar ratio: annualized_return / abs(max_drawdown)."""
    ann_ret = annualized_return(returns, ppy=ppy)
    if max_dd is None:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        equity = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    else:
        dd = float(max_dd)

    abs_dd = abs(dd)
    if abs_dd <= 1e-14:
        return 0.0
    return float(ann_ret / abs_dd)


def historical_var(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Calculate Historical Value at Risk reported as a POSITIVE loss number.

    VaR_alpha = -quantile(returns, 1 - alpha).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    q = np.quantile(arr, 1.0 - alpha)
    return float(max(0.0, -q))


def parametric_var(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Calculate Parametric (Gaussian) Value at Risk reported as a POSITIVE loss number.

    VaR_alpha = - (mean - z_alpha * std).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    z = stats.norm.ppf(alpha)
    loss = -(mu - z * sigma)
    return float(max(0.0, loss))


def cornish_fisher_var(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Calculate Cornish-Fisher expansion Value at Risk adjusting for skewness and raw kurtosis.

    Reported as a POSITIVE loss number.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 2:
        return parametric_var(arr, alpha=alpha)

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 1e-14:
        return 0.0

    s = float(skew(arr))
    k = float(kurtosis(arr))  # raw kurtosis

    z = stats.norm.ppf(alpha)
    # Cornish-Fisher z-score adjustment with raw kurtosis (k - 3 is excess kurtosis)
    z_cf = (
        z
        + (s / 6.0) * (z ** 2 - 1.0)
        + ((k - 3.0) / 24.0) * (z ** 3 - 3.0 * z)
        - (s ** 2 / 36.0) * (2.0 * z ** 3 - 5.0 * z)
    )

    loss = -(mu - z_cf * sigma)
    return float(max(0.0, loss))


def cvar(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall) as a POSITIVE loss number.

    CVaR_alpha = - mean(returns[returns <= quantile(returns, 1 - alpha)]).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    cutoff = np.quantile(arr, 1.0 - alpha)
    tail = arr[arr <= cutoff]
    if len(tail) == 0:
        return float(max(0.0, -cutoff))
    return float(max(0.0, -np.mean(tail)))


def omega_ratio(returns: pd.Series | np.ndarray, threshold: float = 0.0) -> float:
    """Calculate Omega ratio: sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    gains = np.maximum(arr - threshold, 0.0)
    losses = np.maximum(threshold - arr, 0.0)
    denom = np.sum(losses)
    if denom <= 1e-14:
        return float("inf") if np.sum(gains) > 0.0 else 1.0
    return float(np.sum(gains) / denom)


def skew(returns: pd.Series | np.ndarray) -> float:
    """Calculate sample skewness."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 2:
        return 0.0
    return float(stats.skew(arr, bias=False))


def kurtosis(returns: pd.Series | np.ndarray) -> float:
    """Calculate RAW (non-excess) kurtosis: normal distribution has kurtosis = 3.0."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 3:
        return 3.0
    # scipy.stats.kurtosis computes excess kurtosis by default (fisher=True); fisher=False gives raw kurtosis
    return float(stats.kurtosis(arr, fisher=False, bias=False))


def summary(
    returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    ppy: int = 252,
) -> dict[str, float]:
    """Compile comprehensive risk and performance summary metrics dictionary."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]

    ann_ret = annualized_return(arr, ppy=ppy)
    ann_v = annualized_vol(arr, ppy=ppy)
    sr = sharpe(arr, rf=rf, ppy=ppy)
    so = sortino(arr, rf=rf, mar=0.0, ppy=ppy)
    cal = calmar(arr, ppy=ppy)

    var_hist = historical_var(arr, alpha=0.95)
    var_param = parametric_var(arr, alpha=0.95)
    var_cf = cornish_fisher_var(arr, alpha=0.95)
    cvar_val = cvar(arr, alpha=0.95)

    om = omega_ratio(arr, threshold=0.0)
    sk = skew(arr)
    kt = kurtosis(arr)

    var_hist_99 = historical_var(arr, alpha=0.99)
    var_cf_99 = cornish_fisher_var(arr, alpha=0.99)
    cvar_99_val = cvar(arr, alpha=0.99)

    return {
        "annualized_return": ann_ret,
        "annualized_vol": ann_v,
        "sharpe": sr,
        "sharpe_ratio": sr,
        "sortino": so,
        "sortino_ratio": so,
        "calmar": cal,
        "calmar_ratio": cal,
        "historical_var_95": var_hist,
        "parametric_var_95": var_param,
        "cornish_fisher_var_95": var_cf,
        "cvar_95": cvar_val,
        "historical_var_99": var_hist_99,
        "cornish_fisher_var_99": var_cf_99,
        "cvar_99": cvar_99_val,
        "omega_ratio": om,
        "skew": sk,
        "skewness": sk,
        "kurtosis": kt,
        "kurtosis_raw": kt,
    }
