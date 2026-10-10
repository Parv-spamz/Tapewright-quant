"""Cointegration tests and mean-reverting spread metrics.

Includes:
- Engle-Granger two-step cointegration test
- Johansen multivariate cointegration test
- AR(1) Ornstein-Uhlenbeck spread half-life of mean reversion
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Execute the Engle-Granger two-step cointegration test.

    Runs OLS regression y = alpha + beta * x + e, then ADF unit-root test on residuals.

    Args:
        y: Dependent price series.
        x: Independent price series.

    Returns:
        dict with keys:
            - 'stat': Engle-Granger test statistic
            - 'pvalue': p-value of cointegration
            - 'crit': Critical values array (1%, 5%, 10%)
            - 'hedge_ratio': Estimated static beta slope
            - 'residual': pd.Series of cointegrating spread residuals
    """
    y_s = pd.Series(y) if not isinstance(y, pd.Series) else y
    x_s = pd.Series(x) if not isinstance(x, pd.Series) else x

    # Fit OLS regression for hedge ratio and residuals
    x_with_const = sm.add_constant(x_s)
    ols_res = sm.OLS(y_s, x_with_const).fit()

    hedge_ratio = float(ols_res.params.iloc[1])
    residuals = ols_res.resid

    stat, pval, crit = coint(y_s, x_s)

    return {
        "stat": float(stat),
        "pvalue": float(pval),
        "crit": np.asarray(crit, dtype=float),
        "hedge_ratio": hedge_ratio,
        "residual": pd.Series(residuals, index=y_s.index, name="residual"),
    }


def johansen(
    df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict[str, Any]:
    """Execute Johansen multivariate cointegration test.

    Args:
        df: DataFrame containing multivariate asset price series.
        det_order: Deterministic term order (-1: no const, 0: const in coint space, 1: linear trend).
        k_ar_diff: Lag order of difference terms (default 1).

    Returns:
        dict with keys:
            - 'trace_stat': Trace test statistics
            - 'trace_crit': Trace critical values (90%, 95%, 99%)
            - 'eigen_stat': Maximum eigenvalue statistics
            - 'eigen_crit': Eigenvalue critical values
            - 'rank': Cointegrating rank r at 95% significance level
            - 'eigenvectors': Cointegrating eigenvectors matrix
    """
    res = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stat = np.asarray(res.lr1, dtype=float)
    trace_crit = np.asarray(res.cvt, dtype=float)
    eigen_stat = np.asarray(res.lr2, dtype=float)
    eigen_crit = np.asarray(res.cvm, dtype=float)

    # Rank determined by comparing trace stat against 95% critical value (column 1)
    rank = int(np.sum(trace_stat > trace_crit[:, 1]))

    return {
        "trace_stat": trace_stat,
        "trace_crit": trace_crit,
        "eigen_stat": eigen_stat,
        "eigen_crit": eigen_crit,
        "rank": rank,
        "eigenvectors": np.asarray(res.evec, dtype=float),
    }


def half_life(spread: pd.Series | np.ndarray) -> float:
    """Calculate the half-life of mean reversion from an AR(1) model.

    Model: s_t = phi * s_{t-1} + const + e_t.
    Half-life: t_{1/2} = -ln(2) / ln(phi).

    Args:
        spread: Cointegrated spread series.

    Returns:
        float: Half-life in time steps. Returns np.inf if non-stationary (phi >= 1 or phi <= 0).
    """
    s = pd.Series(spread).dropna().values.astype(float)
    if len(s) < 3:
        return float(np.inf)

    s_lag = s[:-1]
    s_curr = s[1:]

    x_const = sm.add_constant(s_lag)
    ols = sm.OLS(s_curr, x_const).fit()
    phi = float(ols.params[1])

    if phi <= 0.0 or phi >= 1.0:
        return float(np.inf)

    return float(-np.log(2.0) / np.log(phi))
