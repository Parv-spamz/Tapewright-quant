"""
GARCH and EGARCH volatility modeling, out-of-sample rolling forecasting, and regime gating.

Provides:
- fit_garch: Estimates GARCH(p, q, o) conditional heteroskedasticity.
- fit_egarch: Estimates Exponential GARCH (Nelson 1991) with asymmetric leverage.
- forecast_vol: Multi-step or 1-step ahead volatility forecast with optional annualization.
- rolling_vol_forecast: Strictly causal, out-of-sample 1-step volatility forecasts.
- regime_probability: Trailing empirical percentile rank of conditional volatility for regime gating.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult


def fit_garch(
    returns: Union[pd.Series, np.ndarray],
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: str = "t",
    mean: str = "Constant",
) -> ARCHModelResult:
    """Fits a GARCH(p, q, o) model on return series.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Asset log or fractional return series.
    p : int, default 1
        Symmetric GARCH lag order (lagged variance).
    q : int, default 1
        ARCH lag order (lagged squared residuals).
    o : int, default 0
        Asymmetric shock lag order (GJR-GARCH terms).
    dist : str, default 't'
        Error distribution ('normal', 't', 'skewt').
    mean : str, default 'Constant'
        Conditional mean specification ('Constant', 'Zero').

    Returns
    -------
    ARCHModelResult
        Fitted model result object from arch.
    """
    if isinstance(returns, np.ndarray):
        r = pd.Series(returns)
    else:
        r = returns.dropna()

    model = arch_model(
        r,
        vol="Garch",
        p=p,
        q=q,
        o=o,
        dist=dist,
        mean=mean,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off")
    return res


def fit_egarch(
    returns: Union[pd.Series, np.ndarray],
    p: int = 1,
    q: int = 1,
    o: int = 1,
    dist: str = "t",
    mean: str = "Constant",
) -> ARCHModelResult:
    """Fits an EGARCH(p, q, o) model to capture volatility clustering and leverage asymmetry.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Asset log or fractional return series.
    p : int, default 1
        Lagged log-variance order.
    q : int, default 1
        Standardized shock magnitude order.
    o : int, default 1
        Asymmetric leverage order.
    dist : str, default 't'
        Error distribution.
    mean : str, default 'Constant'
        Conditional mean model.

    Returns
    -------
    ARCHModelResult
        Fitted EGARCH model result.
    """
    if isinstance(returns, np.ndarray):
        r = pd.Series(returns)
    else:
        r = returns.dropna()

    model = arch_model(
        r,
        vol="EGARCH",
        p=p,
        q=q,
        o=o,
        dist=dist,
        mean=mean,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off")
    return res


def forecast_vol(
    res: ARCHModelResult,
    horizon: int = 1,
    annualize: bool = True,
) -> Union[float, pd.Series]:
    """Computes conditional volatility forecast over the specified horizon.

    Parameters
    ----------
    res : ARCHModelResult
        Fitted ARCH/GARCH model result.
    horizon : int, default 1
        Number of out-of-sample forward forecast steps.
    annualize : bool, default True
        If True, annualizes daily volatility by multiplying by sqrt(252).

    Returns
    -------
    float or pd.Series
        If horizon == 1, returns scalar volatility float.
        If horizon > 1, returns pd.Series of volatility forecasts indexed by step.
    """
    fc = res.forecast(horizon=horizon)
    variance_row = fc.variance.iloc[-1].values
    vol = np.sqrt(np.maximum(variance_row, 1e-12))
    if annualize:
        vol = vol * np.sqrt(252.0)

    if horizon == 1:
        return float(vol[0])
    return pd.Series(vol, index=range(1, horizon + 1), name="cond_vol")


def rolling_vol_forecast(
    returns: Union[pd.Series, np.ndarray],
    window: int = 250,
    refit_every: int = 21,
    dist: str = "normal",
) -> pd.Series:
    """Computes strictly out-of-sample 1-step volatility forecasts using rolling historical data.

    At each bar t, the model utilizes only data in returns[t - window + 1 : t + 1] (past data
    through t) to forecast the conditional volatility for bar t + 1. Parameter estimates are
    refitted periodically every `refit_every` observations and advanced via recursive GARCH(1,1)
    updates between refits to guarantee computational efficiency without lookahead.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical return series.
    window : int, default 250
        Lookback window length.
    refit_every : int, default 21
        Refit interval frequency in bars.
    dist : str, default 'normal'
        Distribution for conditional heteroskedasticity.

    Returns
    -------
    pd.Series
        Series aligned to returns index, where entry t represents the 1-step ahead
        conditional volatility forecast for bar t+1 made at bar t.
    """
    if isinstance(returns, np.ndarray):
        s = pd.Series(returns)
    else:
        s = returns

    n = len(s)
    out = pd.Series(np.nan, index=s.index, dtype=float)

    if n < window:
        return out

    # Cached parameter state
    omega = 0.01
    alpha = 0.05
    beta = 0.90
    mu = 0.0
    last_var = float(np.var(s.iloc[:window]))

    for t in range(window - 1, n):
        is_refit = ((t - (window - 1)) % refit_every == 0)

        if is_refit:
            sub_window = s.iloc[t - window + 1 : t + 1]
            try:
                res = fit_garch(sub_window, p=1, q=1, dist=dist, mean="Constant")
                mu = float(res.params.get("mu", 0.0))
                omega = float(res.params.get("omega", 1e-4))
                alpha = float(res.params.get("alpha[1]", 0.05))
                beta = float(res.params.get("beta[1]", 0.90))

                # 1-step forecast directly from model
                fc_var = float(res.forecast(horizon=1).variance.iloc[-1, 0])
                last_var = max(fc_var, 1e-12)
            except Exception:
                # If solver fails, apply recursive update
                eps = float(s.iloc[t]) - mu
                last_var = max(omega + alpha * (eps**2) + beta * last_var, 1e-12)
        else:
            # Recursive one-step ahead update using past residual observed at t
            eps = float(s.iloc[t]) - mu
            last_var = max(omega + alpha * (eps**2) + beta * last_var, 1e-12)

        out.iloc[t] = np.sqrt(last_var)

    return out


def regime_probability(
    cond_vol: Union[pd.Series, np.ndarray],
    lookback: int = 60,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Calculates the empirical percentile rank of current conditional volatility within past history.

    For each bar t, evaluates the fraction of trailing conditional volatility values
    within [t - lookback + 1, t] that are less than or equal to cond_vol[t]. Strictly
    causal and guaranteed to lie in [0, 1].

    Parameters
    ----------
    cond_vol : pd.Series or np.ndarray
        Conditional volatility series.
    lookback : int, default 60
        Trailing historical window length.
    min_periods : int, optional
        Minimum observations required. If None, defaults to lookback.

    Returns
    -------
    pd.Series
        Regime probability in [0, 1] aligned to cond_vol index.
    """
    if isinstance(cond_vol, np.ndarray):
        s = pd.Series(cond_vol)
    else:
        s = cond_vol

    req_periods = lookback if min_periods is None else min_periods
    n = len(s)
    probs = pd.Series(np.nan, index=s.index, dtype=float)

    for t in range(n):
        start_idx = max(0, t - lookback + 1)
        sub = s.iloc[start_idx : t + 1].dropna()
        if len(sub) < req_periods or np.isnan(s.iloc[t]):
            continue
        current_val = float(s.iloc[t])
        # Empirical percentile rank: fraction of past values <= current
        rank_val = float(np.mean(sub.values <= current_val))
        probs.iloc[t] = max(0.0, min(1.0, rank_val))

    return probs
