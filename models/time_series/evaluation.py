"""
Out-of-sample time-series evaluation, rolling origin cross-validation, and comparative testing.

Provides:
- rolling_origin_evaluation: Strictly causal, walk-forward evaluation advancing forecast origins.
- rmse: Root Mean Squared Error.
- mae: Mean Absolute Error.
- directional_accuracy: Empirical proportion of correct sign / trend predictions.
- diebold_mariano: HAC-corrected Diebold-Mariano test of equal forecast accuracy.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class DMResult(NamedTuple):
    """Container for Diebold-Mariano test results with attribute access and tuple unpacking."""

    statistic: float
    p_value: float

    @property
    def dm_stat(self) -> float:
        return self.statistic

    @property
    def pvalue(self) -> float:
        return self.p_value


def rmse(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Computes the Root Mean Squared Error (RMSE).

    RMSE = sqrt( (1 / n) * sum_{i=1}^n (y_i - y_hat_i)^2 )
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))


def mae(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Computes the Mean Absolute Error (MAE).

    MAE = (1 / n) * sum_{i=1}^n |y_i - y_hat_i|
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def directional_accuracy(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """Computes directional accuracy (fraction of sign concordance).

    DA = (1 / n) * sum_{i=1}^n I( sign(y_i) == sign(y_hat_i) )
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return 0.0
    sign_t = np.sign(yt[mask])
    sign_p = np.sign(yp[mask])
    return float(np.mean(sign_t == sign_p))


def diebold_mariano(
    e1: Union[pd.Series, np.ndarray],
    e2: Union[pd.Series, np.ndarray],
    h: int = 1,
    power: int = 2,
) -> DMResult:
    r"""Performs the Diebold-Mariano (1995) test for equal predictive accuracy.

    Mathematical Specification:
    --------------------------
    Given two forecast error series :math:`e_{1,t}` and :math:`e_{2,t}` of length :math:`T`:
    The loss differential series is defined as:
        :math:`d_t = L(e_{1,t}) - L(e_{2,t}) = |e_{1,t}|^{power} - |e_{2,t}|^{power}`
    Under the null hypothesis of equal forecast accuracy :math:`H_0: \mathbb{E}[d_t] = 0`.

    The sample mean loss differential is:
        :math:`\bar{d} = \frac{1}{T} \sum_{t=1}^T d_t`

    For :math:`h`-step ahead forecasts, errors may be :math:`MA(h-1)` autocorrelated.
    The asymptotic variance of :math:`\bar{d}` is computed using a HAC autocovariance estimator:
        :math:`\hat{\gamma}_0 = \frac{1}{T} \sum_{t=1}^T (d_t - \bar{d})^2`
        :math:`\hat{\gamma}_k = \frac{1}{T} \sum_{t=k+1}^T (d_t - \bar{d})(d_{t-k} - \bar{d})`
        :math:`\hat{V}(\bar{d}) = \frac{1}{T} \left[ \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \hat{\gamma}_k \right]`

    The Diebold-Mariano test statistic is:
        :math:`DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}}`

    Under :math:`H_0`, :math:`DM \sim \mathcal{N}(0, 1)` asymptotically.
    Two-sided p-value: :math:`2 \cdot (1 - \Phi(|DM|))`.

    Edge-Case Handling:
    -------------------
    If the two forecasts produce identical error series (:math:`e_1 \equiv e_2`) or
    sample variance is zero or negative, :math:`d_t = 0` identically; this is handled
    gracefully by returning :math:`DM = 0.0, p = 1.0`.

    Parameters
    ----------
    e1 : array-like
        Forecast error series from Model 1 (e1 = actual - forecast1).
    e2 : array-like
        Forecast error series from Model 2 (e2 = actual - forecast2).
    h : int, default 1
        Forecast horizon (used for HAC lag truncation up to h - 1).
    power : int, default 2
        Loss exponent (2 for MSE loss, 1 for MAE loss).

    Returns
    -------
    DMResult
        NamedTuple with (statistic, p_value), unpackable as `stat, p = diebold_mariano(...)`.
    """
    arr1 = np.asarray(e1, dtype=float)
    arr2 = np.asarray(e2, dtype=float)

    mask = np.isfinite(arr1) & np.isfinite(arr2)
    if not np.any(mask):
        return DMResult(0.0, 1.0)

    arr1 = arr1[mask]
    arr2 = arr2[mask]
    T = len(arr1)

    if T < 2 or np.allclose(arr1, arr2, atol=1e-12):
        return DMResult(0.0, 1.0)

    # Compute loss differential
    if power == 2:
        d = (arr1 ** 2) - (arr2 ** 2)
    elif power == 1:
        d = np.abs(arr1) - np.abs(arr2)
    else:
        d = (np.abs(arr1) ** power) - (np.abs(arr2) ** power)

    d_bar = float(np.mean(d))
    d_centered = d - d_bar

    # Lag 0 autocovariance
    gamma0 = float(np.mean(d_centered ** 2))
    if gamma0 <= 1e-14:
        return DMResult(0.0, 1.0)

    # HAC autocovariances up to lag h - 1
    gamma_sum = 0.0
    for k in range(1, max(1, h)):
        if k < T:
            gamma_k = float(np.sum(d_centered[k:] * d_centered[:-k]) / T)
            gamma_sum += 2.0 * gamma_k

    var_d_bar = (gamma0 + gamma_sum) / T

    if var_d_bar <= 1e-14 or np.isnan(var_d_bar):
        return DMResult(0.0, 1.0)

    dm_stat = float(d_bar / np.sqrt(var_d_bar))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat))))

    return DMResult(statistic=dm_stat, p_value=p_value)


def rolling_origin_evaluation(
    series: Union[pd.Series, np.ndarray],
    model_fn: Callable[[pd.Series], Any],
    initial: int,
    horizon: int,
    step: int = 1,
) -> pd.DataFrame:
    """Executes strictly causal walk-forward cross-validation using expanding rolling origins.

    At each origin index t, only the historical series slice series[:origin] is passed to
    `model_fn`. Under no circumstances is future data (series[origin:]) accessed during model
    estimation or forecasting.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Historical observation sequence.
    model_fn : callable
        Function receiving historical training slice `s[:origin]` and returning a model
        object with a `.forecast(steps)` method, or callable, or forecast array.
    initial : int
        Minimum training history length before the first forecast origin is evaluated.
    horizon : int
        Number of forward steps to forecast from each origin.
    step : int, default 1
        Stride length between successive forecast origins.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['origin', 'horizon_step', 'forecast', 'actual'].
    """
    if isinstance(series, np.ndarray):
        s = pd.Series(series)
    else:
        s = series

    n = len(s)
    records: list[dict[str, Any]] = []

    for origin in range(initial, n - horizon + 1, step):
        # Strict out-of-sample boundary: training series terminates at origin
        train_data = s.iloc[:origin]

        model = model_fn(train_data)

        # Extract predictions
        if hasattr(model, "forecast"):
            raw_fc = model.forecast(steps=horizon)
        elif hasattr(model, "predict"):
            raw_fc = model.predict(steps=horizon)
        elif callable(model):
            raw_fc = model(horizon)
        elif isinstance(model, (np.ndarray, pd.Series, list)):
            raw_fc = model
        else:
            raise ValueError(f"Unrecognized model return type from model_fn: {type(model)}")

        if isinstance(raw_fc, pd.Series):
            fc_vals = raw_fc.values
        else:
            fc_vals = np.asarray(raw_fc, dtype=float).flatten()

        for h in range(1, horizon + 1):
            act_idx = origin + h - 1
            if act_idx < n:
                actual_val = float(s.iloc[act_idx])
                fc_val = float(fc_vals[h - 1]) if len(fc_vals) >= h else np.nan
                records.append(
                    {
                        "origin": origin,
                        "horizon_step": h,
                        "forecast": fc_val,
                        "actual": actual_val,
                    }
                )

    return pd.DataFrame(records, columns=["origin", "horizon_step", "forecast", "actual"])
