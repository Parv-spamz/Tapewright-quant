"""Fractionally Differentiated Features (FFD) based on Marcos Lopez de Prado.

Preserves memory while achieving stationarity through fractional differentiation.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Generate memory weights for Fractional Differentiation with Fixed-Width Window (FFD).

    Recurrence relation:
        w_0 = 1.0
        w_k = -w_{k-1} * (d - k + 1) / k
    Generation terminates when |w_k| < threshold.

    Args:
        d: Real-valued differentiation order (typically in [0, 1]).
        threshold: Weight cutoff threshold (default 1e-5).

    Returns:
        np.ndarray: Array of weights w_0, w_1, ..., w_{K-1}.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1.0) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w, dtype=float)


def frac_diff_ffd(
    series: pd.Series | np.ndarray,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply Fractional Differentiation with Fixed-Width Window (FFD).

    Generates a stationary series while retaining maximal memory of the original series.
    The first K - 1 observations (warm-up window) are set to NaN.

    Args:
        series: Time-series observations (prices, log-prices).
        d: Fractional differentiation degree (d=0 returns original, d=1 returns first diff).
        threshold: Weight drop-off threshold (default 1e-5).

    Returns:
        pd.Series: Fractionally differentiated series on the original index.
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    if d == 0.0:
        return s.copy()

    w = get_weights_ffd(d, threshold=threshold)
    k = len(w)
    n = len(s)

    res = np.full(n, np.nan, dtype=float)
    if n < k:
        return pd.Series(res, index=s.index, name=s.name)

    vals = s.values.astype(float)
    w_rev = w[::-1]

    # Causal dot-product over fixed window of length k
    for t in range(k - 1, n):
        res[t] = np.dot(w_rev, vals[t - k + 1 : t + 1])

    return pd.Series(res, index=s.index, name=s.name)


def find_min_d(
    series: pd.Series | np.ndarray,
    d_grid: np.ndarray | Sequence[float] | None = None,
    adf_pvalue: float = 0.05,
    threshold: float = 1e-5,
) -> tuple[float, pd.DataFrame]:
    """Find the minimum degree of fractional differentiation d that passes the ADF test.

    Evaluates grid of d candidates and reports ADF statistics, p-values, and
    measured memory retention (Pearson correlation with original series on overlap).

    Args:
        series: Input non-stationary price/log-price series.
        d_grid: Candidate differentiation orders (default np.linspace(0, 1, 21)).
        adf_pvalue: Target p-value threshold for ADF stationarity test (default 0.05).
        threshold: FFD weight threshold.

    Returns:
        tuple[float, pd.DataFrame]: (min_d, table_df).
            table_df columns: ['d', 'adf_stat', 'pvalue', 'corr_with_original', 'n_obs'].
    """
    if d_grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    else:
        grid = np.asarray(d_grid, dtype=float)

    s = pd.Series(series) if not isinstance(series, pd.Series) else series

    rows: list[dict[str, float | int]] = []
    min_d: float = 1.0
    found = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d_val in grid:
            ffd_series = frac_diff_ffd(s, d=float(d_val), threshold=threshold)
            valid = ffd_series.dropna()
            if len(valid) < 20:
                continue

            adf_res = adfuller(valid, autolag="AIC")
            adf_stat = float(adf_res[0])
            pval = float(adf_res[1])

            # Measure overlap memory retention (Pearson correlation)
            orig_overlap = s.loc[valid.index].astype(float)
            corr = float(orig_overlap.corr(valid))

            rows.append(
                {
                    "d": round(float(d_val), 4),
                    "adf_stat": adf_stat,
                    "pvalue": pval,
                    "corr_with_original": corr,
                    "n_obs": int(len(valid)),
                }
            )

            if not found and pval <= adf_pvalue:
                min_d = round(float(d_val), 4)
                found = True

    table_df = pd.DataFrame(rows)
    return min_d, table_df
