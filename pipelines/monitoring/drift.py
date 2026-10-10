"""
Model monitoring and distribution drift detection.

Features:
- Population Stability Index (PSI) with quantile binning.
- Kolmogorov-Smirnov (KS) two-sample drift testing.
- Automated feature drift reporting across train and live datasets.
- Performance decay detection (rolling Sharpe vs historical Sharpe).
- ModelMonitor orchestration class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from backtesting.metrics.risk_measures import sharpe


def population_stability_index(
    expected: Union[Sequence[float], np.ndarray, pd.Series],
    actual: Union[Sequence[float], np.ndarray, pd.Series],
    bins: int = 10,
    eps: float = 1e-4,
    bin_edges: Optional[np.ndarray] = None,
) -> float:
    """Computes Population Stability Index (PSI) between reference and observed distributions.

    Formula:
        PSI = sum_{b=1}^B (P_b - Q_b) * ln(P_b / Q_b)
    where Q_b is the reference proportion and P_b is the actual proportion.

    Parameters
    ----------
    expected : array-like
        Baseline/training distribution values.
    actual : array-like
        Live/monitoring distribution values.
    bins : int, default 10
        Number of quantile bins to form on expected data.
    eps : float, default 1e-4
        Floor replacement for empty bins to avoid division by zero or ln(0).
    bin_edges : np.ndarray, optional
        Pre-specified bin boundaries. If None, derived from expected quantiles.

    Returns
    -------
    float
        Computed PSI score.
    """
    exp_arr = np.asarray(expected, dtype=np.float64)
    act_arr = np.asarray(actual, dtype=np.float64)

    exp_clean = exp_arr[np.isfinite(exp_arr)]
    act_clean = act_arr[np.isfinite(act_arr)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    if bin_edges is None:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(exp_clean, quantiles)
        edges[0] = -np.inf
        edges[-1] = np.inf
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([-np.inf, np.mean(exp_clean), np.inf])
    else:
        edges = np.asarray(bin_edges, dtype=np.float64)

    exp_counts, _ = np.histogram(exp_clean, bins=edges)
    act_counts, _ = np.histogram(act_clean, bins=edges)

    total_exp = float(len(exp_clean))
    total_act = float(len(act_clean))

    p_exp = np.maximum(exp_counts / total_exp, eps)
    p_act = np.maximum(act_counts / total_act, eps)

    # Normalize to valid probabilities
    p_exp = p_exp / np.sum(p_exp)
    p_act = p_act / np.sum(p_act)

    psi_val = np.sum((p_act - p_exp) * np.log(p_act / p_exp))
    return float(max(0.0, psi_val))


def ks_drift(
    expected: Union[Sequence[float], np.ndarray, pd.Series],
    actual: Union[Sequence[float], np.ndarray, pd.Series],
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Evaluates two-sample Kolmogorov-Smirnov test for distribution drift.

    Parameters
    ----------
    expected : array-like
        Baseline training sample.
    actual : array-like
        Live monitoring sample.
    alpha : float, default 0.05
        Significance threshold for rejection.

    Returns
    -------
    tuple of (statistic, p_value, is_drift)
    """
    exp_arr = np.asarray(expected, dtype=np.float64)
    act_arr = np.asarray(actual, dtype=np.float64)
    exp_clean = exp_arr[np.isfinite(exp_arr)]
    act_clean = act_arr[np.isfinite(act_arr)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0, 1.0, False

    stat_res = stats.ks_2samp(exp_clean, act_clean)
    statistic = float(stat_res.statistic)
    p_val = float(stat_res.pvalue)
    is_drift = bool(p_val < alpha)

    return statistic, p_val, is_drift


def feature_drift_report(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    bins: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Analyzes distribution shift across all common numeric features.

    Drift severity classification:
        PSI < 0.10: "stable"
        0.10 <= PSI <= 0.25: "moderate"
        PSI > 0.25: "major"

    Parameters
    ----------
    train_df : pd.DataFrame
        Reference baseline feature matrix.
    live_df : pd.DataFrame
        Observed production feature matrix.
    bins : int, default 10
        Quantile bins for PSI.
    alpha : float, default 0.05
        KS test significance level.

    Returns
    -------
    pd.DataFrame
        Drift diagnostics per feature with columns:
        [feature, psi, psi_flag, ks_stat, ks_pvalue, ks_drift]
    """
    common_cols = [c for c in train_df.columns if c in live_df.columns]
    records = []

    for col in common_cols:
        s_train = train_df[col].dropna()
        s_live = live_df[col].dropna()

        if not np.issubdtype(s_train.dtype, np.number) or not np.issubdtype(s_live.dtype, np.number):
            continue

        if len(s_train) == 0 or len(s_live) == 0:
            continue

        psi_val = population_stability_index(s_train, s_live, bins=bins)
        ks_stat, ks_pval, ks_flag = ks_drift(s_train, s_live, alpha=alpha)

        if psi_val > 0.25:
            psi_flag = "major"
        elif psi_val >= 0.10:
            psi_flag = "moderate"
        else:
            psi_flag = "stable"

        records.append(
            {
                "feature": str(col),
                "psi": psi_val,
                "psi_flag": psi_flag,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pval,
                "ks_drift": ks_flag,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["feature", "psi", "psi_flag", "ks_stat", "ks_pvalue", "ks_drift"]
        )

    return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)


def performance_decay(
    returns: Union[Sequence[float], np.ndarray, pd.Series],
    window: int = 60,
    decay_ratio: float = 0.5,
    ppy: int = 252,
) -> dict[str, Any]:
    """Detects strategy decay when recent rolling Sharpe drops below historical baseline.

    Parameters
    ----------
    returns : array-like
        Full historical return stream.
    window : int, default 60
        Lookback window in periods for evaluating recent Sharpe.
    decay_ratio : float, default 0.5
        Threshold fraction of full-history Sharpe below which decay is flagged.
    ppy : int, default 252
        Periods per year.

    Returns
    -------
    dict with full_sharpe, rolling_sharpe, is_decayed, message.
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if len(arr) < window:
        return {
            "full_sharpe": 0.0,
            "rolling_sharpe": 0.0,
            "is_decayed": False,
            "message": "Insufficient data length",
        }

    full_sr = sharpe(arr, ppy=ppy)
    recent_sr = sharpe(arr[-window:], ppy=ppy)

    # Flag decay if full history had edge (full_sr > 0) and recent Sharpe collapsed
    is_decayed = bool(full_sr > 0.0 and recent_sr < (full_sr * decay_ratio))

    return {
        "full_sharpe": float(full_sr),
        "rolling_sharpe": float(recent_sr),
        "is_decayed": is_decayed,
        "message": (
            f"Decay detected: rolling Sharpe ({recent_sr:.2f}) < "
            f"{decay_ratio:.1f}x full Sharpe ({full_sr:.2f})"
            if is_decayed
            else "Performance stable"
        ),
    }


class ModelMonitor:
    """Automated model monitoring pipeline checking data drift and strategy decay."""

    def __init__(self, reference_features: pd.DataFrame, psi_threshold: float = 0.25) -> None:
        self.reference_features = reference_features.copy()
        self.psi_threshold = float(psi_threshold)

    def check_drift(self, live_features: pd.DataFrame) -> pd.DataFrame:
        """Evaluates feature drift between reference baseline and incoming live features."""
        return feature_drift_report(self.reference_features, live_features)

    def check_performance(self, returns: pd.Series | np.ndarray, window: int = 60) -> dict[str, Any]:
        """Evaluates return stream for performance decay."""
        return performance_decay(returns, window=window)
