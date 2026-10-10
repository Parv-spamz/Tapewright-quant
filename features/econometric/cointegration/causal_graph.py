"""Predictive causality gating and transfer-entropy directed graph with FDR control.

NOTE: In time-series econometrics, Granger causality and Transfer Entropy measure
PREDICTIVE PRECEDENCE (information flow / incremental forecasting power) and do NOT
imply true philosophical or structural causation.

With N asset series, there are N * (N - 1) directed ordered hypotheses tested.
The Benjamini-Hochberg (BH) step-up procedure is applied to control the False Discovery
Rate (FDR) across all simultaneous tests, preventing false discovery inflation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def ensure_stationary(
    df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Ensure all series in DataFrame are stationary via ADF testing.

    If a column has ADF p-value > pvalue_threshold:
    - If all values are positive, differences via log-return: np.log(col / col.shift(1))
    - Otherwise, differences via first difference: col.diff()

    Args:
        df: Input DataFrame of price or feature series.
        pvalue_threshold: Maximum p-value to be considered stationary (default 0.05).

    Returns:
        tuple[pd.DataFrame, dict[str, str]]: (transformed_stationary_df, stationarity_log)
    """
    transformed = pd.DataFrame(index=df.index)
    stationarity_log: dict[str, str] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in df.columns:
            s = df[col].dropna()
            if len(s) < 20:
                transformed[col] = df[col]
                stationarity_log[col] = "level"
                continue

            pval = float(adfuller(s, autolag="AIC")[1])
            if pval <= pvalue_threshold:
                transformed[col] = df[col]
                stationarity_log[col] = "level"
            else:
                if (df[col] > 0.0).all():
                    transformed[col] = np.log(df[col] / df[col].shift(1))
                    stationarity_log[col] = "log_return"
                else:
                    transformed[col] = df[col].diff()
                    stationarity_log[col] = "diff"

    clean_df = transformed.dropna()
    return clean_df, stationarity_log


def granger_matrix(
    df: pd.DataFrame,
    maxlag: int = 5,
    test: str = "ssr_ftest",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute bivariate Granger predictive precedence matrix with AIC lag selection.

    For each ordered pair (source -> target), fits a bivariate VAR up to maxlag,
    picks the optimal lag by AIC, and evaluates Granger test at that specific lag
    (avoiding multi-lag minimum p-value mining).

    Args:
        df: Stationary input DataFrame.
        maxlag: Maximum lag order to consider in VAR (default 5).
        test: Test statistic key in statsmodels (default 'ssr_ftest').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            p_matrix: DataFrame[source, target] containing Granger p-values.
            lag_matrix: DataFrame[source, target] containing chosen AIC lag orders.
    """
    cols = list(df.columns)
    p_matrix = pd.DataFrame(1.0, index=cols, columns=cols, dtype=float)
    lag_matrix = pd.DataFrame(1, index=cols, columns=cols, dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for src in cols:
            for tgt in cols:
                if src == tgt:
                    continue

                pair_df = df[[tgt, src]].dropna()
                if len(pair_df) < maxlag * 3:
                    continue

                # Select optimal lag via bivariate VAR AIC
                try:
                    var_model = VAR(pair_df)
                    order_res = var_model.select_order(maxlags=maxlag)
                    chosen_lag = int(order_res.aic)
                    chosen_lag = max(1, min(maxlag, chosen_lag))
                except Exception:
                    chosen_lag = 1

                # Statsmodels grangercausalitytests: col 0 is target (y), col 1 is source (x)
                try:
                    gc_res = grangercausalitytests(pair_df, maxlag=[chosen_lag])
                    p_val = float(gc_res[chosen_lag][0][test][1])
                except Exception:
                    p_val = 1.0

                p_matrix.loc[src, tgt] = p_val
                lag_matrix.loc[src, tgt] = chosen_lag

    return p_matrix, lag_matrix


def benjamini_hochberg(
    pvalues: Sequence[float] | np.ndarray,
    q: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply standard Benjamini-Hochberg (BH) step-up procedure for FDR control.

    Args:
        pvalues: 1D array-like sequence of p-values from multiple hypotheses.
        q: Target False Discovery Rate threshold (default 0.01).

    Returns:
        tuple[np.ndarray, np.ndarray]: (adjusted_qvalues, reject_boolean_array)
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]

    # Step-up threshold: largest k such that p_(k) <= (k / m) * q
    k_indices = np.arange(1, m + 1)
    thresholds = (k_indices / m) * q
    below_thresh = sorted_p <= thresholds

    max_k = 0
    if np.any(below_thresh):
        max_k = int(np.max(np.where(below_thresh)[0])) + 1

    reject_sorted = np.zeros(m, dtype=bool)
    if max_k > 0:
        reject_sorted[:max_k] = True

    reject = np.zeros(m, dtype=bool)
    reject[sorted_indices] = reject_sorted

    # Monotonic adjusted p-values (q-values)
    q_vals_sorted = np.zeros(m, dtype=float)
    min_val = 1.0
    for i in range(m - 1, -1, -1):
        k = i + 1
        adj_p = (m / k) * sorted_p[i]
        min_val = min(min_val, adj_p)
        q_vals_sorted[i] = min_val

    q_vals_sorted = np.clip(q_vals_sorted, 0.0, 1.0)
    q_vals = np.zeros(m, dtype=float)
    q_vals[sorted_indices] = q_vals_sorted

    return q_vals, reject


def _te_from_discrete(
    x_binned: np.ndarray,
    y_binned: np.ndarray,
    k: int = 1,
    bins: int = 3,
) -> float:
    yn = y_binned[k:]
    yc = y_binned[:-k]
    xc = x_binned[:-k]
    n_samples = len(yn)
    if n_samples == 0:
        return 0.0

    flat_idx = yn * (bins * bins) + yc * bins + xc
    counts_3d = np.bincount(flat_idx, minlength=bins**3).reshape((bins, bins, bins))

    counts_yx = np.sum(counts_3d, axis=0)
    counts_yy = np.sum(counts_3d, axis=2)
    counts_y = np.sum(counts_yx, axis=1)

    mask = counts_3d > 0
    p3 = counts_3d[mask] / n_samples
    yn_idx, yc_idx, xc_idx = np.where(mask)

    pyx = counts_yx[yc_idx, xc_idx] / n_samples
    pyy = counts_yy[yn_idx, yc_idx] / n_samples
    py = counts_y[yc_idx] / n_samples

    ratio = (p3 * py) / (pyx * pyy)
    valid_r = ratio > 0.0
    te = np.sum(p3[valid_r] * np.log2(ratio[valid_r]))
    return max(0.0, float(te))


def transfer_entropy(
    x: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    k: int = 1,
    bins: int = 3,
) -> float:
    """Calculate Transfer Entropy TE_{X -> Y} with quantile binning.

    Formula:
        TE_{X -> Y} = sum p(y_{t+1}, y_t, x_t) * log2( p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t) )
                    = sum p(y_{t+1}, y_t, x_t) * log2( (p(y_{t+1}, y_t, x_t) * p(y_t)) /
                                                       (p(y_t, x_t) * p(y_{t+1}, y_t)) )

    Args:
        x: Source time series.
        y: Target time series.
        k: History length lag (default 1).
        bins: Number of equal-frequency quantile discretization bins (default 3).

    Returns:
        float: Non-negative Transfer Entropy in bits.
    """
    x_s = pd.Series(x).dropna()
    y_s = pd.Series(y).dropna()
    common_idx = x_s.index.intersection(y_s.index)
    if len(common_idx) < k + 2:
        return 0.0

    x_arr = x_s.loc[common_idx].values.astype(float)
    y_arr = y_s.loc[common_idx].values.astype(float)

    try:
        x_binned = np.asarray(pd.qcut(x_arr, q=bins, labels=False, duplicates="drop"))
        y_binned = np.asarray(pd.qcut(y_arr, q=bins, labels=False, duplicates="drop"))
    except Exception:
        return 0.0

    actual_bins = max(int(np.max(x_binned)), int(np.max(y_binned))) + 1
    return _te_from_discrete(x_binned, y_binned, k=k, bins=actual_bins)


def te_permutation_pvalue(
    x: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    n_perm: int = 200,
    bins: int = 3,
    seed: int | None = None,
) -> tuple[float, float]:
    """Calculate Transfer Entropy and empirical permutation p-value under the null.

    Shuffles source x (block shuffle with block 1) to destroy temporal precedence.
    p = (1 + count(null >= obs)) / (1 + n_perm).

    Args:
        x: Source series.
        y: Target series.
        n_perm: Number of null permutations (default 200).
        bins: Quantile bins.
        seed: Random seed for permutations.

    Returns:
        tuple[float, float]: (observed_te, permutation_pvalue).
    """
    x_s = pd.Series(x).dropna()
    y_s = pd.Series(y).dropna()
    common_idx = x_s.index.intersection(y_s.index)
    if len(common_idx) < 10 or n_perm <= 0:
        return 0.0, 1.0

    x_arr = x_s.loc[common_idx].values.astype(float)
    y_arr = y_s.loc[common_idx].values.astype(float)

    try:
        x_binned = np.asarray(pd.qcut(x_arr, q=bins, labels=False, duplicates="drop"))
        y_binned = np.asarray(pd.qcut(y_arr, q=bins, labels=False, duplicates="drop"))
    except Exception:
        return 0.0, 1.0

    actual_bins = max(int(np.max(x_binned)), int(np.max(y_binned))) + 1
    obs_te = _te_from_discrete(x_binned, y_binned, k=1, bins=actual_bins)

    rng = np.random.default_rng(seed)
    greater_count = 0

    for _ in range(n_perm):
        shuffled_x = rng.permutation(x_binned)
        null_te = _te_from_discrete(shuffled_x, y_binned, k=1, bins=actual_bins)
        if null_te >= obs_te:
            greater_count += 1

    p_val = (1.0 + greater_count) / (1.0 + n_perm)
    return obs_te, p_val


@dataclass
class CausalGraph:
    """Directed Predictive-Precedence Causal Graph structure."""

    edges: pd.DataFrame
    adjacency: pd.DataFrame
    n_tests: int
    stationarity_log: dict[str, str]

    def parents(self, target: str) -> list[str]:
        """Return all admitted predictive predecessor sources for target."""
        if target not in self.adjacency.columns:
            return []
        admitted_col = self.adjacency[target]
        return list(admitted_col[admitted_col].index)

    def is_admitted(self, source: str, target: str) -> bool:
        """Check whether edge (source -> target) is statistically admitted."""
        if source in self.adjacency.index and target in self.adjacency.columns:
            return bool(self.adjacency.loc[source, target])
        return False


def build_causal_graph(
    df: pd.DataFrame,
    maxlag: int = 5,
    q: float = 0.01,
    te_perm: int = 200,
    seed: int | None = None,
) -> CausalGraph:
    """Construct directed predictive-precedence graph with dual Granger-TE gating.

    1. Enforces stationarity across all series via ADF tests and log-returns.
    2. Tests all N*(N-1) ordered pairs for Granger causality with AIC lag selection.
    3. Applies Benjamini-Hochberg step-up procedure at FDR threshold q across all tests.
    4. Evaluates Transfer Entropy and permutation p-values.
    5. An edge (source -> target) is admitted iff:
       - Granger p-value is BH-significant (granger_q <= q)
       - Transfer Entropy permutation p-value < 0.05.

    Args:
        df: Input DataFrame of multivariate asset price/feature series.
        maxlag: Maximum VAR lag for Granger test (default 5).
        q: False Discovery Rate target threshold (default 0.01).
        te_perm: Permutations for Transfer Entropy null distribution (default 200).
        seed: Random seed for TE permutations.

    Returns:
        CausalGraph: Graph container with edge details, adjacency matrix, and query methods.
    """
    stat_df, stat_log = ensure_stationary(df)
    cols = list(stat_df.columns)
    n = len(cols)

    p_mat, lag_mat = granger_matrix(stat_df, maxlag=maxlag)

    # Collect all N*(N-1) ordered pairs
    pairs: list[tuple[str, str]] = []
    pvals: list[float] = []
    lags: list[int] = []

    for src in cols:
        for tgt in cols:
            if src != tgt:
                pairs.append((src, tgt))
                pvals.append(float(p_mat.loc[src, tgt]))
                lags.append(int(lag_mat.loc[src, tgt]))

    n_tests = len(pairs)
    q_vals, bh_reject = benjamini_hochberg(pvals, q=q)

    # Transfer Entropy and dual admission
    edge_records: list[dict[str, Any]] = []
    adj_mat = pd.DataFrame(False, index=cols, columns=cols, dtype=bool)

    rng = np.random.default_rng(seed)

    for i, (src, tgt) in enumerate(pairs):
        perm_seed = int(rng.integers(0, 1_000_000)) if seed is not None else None
        te_val, te_p = te_permutation_pvalue(
            stat_df[src],
            stat_df[tgt],
            n_perm=te_perm,
            seed=perm_seed,
        )

        granger_sig = bool(bh_reject[i])
        te_sig = bool(te_p < 0.05)
        admitted = granger_sig and te_sig

        adj_mat.loc[src, tgt] = admitted

        edge_records.append(
            {
                "source": src,
                "target": tgt,
                "lag": lags[i],
                "granger_p": pvals[i],
                "granger_q": q_vals[i],
                "te": te_val,
                "te_p": te_p,
                "admitted": admitted,
            }
        )

    edges_df = pd.DataFrame(edge_records)
    return CausalGraph(
        edges=edges_df,
        adjacency=adj_mat,
        n_tests=n_tests,
        stationarity_log=stat_log,
    )
