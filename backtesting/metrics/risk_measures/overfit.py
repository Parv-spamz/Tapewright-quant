from __future__ import annotations

import itertools
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

from backtesting.metrics.risk_measures.measures import sharpe


def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Calculate the Probabilistic Sharpe Ratio (PSR).

    PSR = Phi( (sr_hat - sr_benchmark) * sqrt(T - 1) / sqrt(1 - skew*sr_hat + (kurt - 1)/4 * sr_hat^2) )
    where kurt is raw (non-excess) kurtosis, and sr values are in per-period units.

    Args:
        sr_hat: Observed Sharpe ratio estimate.
        sr_benchmark: Benchmark Sharpe ratio.
        T: Sample size (number of return observations).
        skew: Sample skewness of returns.
        kurt: Sample raw (non-excess) kurtosis of returns (normal = 3.0).

    Returns:
        float: Probability that the true Sharpe ratio exceeds the benchmark.
    """
    if T <= 1:
        return 0.5

    # Denominator variance adjustment (Mertens, 2002; Lopez de Prado, 2018)
    denom_var = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    denom = np.sqrt(max(denom_var, 1e-12))
    
    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Compute expected maximum Sharpe ratio under multiple testing selection bias.

    Formula (Lopez de Prado):
    E[max_N SR] = sqrt(var_sr) * ((1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N * e)))
    where gamma is the Euler-Mascheroni constant (0.5772156649).

    For n_trials <= 1 or var_sr <= 0, returns 0.0 since there is no multi-testing bias.

    Args:
        n_trials: Number of independent/trial backtest iterations (N).
        var_sr: Variance of Sharpe ratios across all trials.

    Returns:
        float: Expected maximum Sharpe ratio under the null hypothesis of no true alpha.
    """
    if n_trials <= 1 or var_sr <= 0.0:
        return 0.0

    gamma = 0.5772156649
    n = float(n_trials)
    
    # Arguments to inverse normal CDF
    arg1 = 1.0 - (1.0 / n)
    arg2 = 1.0 - (1.0 / (n * np.e))

    # Guard against precision limits
    arg1 = min(max(arg1, 1e-15), 1.0 - 1e-15)
    arg2 = min(max(arg2, 1e-15), 1.0 - 1e-15)

    z1 = stats.norm.ppf(arg1)
    z2 = stats.norm.ppf(arg2)

    val = np.sqrt(var_sr) * ((1.0 - gamma) * z1 + gamma * z2)
    return float(val)


def deflated_sharpe_ratio(
    sr_hat: float,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_trials: int = 1,
    var_sr: float = 0.0,
) -> float:
    """Calculate the Deflated Sharpe Ratio (DSR) discounting selection bias from multiple testing.

    DSR = PSR(sr_hat, sr_benchmark = expected_max_sharpe(n_trials, var_sr), T, skew, kurt)

    Args:
        sr_hat: Observed Sharpe ratio estimate.
        T: Sample size (number of observations).
        skew: Sample skewness of returns.
        kurt: Sample raw kurtosis of returns (normal = 3.0).
        n_trials: Number of strategies/trials tested.
        var_sr: Variance of Sharpe ratios across all trials.

    Returns:
        float: Deflated Sharpe Ratio probability.
    """
    benchmark = expected_max_sharpe(n_trials=n_trials, var_sr=var_sr)
    return probabilistic_sharpe_ratio(
        sr_hat=sr_hat,
        sr_benchmark=benchmark,
        T=T,
        skew=skew,
        kurt=kurt,
    )


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame | np.ndarray,
    n_blocks: int = 16,
    max_combos: int = 2000,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Calculate Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation (CSCV).

    Splits T observations into S = n_blocks contiguous blocks. For every combination of S/2 blocks
    as in-sample (sampled up to max_combos if total combinations exceed max_combos), identifies the
    best strategy in-sample, computes its out-of-sample rank w = rank / (N + 1), and logit
    lambda = ln(w / (1 - w)). PBO is the fraction of combinations where lambda <= 0.

    Args:
        returns_matrix: Array or DataFrame of shape (T, N) holding returns for N strategies across T periods.
        n_blocks: Number of contiguous blocks (S, must be even). Default is 16.
        max_combos: Maximum number of combinations to evaluate. Default is 2000.
        seed: Random seed for sampling combinations when total > max_combos.

    Returns:
        tuple[float, np.ndarray]: (pbo, logits_array)
    """
    mat = np.asarray(returns_matrix, dtype=float)
    T, N = mat.shape
    if N < 2:
        raise ValueError(f"PBO requires at least 2 strategies, got {N}")
    if n_blocks < 2:
        raise ValueError(f"n_blocks must be >= 2, got {n_blocks}")

    S = n_blocks
    k = S // 2

    # Partition T rows into S contiguous blocks
    block_size = T // S
    block_slices: list[slice] = []
    for i in range(S):
        start = i * block_size
        end = (i + 1) * block_size if i < S - 1 else T
        block_slices.append(slice(start, end))

    total_combos = math.comb(S, k)
    if total_combos <= max_combos:
        combos = list(itertools.combinations(range(S), k))
    else:
        rng = np.random.default_rng(seed)
        all_combos = list(itertools.combinations(range(S), k))
        chosen_indices = rng.choice(len(all_combos), size=max_combos, replace=False)
        combos = [all_combos[idx] for idx in chosen_indices]

    logits: list[float] = []

    for is_blocks in combos:
        is_set = set(is_blocks)
        oos_blocks = [b for b in range(S) if b not in is_set]

        # Extract IS and OOS rows
        is_rows = np.concatenate([np.arange(block_slices[b].start, block_slices[b].stop) for b in is_blocks])
        oos_rows = np.concatenate([np.arange(block_slices[b].start, block_slices[b].stop) for b in oos_blocks])

        is_mat = mat[is_rows, :]
        oos_mat = mat[oos_rows, :]

        # Compute Sharpe ratio per strategy in IS
        is_mean = np.mean(is_mat, axis=0)
        is_std = np.std(is_mat, axis=0, ddof=1)
        is_sharpe = np.where(is_std > 1e-12, is_mean / is_std, 0.0)

        # In-sample optimal strategy
        best_is_idx = int(np.argmax(is_sharpe))

        # Compute Sharpe ratio per strategy in OOS
        oos_mean = np.mean(oos_mat, axis=0)
        oos_std = np.std(oos_mat, axis=0, ddof=1)
        oos_sharpe = np.where(oos_std > 1e-12, oos_mean / oos_std, 0.0)

        # Out-of-sample rank of best IS strategy (rank 1 = worst, N = best)
        oos_ranks = stats.rankdata(oos_sharpe)
        rank_best = float(oos_ranks[best_is_idx])

        # Relative rank w in (0, 1)
        w = rank_best / (N + 1.0)
        w = min(max(w, 1e-12), 1.0 - 1e-12)

        # Logit lambda
        logit_val = float(np.log(w / (1.0 - w)))
        logits.append(logit_val)

    logits_arr = np.array(logits, dtype=float)
    pbo = float(np.mean(logits_arr <= 0.0))
    return pbo, logits_arr


def bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray,
    n_boot: int = 2000,
    block: int = 10,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate moving block bootstrap confidence interval for the annualized Sharpe ratio.

    Args:
        returns: Periodic simple returns series.
        n_boot: Number of bootstrap replications. Default is 2000.
        block: Block length in periods. Default is 10.
        alpha: Significance level (e.g. 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        tuple[float, float]: (lower_ci, upper_ci)
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    T = len(arr)
    if T <= 1:
        return 0.0, 0.0

    b = max(1, min(block, T))
    num_blocks = int(np.ceil(T / b))
    max_start = T - b + 1
    if max_start <= 0:
        max_start = 1
        b = T

    rng = np.random.default_rng(seed)
    boot_sharpes: list[float] = []

    for _ in range(n_boot):
        start_indices = rng.integers(0, max_start, size=num_blocks)
        sampled_parts = [arr[s : s + b] for s in start_indices]
        sample = np.concatenate(sampled_parts)[:T]
        sr = sharpe(sample, annualized=True)
        boot_sharpes.append(sr)

    boot_arr = np.array(boot_sharpes, dtype=float)
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower_ci = float(np.percentile(boot_arr, lower_pct))
    upper_ci = float(np.percentile(boot_arr, upper_pct))
    return lower_ci, upper_ci


def monte_carlo_equity_cone(
    returns: pd.Series | np.ndarray,
    horizon: int,
    n_paths: int = 1000,
    block: int = 10,
    seed: int = 42,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Generate block-bootstrap Monte Carlo forward equity paths and extract quantile cones.

    Args:
        returns: Historical return series to resample.
        horizon: Forecast horizon (number of steps).
        n_paths: Number of simulated forward paths.
        block: Block length for block bootstrapping.
        seed: Random seed.
        quantiles: Tuple of quantiles to evaluate at each step (e.g., 0.05, 0.5, 0.95).

    Returns:
        pd.DataFrame: DataFrame indexed 0..horizon with columns corresponding to quantiles.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    T = len(arr)
    if T == 0:
        zeros = np.ones((horizon + 1, len(quantiles)))
        return pd.DataFrame(zeros, index=range(horizon + 1), columns=list(quantiles))

    b = max(1, min(block, T))
    num_blocks = int(np.ceil(horizon / b))
    max_start = max(1, T - b + 1)

    rng = np.random.default_rng(seed)
    all_paths = np.ones((n_paths, horizon + 1), dtype=float)

    for i in range(n_paths):
        start_indices = rng.integers(0, max_start, size=num_blocks)
        sampled = np.concatenate([arr[s : s + b] for s in start_indices])[:horizon]
        all_paths[i, 1:] = np.cumprod(1.0 + sampled)

    result_df = pd.DataFrame(index=range(horizon + 1))
    for q in sorted(quantiles):
        result_df[q] = np.quantile(all_paths, q, axis=0)

    return result_df
