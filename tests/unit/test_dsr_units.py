from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.stats import norm

from backtesting.metrics.risk_measures.overfit import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def reference_dsr(
    returns_matrix: np.ndarray,
) -> tuple[float, float, float, int, float, float, float, float]:
    """Reference implementation using numpy and scipy.stats.norm."""
    T, N = returns_matrix.shape
    g = 0.5772156649

    # Per-period Sharpe ratios across N trials
    sr_trials = np.mean(returns_matrix, axis=0) / np.std(returns_matrix, axis=0, ddof=1)
    var_sr = float(np.var(sr_trials, ddof=1))

    # Select best-Sharpe column
    best_idx = int(np.argmax(sr_trials))
    sr_hat = float(sr_trials[best_idx])
    best_returns = returns_matrix[:, best_idx]

    # Expected maximum Sharpe under selection bias
    sr0 = float(
        np.sqrt(var_sr)
        * ((1.0 - g) * norm.ppf(1.0 - 1.0 / N) + g * norm.ppf(1.0 - 1.0 / (N * np.e)))
    )

    # Skew and raw (non-excess) kurtosis
    skew = float(stats.skew(best_returns))
    kurt = float(stats.kurtosis(best_returns, fisher=False))

    # Mertens denominator and standard normal test statistic
    denom_var = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    denom = np.sqrt(max(denom_var, 1e-12))
    z = float((sr_hat - sr0) * np.sqrt(T - 1) / denom)
    psr = float(norm.cdf(z))

    return psr, sr_hat, sr0, T, skew, kurt, z, var_sr


def test_dsr_reference_reconciliation() -> None:
    """Test 1: Returns matrix T=1000, N=30, iid normal(0, 0.01), seed 1.
    The library's DSR must equal the reference to 1e-9.
    """
    np.random.seed(1)
    T = 1000
    N = 30
    R = np.random.normal(0.0, 0.01, (T, N))

    ref_psr, sr_hat, sr0, t_obs, sk, kt, z, var_sr = reference_dsr(R)

    lib_dsr = deflated_sharpe_ratio(
        sr_hat=sr_hat,
        T=t_obs,
        skew=sk,
        kurt=kt,
        n_trials=N,
        var_sr=var_sr,
    )

    print(
        f"\n[Test 1 intermediates]: sr_hat={sr_hat:.6f}, sr0={sr0:.6f}, T={t_obs}, "
        f"skew={sk:.6f}, kurt={kt:.6f}, z={z:.6f}, ref={ref_psr:.9f}, lib={lib_dsr:.9f}"
    )

    assert abs(lib_dsr - ref_psr) < 1e-9


def test_dsr_units_contract() -> None:
    """Test 2 (units): Contract test for per-period vs annualized inputs.

    Real contract of deflated_sharpe_ratio / probabilistic_sharpe_ratio:
    By mathematical definition (Mertens, 2002; Bailey & Lopez de Prado, 2014),
    the denominator formula sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) strictly requires
    per-period quantities. If annualized inputs are supplied (annualized=True),
    the function internally converts sr_hat by dividing by sqrt(ppy) and var_sr
    by dividing by ppy, producing the exact same invariant DSR probability.
    """
    sr_period = 0.055711
    var_period = 0.000624
    T = 1000
    sk = 0.074392
    kt = 2.896776
    N = 30
    ppy = 252

    # Annualized counterparts
    sr_ann = sr_period * np.sqrt(ppy)
    var_ann = var_period * ppy

    # Per-period evaluation
    dsr_from_period = deflated_sharpe_ratio(
        sr_hat=sr_period,
        T=T,
        skew=sk,
        kurt=kt,
        n_trials=N,
        var_sr=var_period,
        annualized=False,
    )

    # Annualized evaluation with internal conversion
    dsr_from_ann = deflated_sharpe_ratio(
        sr_hat=sr_ann,
        T=T,
        skew=sk,
        kurt=kt,
        n_trials=N,
        var_sr=var_ann,
        annualized=True,
        ppy=ppy,
    )

    print(
        f"\n[Test 2 units]: dsr_period={dsr_from_period:.9f}, dsr_ann={dsr_from_ann:.9f}, "
        f"diff={abs(dsr_from_period - dsr_from_ann):.2e}"
    )

    assert abs(dsr_from_period - dsr_from_ann) < 1e-9


def test_dsr_noise_50_seeds_distribution() -> None:
    """Test 3: Over 50 seeds (1..50) of pure noise, N=30, T=1000:
    median DSR is within [0.25, 0.75], min > 0.001, max < 0.999.
    """
    T = 1000
    N = 30
    dsrs = []

    for s in range(1, 51):
        np.random.seed(s)
        R_s = np.random.normal(0.0, 0.01, (T, N))
        ref_psr, sr_hat, sr0, _, sk, kt, _, var_sr = reference_dsr(R_s)
        lib_val = deflated_sharpe_ratio(
            sr_hat=sr_hat,
            T=T,
            skew=sk,
            kurt=kt,
            n_trials=N,
            var_sr=var_sr,
        )
        dsrs.append(lib_val)

    dsrs_arr = np.array(dsrs)
    q0, q25, q50, q75, q100 = np.quantile(dsrs_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(
        f"\n[Test 3 50 seeds quantiles]: min={q0:.6f}, q25={q25:.6f}, "
        f"median={q50:.6f}, q75={q75:.6f}, max={q100:.6f}"
    )

    assert 0.25 <= q50 <= 0.75
    assert q0 > 0.001
    assert q100 < 0.999
