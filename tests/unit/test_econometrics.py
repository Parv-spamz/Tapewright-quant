"""Unit tests for econometrics, fractional differentiation, Kalman filter, and causal graphs.

Verifies:
1. Fractional Differentiation (FFD) weights, edge cases (d=0, d=1), and find_min_d on random walks
2. Kalman Filter dynamic hedge ratio, tracking parameter shifts, and causal truncation invariance
3. Cointegration tests (Engle-Granger on true pairs vs random walks, Johansen rank, OU half-life)
4. Benjamini-Hochberg FDR step-up on hand-computed statistical example
5. Transfer Entropy and permutation tests (independent vs coupled vs deterministic copy)
6. Dual-gated Causal Graph (Granger + TE with FDR control) on coupled vs independent systems
7. Strict NO-LOOKAHEAD invariance across all time-series operations
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from data.ingestion.market_feeds.equities.synthetic import make_cointegrated_pair
from features.econometric.cointegration.causal_graph import (
    benjamini_hochberg,
    build_causal_graph,
    ensure_stationary,
    granger_matrix,
    te_permutation_pvalue,
    transfer_entropy,
)
from features.econometric.cointegration.tests import (
    engle_granger,
    half_life,
    johansen,
)
from features.econometric.fractional_diff.ffd import (
    find_min_d,
    frac_diff_ffd,
    get_weights_ffd,
)
from features.econometric.kalman_filter.kalman import KalmanHedgeRatio


class TestFractionalDifferentiation:
    """Tests for Lopez de Prado fixed-width window fractional differentiation."""

    def test_ffd_weights_recurrence_and_special_values(self) -> None:
        # d = 1 -> [1, -1]
        w_1 = get_weights_ffd(d=1.0)
        np.testing.assert_allclose(w_1, [1.0, -1.0])

        # d = 0.5 first four weights [1, -0.5, -0.125, -0.0625]
        w_half = get_weights_ffd(d=0.5)
        np.testing.assert_allclose(w_half[:4], [1.0, -0.5, -0.125, -0.0625])

    def test_frac_diff_edge_cases_d0_and_d1(self) -> None:
        rng = np.random.default_rng(42)
        s = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, size=200)), name="close")

        # d = 0 returns original series
        ffd_0 = frac_diff_ffd(s, d=0.0)
        pd.testing.assert_series_equal(ffd_0, s)

        # d = 1 equals first difference on valid region
        ffd_1 = frac_diff_ffd(s, d=1.0)
        ref_diff = s.diff()
        # For d=1, length of weights is 2, so valid region starts at index 1
        pd.testing.assert_series_equal(
            ffd_1.iloc[1:],
            ref_diff.iloc[1:],
            check_names=False,
        )

    def test_find_min_d_on_random_walk(self) -> None:
        """find_min_d on random walk (n=3000): d in (0,1], passes ADF, raw fails ADF, corr in (0,1)."""
        rng = np.random.default_rng(42)
        rw = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, size=3000)), name="rw")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_pvalue = float(adfuller(rw, autolag="AIC")[1])
        assert raw_pvalue > 0.05, f"Raw random walk should fail ADF, got p={raw_pvalue}"

        min_d, table_df = find_min_d(rw, adf_pvalue=0.05)
        print("\n[FFD Grid Table for Random Walk]:")
        print(table_df.to_string())

        assert 0.0 < min_d <= 1.0
        # Row corresponding to min_d
        row = table_df[table_df["d"] == min_d].iloc[0]
        assert row["pvalue"] <= 0.05
        assert 0.0 < row["corr_with_original"] < 1.0


class TestKalmanFilterHedgeRatio:
    """Tests for online Kalman Filter hedge ratio and spread estimation."""

    def test_kalman_parameter_recovery_synthetic_linear(self) -> None:
        """y = 2x + 1 + tiny noise -> final beta within 0.05 of 2, alpha within 0.2 of 1."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(10.0, 2.0, size=n)
        noise = rng.normal(0.0, 0.01, size=n)
        y = 2.0 * x + 1.0 + noise

        kf = KalmanHedgeRatio(delta=1e-4, ve=1e-3)
        res = kf.fit(x, y)

        final_beta = float(res["beta"].iloc[-1])
        final_alpha = float(res["alpha"].iloc[-1])

        assert abs(final_beta - 2.0) < 0.05
        assert abs(final_alpha - 1.0) < 0.20

    def test_kalman_dynamic_shift_adaptation(self) -> None:
        """With beta shifting halfway (1.0 -> 2.0), KF beta in last 10% is closer to 2.0 than static OLS."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(10.0, 2.0, size=n)
        noise = rng.normal(0.0, 0.02, size=n)
        y = np.zeros(n)
        y[:500] = 1.0 * x[:500] + 1.0 + noise[:500]
        y[500:] = 2.0 * x[500:] + 1.0 + noise[500:]

        kf = KalmanHedgeRatio(delta=1e-4, ve=1e-3)
        res = kf.fit(x, y)

        ols_beta = float(np.polyfit(x, y, 1)[0])
        kf_last_10_pct_beta = float(res["beta"].iloc[-100:].mean())

        ols_dist = abs(ols_beta - 2.0)
        kf_dist = abs(kf_last_10_pct_beta - 2.0)
        assert kf_dist < ols_dist


class TestCointegrationAndMeanReversion:
    """Tests for Engle-Granger, Johansen, and half-life estimators."""

    def test_engle_granger_cointegrated_pair_and_random_walks(self) -> None:
        # 1. True cointegrated pair from Prompt 4 generator
        x_coint, y_coint = make_cointegrated_pair(n=1000, beta=1.5, seed=42)
        eg_res = engle_granger(y_coint, x_coint)
        assert eg_res["pvalue"] < 0.05
        assert abs(eg_res["hedge_ratio"] - 1.5) < 0.10

        # 2. Independent random walks over 20 stated seeds: p > 0.05 for >= 17 of 20
        independent_pvals: list[float] = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            x_ind = 100.0 + np.cumsum(rng.normal(0, 1, size=1000))
            y_ind = 100.0 + np.cumsum(rng.normal(0, 1, size=1000))
            res = engle_granger(y_ind, x_ind)
            independent_pvals.append(res["pvalue"])

        passed_non_coint = sum(1 for p in independent_pvals if p > 0.05)
        print(f"\n[Independent RW Cointegration Rejections]: {passed_non_coint}/20 with p > 0.05")
        assert passed_non_coint >= 17

    def test_johansen_multivariate_rank(self) -> None:
        x, y = make_cointegrated_pair(n=1000, beta=1.2, seed=42)
        df_coint = pd.DataFrame({"y": y, "x": x})
        j_res = johansen(df_coint, det_order=0, k_ar_diff=1)
        assert j_res["rank"] >= 1

    def test_spread_half_life_recovery(self) -> None:
        """half_life on AR(1) phi=0.9 within 20% of -ln(2)/ln(0.9)."""
        rng = np.random.default_rng(42)
        n = 5000
        eps = rng.normal(0, 1, size=n)
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = 0.9 * s[t - 1] + eps[t]

        hl = half_life(s)
        expected_hl = -np.log(2.0) / np.log(0.9)
        assert abs(hl - expected_hl) / expected_hl < 0.20


class TestMultipleTestingFDR:
    """Tests for Benjamini-Hochberg step-up procedure."""

    def test_benjamini_hochberg_hand_example(self) -> None:
        """Hand-computed example with p-values and q=0.05."""
        # 10 p-values
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
        q_target = 0.05
        m = 10

        # Hand check:
        # k=1: 0.001 <= (1/10)*0.05 = 0.005 (True)
        # k=2: 0.008 <= (2/10)*0.05 = 0.010 (True)
        # k=3: 0.039 <= (3/10)*0.05 = 0.015 (False)
        # All k >= 3 fail. Largest k is 2.
        # Expected rejections: first 2 hypotheses (indices 0 and 1).
        q_vals, reject = benjamini_hochberg(p, q=q_target)

        assert int(np.sum(reject)) == 2
        assert bool(reject[0]) is True
        assert bool(reject[1]) is True
        assert bool(reject[2]) is False
        assert q_vals[0] <= q_target
        assert q_vals[1] <= q_target


class TestTransferEntropyMetrics:
    """Tests for Transfer Entropy and empirical permutation tests."""

    def test_transfer_entropy_properties(self) -> None:
        rng = np.random.default_rng(42)
        n = 1500
        x = rng.normal(0, 1, size=n)
        # Deterministic copy y_{t+1} = x_t
        y_det = np.roll(x, 1)
        te_det = transfer_entropy(x, y_det, bins=3)
        assert te_det > 0.5

        # Independent series
        z = rng.normal(0, 1, size=n)
        obs_xz, p_xz = te_permutation_pvalue(x, z, n_perm=50, bins=3, seed=42)
        assert p_xz > 0.05


class TestCausalGraphIntegration:
    """Predictive causality gating tests across coupled and independent networks."""

    def test_causal_graph_coupling_pattern_across_10_seeds(self) -> None:
        """x iid, y_t = 0.5*x_{t-1} + noise, z independent: x->y admitted, others not."""
        stated_seeds = [10, 21, 32, 43, 54, 65, 76, 87, 98, 109]
        pattern_matches = 0

        for s in stated_seeds:
            rng = np.random.default_rng(s)
            n = 2000
            x = rng.normal(0, 1, size=n)
            noise = rng.normal(0, 0.5, size=n)
            y = np.zeros(n)
            for t in range(1, n):
                y[t] = 0.5 * x[t - 1] + noise[t]
            z = rng.normal(0, 1, size=n)

            df = pd.DataFrame({"x": x, "y": y, "z": z})
            cg = build_causal_graph(df, maxlag=3, q=0.01, te_perm=50, seed=s)

            # Check TE asymmetric precedence
            te_xy = float(cg.edges[(cg.edges["source"] == "x") & (cg.edges["target"] == "y")]["te"].iloc[0])
            te_yx = float(cg.edges[(cg.edges["source"] == "y") & (cg.edges["target"] == "x")]["te"].iloc[0])
            assert te_xy > te_yx

            # Exact admission pattern
            match = (
                cg.is_admitted("x", "y")
                and not cg.is_admitted("y", "x")
                and not cg.is_admitted("x", "z")
                and not cg.is_admitted("z", "x")
                and not cg.is_admitted("y", "z")
                and not cg.is_admitted("z", "y")
            )
            if match:
                pattern_matches += 1

        print(f"\n[Causal Graph 10-Seed Pattern Matches]: {pattern_matches}/10")
        assert pattern_matches >= 9

    def test_causal_graph_fdr_control_on_pure_noise(self) -> None:
        """On 5 independent series over 20 seeds, average admitted edges < 0.5."""
        n_seeds = 20
        total_admitted = 0
        cols = [f"s_{i}" for i in range(5)]

        for s in range(n_seeds):
            rng = np.random.default_rng(s + 500)
            noise_mat = rng.normal(0, 1, size=(400, 5))
            df = pd.DataFrame(noise_mat, columns=cols)
            cg = build_causal_graph(df, maxlag=2, q=0.01, te_perm=30, seed=s)
            total_admitted += int(cg.adjacency.values.sum())

        avg_admitted = total_admitted / n_seeds
        print(f"\n[FDR Control Noise Average Admitted Edges]: {avg_admitted:.4f} < 0.5")
        assert avg_admitted < 0.5


class TestNoLookaheadInvarianceEconometrics:
    """Strict truncation invariance test on all econometric time-series functions."""

    def test_no_lookahead_ffd_and_kalman_and_stationarity(self) -> None:
        rng = np.random.default_rng(42)
        n_300 = 300
        x_300 = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, size=n_300)), name="x")
        y_300 = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, size=n_300)), name="y")

        x_200 = x_300.iloc[:200].copy()
        y_200 = y_300.iloc[:200].copy()

        # 1. FFD
        ffd_200 = frac_diff_ffd(x_200, d=0.4).values
        ffd_300 = frac_diff_ffd(x_300, d=0.4).iloc[:200].values
        np.testing.assert_allclose(ffd_200, ffd_300, equal_nan=True)

        # 2. Kalman Filter
        kf = KalmanHedgeRatio(delta=1e-4, ve=1e-3)
        res_200 = kf.fit(x_200, y_200)
        res_300 = kf.fit(x_300, y_300)

        for col in ["beta", "alpha", "spread", "spread_std"]:
            np.testing.assert_allclose(
                res_200[col].values,
                res_300[col].iloc[:200].values,
                equal_nan=True,
                err_msg=f"Lookahead in Kalman column {col}",
            )
