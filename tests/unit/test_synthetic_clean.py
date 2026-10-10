from datetime import date
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, pearsonr, spearmanr
import statsmodels.tsa.stattools as ts

from data.cleaning.corporate_actions.adjust import adjust_for_dividends, adjust_for_splits
from data.cleaning.outlier_filters.filters import detect_price_spikes, mad_filter, remove_bad_ticks
from data.ingestion.market_feeds.equities.l2_tape import simulate_l2_tape
from data.ingestion.market_feeds.equities.synthetic import (
    make_cointegrated_pair,
    make_ohlcv,
    make_universe_panel,
    simulate_gbm,
    simulate_jump_diffusion,
    simulate_regime_switching,
)
from data.validation.schema_rules.rules import validate_book, validate_ohlcv


class TestSyntheticGenerators:
    def test_seed_determinism_and_divergence(self) -> None:
        # simulate_gbm
        gbm_1 = simulate_gbm(100.0, 0.05, 0.20, n=100, seed=42)
        gbm_2 = simulate_gbm(100.0, 0.05, 0.20, n=100, seed=42)
        gbm_diff = simulate_gbm(100.0, 0.05, 0.20, n=100, seed=99)
        np.testing.assert_array_equal(gbm_1, gbm_2)
        assert not np.array_equal(gbm_1, gbm_diff)

        # simulate_jump_diffusion
        jd_1 = simulate_jump_diffusion(100.0, 0.05, 0.20, lam=2.0, jump_mu=-0.02, jump_sigma=0.05, n=100, seed=42)
        jd_2 = simulate_jump_diffusion(100.0, 0.05, 0.20, lam=2.0, jump_mu=-0.02, jump_sigma=0.05, n=100, seed=42)
        jd_diff = simulate_jump_diffusion(100.0, 0.05, 0.20, lam=2.0, jump_mu=-0.02, jump_sigma=0.05, n=100, seed=99)
        np.testing.assert_array_equal(jd_1, jd_2)
        assert not np.array_equal(jd_1, jd_diff)

        # simulate_regime_switching
        rs_p1, rs_s1 = simulate_regime_switching(100.0, n=100, seed=42)
        rs_p2, rs_s2 = simulate_regime_switching(100.0, n=100, seed=42)
        rs_pdiff, _ = simulate_regime_switching(100.0, n=100, seed=99)
        np.testing.assert_array_equal(rs_p1, rs_p2)
        np.testing.assert_array_equal(rs_s1, rs_s2)
        assert not np.array_equal(rs_p1, rs_pdiff)

        # make_ohlcv
        ohlcv_1 = make_ohlcv("AAPL", start="2024-01-02", n_days=20, seed=42)
        ohlcv_2 = make_ohlcv("AAPL", start="2024-01-02", n_days=20, seed=42)
        ohlcv_diff = make_ohlcv("AAPL", start="2024-01-02", n_days=20, seed=99)
        pd.testing.assert_frame_equal(ohlcv_1, ohlcv_2)
        assert not ohlcv_1.equals(ohlcv_diff)

    def test_gbm_return_std_over_200_paths(self) -> None:
        target_std = 0.20 / np.sqrt(252.0)
        stds: list[float] = []

        for s in range(200):
            p = simulate_gbm(s0=100.0, mu=0.05, sigma=0.20, n=252, seed=s)
            log_ret = np.diff(np.log(p))
            stds.append(float(np.std(log_ret, ddof=1)))

        mean_std = float(np.mean(stds))
        rel_diff = abs(mean_std - target_std) / target_std
        assert rel_diff < 0.05, f"Mean std {mean_std:.6f} differed by {rel_diff:.2%} (> 5%) from target {target_std:.6f}"

    def test_kurtosis_jump_vs_gbm(self) -> None:
        # Jump-diffusion has excess kurtosis > 1 on a long path; GBM's is near 0 (abs < 0.5) for n=100000
        p_jump = simulate_jump_diffusion(
            s0=100.0, mu=0.05, sigma=0.15, lam=5.0,
            jump_mu=-0.03, jump_sigma=0.08, n=100_000, seed=42,
        )
        r_jump = np.diff(np.log(p_jump))
        k_jump = float(kurtosis(r_jump))
        assert k_jump > 1.0, f"Expected jump excess kurtosis > 1.0, got {k_jump:.4f}"

        p_gbm = simulate_gbm(s0=100.0, mu=0.05, sigma=0.15, n=100_000, seed=42)
        r_gbm = np.diff(np.log(p_gbm))
        k_gbm = float(kurtosis(r_gbm))
        assert abs(k_gbm) < 0.5, f"Expected GBM excess kurtosis near 0 (|k| < 0.5), got {k_gbm:.4f}"

    def test_regime_switching_volatility_difference(self) -> None:
        # Realized vol in true stress state > 2x realized vol in calm state
        p, s = simulate_regime_switching(s0=100.0, n=20_000, seed=42)
        r = np.diff(np.log(p))
        calm_r = r[s[1:] == 0]
        stress_r = r[s[1:] == 1]

        vol_calm = float(np.std(calm_r))
        vol_stress = float(np.std(stress_r))

        assert vol_stress > 2.0 * vol_calm, (
            f"Expected stress vol > 2x calm vol, got calm={vol_calm:.6f}, stress={vol_stress:.6f}"
        )

    def test_ohlc_invariants_20_seeds_x_3_models(self) -> None:
        models = ["gbm", "jump", "regime"]
        for model in models:
            for seed in range(20):
                df = make_ohlcv("TEST", start="2024-01-02", n_days=15, seed=seed, model=model)
                assert (df["high"] >= df["low"]).all()
                assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
                assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
                assert (df["open"] > 0).all() and (df["close"] > 0).all()
                assert (df["volume"] >= 0).all()

    def test_cointegrated_pair_adf_significance(self) -> None:
        # ADF p < 0.01 on y - beta * x
        beta = 1.5
        x, y = make_cointegrated_pair(n=2000, beta=beta, seed=42)
        spread = y - beta * x
        adf_result = ts.adfuller(spread)
        p_value = float(adf_result[1])
        assert p_value < 0.01, f"Expected ADF p-value < 0.01, got {p_value:.6f}"


class TestL2TapeSimulation:
    def test_l2_tape_planted_signal_recovery(self) -> None:
        n_steps = 50_000
        seed = 42
        book_df, trades_df = simulate_l2_tape(n_steps=n_steps, seed=seed)

        # 1. Book validity
        report = validate_book(book_df)
        assert report.ok is True, f"validate_book failed: {report.errors}"

        # 2. Planted signal recovery:
        # Spearman correlation between imbalance_true.shift(1) and mid log-return is > 0.05
        mid_log_ret = np.diff(np.log(book_df["mid"]), prepend=np.log(book_df["mid"].iloc[0]))
        spearman_corr, p_val = spearmanr(book_df["imbalance_true"].iloc[:-1], mid_log_ret[1:])
        print(f"Recovered L2 planted signal Spearman correlation: {spearman_corr:.4f} (p={p_val})")
        assert spearman_corr > 0.05, f"Expected Spearman correlation > 0.05, got {spearman_corr:.4f}"

        # 3. Top-of-book size imbalance vs net signed trade volume per step
        tob_imb = (book_df["bid_sz_1"] - book_df["ask_sz_1"]) / (book_df["bid_sz_1"] + book_df["ask_sz_1"])

        trades_df["signed_vol"] = trades_df["side"] * trades_df["size"]
        signed_vol_per_step = trades_df.groupby("timestamp")["signed_vol"].sum()

        # Reindex to all book timestamps (0 for steps with no trades)
        step_signed_vol = signed_vol_per_step.reindex(book_df.index, fill_value=0.0)

        corr_vol, _ = pearsonr(tob_imb, step_signed_vol)
        assert corr_vol > 0.0, f"Expected positive correlation with net signed volume, got {corr_vol:.4f}"


class TestValidationRules:
    def test_validate_ohlcv_catches_all_defects(self) -> None:
        valid_df = make_ohlcv("AAPL", start="2024-01-02", n_days=30, seed=42)
        assert validate_ohlcv(valid_df).ok is True

        # Injected defect 1: NaN
        df_nan = valid_df.copy()
        df_nan.iloc[5, df_nan.columns.get_loc("close")] = np.nan
        assert validate_ohlcv(df_nan).ok is False

        # Injected defect 2: Negative price
        df_neg = valid_df.copy()
        df_neg.iloc[5, df_neg.columns.get_loc("open")] = -10.0
        assert validate_ohlcv(df_neg).ok is False

        # Injected defect 3: high < low
        df_inv = valid_df.copy()
        df_inv.iloc[5, df_inv.columns.get_loc("high")] = df_inv.iloc[5, df_inv.columns.get_loc("low")] - 1.0
        assert validate_ohlcv(df_inv).ok is False

        # Injected defect 4: Unsorted index
        df_unsorted = valid_df.copy()
        idx = list(df_unsorted.index)
        idx[5], idx[6] = idx[6], idx[5]
        df_unsorted.index = pd.DatetimeIndex(idx)
        assert validate_ohlcv(df_unsorted).ok is False

        # Injected defect 5: Gap beyond allowed_gap_days
        df_gap = valid_df.copy()
        idx_gap = list(df_gap.index)
        idx_gap[10] = idx_gap[9] + pd.Timedelta(days=15)
        df_gap.index = pd.DatetimeIndex(idx_gap)
        assert validate_ohlcv(df_gap, allowed_gap_days=4).ok is False


class TestOutlierFiltersAndCleaning:
    def test_mad_filter_spikes_and_no_lookahead(self) -> None:
        np.random.seed(42)
        s = pd.Series(100.0 + np.random.normal(0.0, 0.5, size=1000))
        spike_indices = [100, 250, 400, 650, 800]
        for idx in spike_indices:
            s.iloc[idx] *= 1.40  # +40% spike

        flags = mad_filter(s, window=20, threshold=5.0)

        # Must catch all 5 injected spikes
        for idx in spike_indices:
            assert flags.iloc[idx] is True or flags.iloc[idx] == 1, f"Failed to catch spike at index {idx}"

        # Must flag < 1% of other points
        other_flags = flags.drop(index=spike_indices)
        false_positive_rate = float(other_flags.mean())
        assert false_positive_rate < 0.01, f"False positive rate {false_positive_rate:.2%} exceeded 1%"

        # No lookahead test: altering future values does not change earlier flags
        s_future_modified = s.copy()
        s_future_modified.iloc[900:] += 50.0  # Big change at index >= 900
        flags_modified = mad_filter(s_future_modified, window=20, threshold=5.0)

        pd.testing.assert_series_equal(flags.iloc[:900], flags_modified.iloc[:900])

    def test_remove_bad_ticks_and_detect_spikes(self) -> None:
        df = make_ohlcv("AAPL", start="2024-01-02", n_days=50, seed=42)
        df.iloc[25, df.columns.get_loc("close")] *= 2.0  # Inject bad tick

        clean_df, mask = remove_bad_ticks(df, window=10, threshold=4.0)
        assert mask.iloc[25] is True or mask.iloc[25] == 1
        assert clean_df.iloc[25]["close"] == df.iloc[24]["close"]


class TestCorporateActionsAdjust:
    def test_split_adjustment_math(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 50.0, 50.0],
                "high": [105.0, 105.0, 52.0, 52.0],
                "low": [95.0, 95.0, 48.0, 48.0],
                "close": [100.0, 100.0, 50.0, 50.0],
                "volume": [1000.0, 1000.0, 2000.0, 2000.0],
            },
            index=dates,
        )

        # 2:1 split on Day 3 (2024-01-03)
        adj = adjust_for_splits(df, {"2024-01-03": 2.0})

        # Pre-split rows (Day 1 and Day 2) prices halved, volume doubled
        assert adj.loc["2024-01-01", "close"] == 50.0
        assert adj.loc["2024-01-01", "volume"] == 2000.0

        # Dollar volume preserved exactly
        dollar_vol_orig = df.loc["2024-01-01", "close"] * df.loc["2024-01-01", "volume"]
        dollar_vol_adj = adj.loc["2024-01-01", "close"] * adj.loc["2024-01-01", "volume"]
        assert dollar_vol_orig == dollar_vol_adj == 100_000.0

    def test_dividend_adjustment_3_row_hand_example(self) -> None:
        # Hand 3-row example:
        # Day 1: close = 100
        # Day 2 (ex-div): div = 2.0, close = 100
        # Day 3: close = 102
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 102.0],
                "high": [100.0, 100.0, 102.0],
                "low": [100.0, 100.0, 102.0],
                "close": [100.0, 100.0, 102.0],
                "volume": [1000.0, 1000.0, 1000.0],
            },
            index=dates,
        )

        # Dividend of $2.0 on 2024-01-02
        # Previous close = 100.0 -> factor = 1.0 - 2.0 / 100.0 = 0.98
        adj = adjust_for_dividends(df, {"2024-01-02": 2.0})

        # Day 1 close adjusted to 100 * 0.98 = 98.0
        assert adj.loc["2024-01-01", "close"] == pytest.approx(98.0, rel=1e-10)
        assert adj.loc["2024-01-02", "close"] == 100.0
        assert adj.loc["2024-01-03", "close"] == 102.0
