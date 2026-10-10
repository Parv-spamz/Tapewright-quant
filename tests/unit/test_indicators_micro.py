"""Unit tests for technical indicators and market microstructure estimators.

Verifies:
1. Exact hand-computed values (RSI-14 within 1e-9, SMA, ROC, Bollinger, OBV, CMF)
2. Mathematical reference cross-checks (EMA, Historical Volatility, MACD)
3. Invariant properties across 10 random seeds (bounds, ordering)
4. Strict NO-LOOKAHEAD verification across all time-series functions
5. Microstructure spread estimators (Roll, Corwin-Schultz, OBI, Microprice)
6. Kyle's Lambda price impact regression recovery
7. Directional tick runs Shannon entropy (alternating vs i.i.d.)
8. VPIN toxicity index under all-buy vs balanced flow
9. Level-2 order book depth imbalance predictive information coefficient (IC)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from data.ingestion.market_feeds.equities.l2_tape import simulate_l2_tape
from data.ingestion.market_feeds.equities.synthetic import make_ohlcv
from features.market_microstructure.order_book.spreads import (
    amihud_illiquidity,
    corwin_schultz,
    depth_imbalance,
    estimate_kyle_lambda,
    microprice,
    order_book_imbalance,
    roll_spread,
)
from features.market_microstructure.tick_entropy.entropy import (
    shannon_entropy_of_runs,
    vpin,
    vpin_from_signed,
)
from features.technical_indicators.builder import build_indicator_frame
from features.technical_indicators.momentum.momentum import (
    macd,
    roc,
    rsi,
    stochastic,
)
from features.technical_indicators.trend.trend import (
    adx,
    ema,
    sma,
    supertrend,
)
from features.technical_indicators.volatility.volatility import (
    atr,
    bollinger,
    historical_vol,
    keltner,
    true_range,
)
from features.technical_indicators.volume.volume import (
    chaikin_money_flow,
    obv,
    vwap,
)


class TestHandComputedExactValues:
    """Exact arithmetic verification against hand-calculated values."""

    def test_rsi_14_first_value_exact_plain_python_arithmetic(self) -> None:
        """15-row hand-written close series with exact arithmetic match within 1e-9."""
        # Hand-crafted 15 close prices
        closes = [
            100.0, 102.0, 101.0, 104.0, 103.5,
            105.0, 104.0, 106.5, 108.0, 107.0,
            109.0, 108.5, 110.0, 111.0, 113.0,
        ]
        s = pd.Series(closes)
        computed = rsi(s, n=14)

        # Plain Python arithmetic:
        # 14 price changes from index 1 to 14
        diffs = [closes[i] - closes[i - 1] for i in range(1, 15)]
        gains = [d if d > 0.0 else 0.0 for d in diffs]
        losses = [-d if d < 0.0 else 0.0 for d in diffs]

        avg_gain = sum(gains) / 14.0
        avg_loss = sum(losses) / 14.0
        rs = avg_gain / avg_loss
        expected_rsi_14 = 100.0 - (100.0 / (1.0 + rs))

        # First 14 rows (0 to 13) are NaN
        assert computed.iloc[:14].isna().all()

        # Row 14 matches hand math within 1e-9
        first_val = float(computed.iloc[14])
        assert abs(first_val - expected_rsi_14) < 1e-9

    def test_sma_small_hand_example(self) -> None:
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        res = sma(s, n=3)
        assert res.iloc[:2].isna().all()
        assert res.iloc[2] == pytest.approx(20.0)
        assert res.iloc[3] == pytest.approx(30.0)
        assert res.iloc[4] == pytest.approx(40.0)

    def test_roc_small_hand_example(self) -> None:
        s = pd.Series([100.0, 110.0, 120.0, 105.0])
        res = roc(s, n=1)
        assert pd.isna(res.iloc[0])
        assert res.iloc[1] == pytest.approx(0.10)
        assert res.iloc[2] == pytest.approx(120.0 / 110.0 - 1.0)
        assert res.iloc[3] == pytest.approx(105.0 / 120.0 - 1.0)

    def test_bollinger_small_hand_example(self) -> None:
        s = pd.Series([10.0, 12.0, 14.0, 16.0])
        res = bollinger(s, n=2, k=2.0)
        # At index 1: window [10, 12], mean=11, std(ddof=0)=1.0
        assert res["mid"].iloc[1] == pytest.approx(11.0)
        assert res["upper"].iloc[1] == pytest.approx(13.0)
        assert res["lower"].iloc[1] == pytest.approx(9.0)
        assert res["pctb"].iloc[1] == pytest.approx(0.75)
        assert res["bandwidth"].iloc[1] == pytest.approx(4.0 / 11.0)

    def test_obv_small_hand_example(self) -> None:
        c = pd.Series([100.0, 105.0, 102.0, 108.0])
        v = pd.Series([100.0, 200.0, 150.0, 300.0])
        res = obv(c, v)
        # index 0: 0, index 1: +200, index 2: 200 - 150 = 50, index 3: 50 + 300 = 350
        assert list(res.values) == [0.0, 200.0, 50.0, 350.0]

    def test_cmf_small_hand_example(self) -> None:
        # Row 1: H=105, L=95, C=105 -> (10-0)/10 = +1.0, V=100 -> MFV=+100
        # Row 2: H=110, L=100, C=100 -> (0-10)/10 = -1.0, V=100 -> MFV=-100
        df = pd.DataFrame({
            "high": [105.0, 110.0],
            "low": [95.0, 100.0],
            "close": [105.0, 100.0],
            "volume": [100.0, 100.0],
        })
        res = chaikin_money_flow(df, n=2)
        # window 2 sum: MFV = 0.0, Vol = 200.0 -> CMF = 0.0
        assert res.iloc[1] == pytest.approx(0.0)


class TestReferenceCrossChecks:
    """Cross-check implementations against standard formulas and pandas references."""

    def test_ema_vs_pandas_ewm(self) -> None:
        df = make_ohlcv("AAPL", start="2023-01-01", n_days=60, seed=10)
        close = df["close"]
        my_ema = ema(close, n=20)
        ref_ema = close.ewm(span=20, adjust=False).mean()
        pd.testing.assert_series_equal(my_ema, ref_ema)

    def test_historical_vol_vs_reference(self) -> None:
        df = make_ohlcv("AAPL", start="2023-01-01", n_days=60, seed=11)
        close = df["close"]
        my_hv = historical_vol(close, n=21)
        ref_hv = np.log(close).diff().rolling(21).std(ddof=1) * np.sqrt(252.0)
        pd.testing.assert_series_equal(my_hv, ref_hv, check_names=False)

    def test_macd_vs_two_ewm_spans(self) -> None:
        df = make_ohlcv("AAPL", start="2023-01-01", n_days=60, seed=12)
        close = df["close"]
        res = macd(close, fast=12, slow=26, signal=9)
        ref_macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        ref_sig = ref_macd.ewm(span=9, adjust=False).mean()
        ref_hist = ref_macd - ref_sig

        pd.testing.assert_series_equal(res["macd"], ref_macd, check_names=False)
        pd.testing.assert_series_equal(res["signal"], ref_sig, check_names=False)
        pd.testing.assert_series_equal(res["hist"], ref_hist, check_names=False)


class TestInvariantPropertiesAcrossSeeds:
    """Property tests over 10 distinct random seeds."""

    @pytest.mark.parametrize("seed", list(range(100, 110)))
    def test_invariants_across_seeds(self, seed: int) -> None:
        df = make_ohlcv("AAPL", start="2023-01-01", n_days=80, seed=seed)
        close = df["close"]

        # RSI in [0, 100]
        r = rsi(close, n=14).dropna()
        assert (r >= 0.0).all() and (r <= 100.0).all()

        # Stochastic in [0, 100]
        stoch = stochastic(df, k=14, d=3).dropna()
        assert (stoch["stoch_k"] >= 0.0).all() and (stoch["stoch_k"] <= 100.0).all()
        assert (stoch["stoch_d"] >= 0.0).all() and (stoch["stoch_d"] <= 100.0).all()

        # ADX in [0, 100]
        adx_res = adx(df, n=14).dropna()
        assert (adx_res["adx"] >= 0.0).all() and (adx_res["adx"] <= 100.0).all()

        # Bollinger ordering: upper >= mid >= lower
        bb = bollinger(close, n=20, k=2.0).dropna()
        assert (bb["upper"] >= bb["mid"] - 1e-12).all()
        assert (bb["mid"] >= bb["lower"] - 1e-12).all()

        # ATR >= 0
        atr_val = atr(df, n=14)
        assert (atr_val >= 0.0).all()

        # CMF in [-1, 1]
        cmf_val = chaikin_money_flow(df, n=20).dropna()
        assert (cmf_val >= -1.0).all() and (cmf_val <= 1.0).all()


class TestNoLookaheadInvariance:
    """Strict test: output on first 200 rows equals output on first 300 rows restricted to 200."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_300 = make_ohlcv("AAPL", start="2023-01-01", n_days=300, seed=42)
        df_200 = df_300.iloc[:200].copy()
        return df_200, df_300

    def test_no_lookahead_builder_all_indicators(
        self,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        df_200, df_300 = sample_data
        b_200 = build_indicator_frame(df_200)
        b_300 = build_indicator_frame(df_300)

        for col in b_200.columns:
            np.testing.assert_allclose(
                b_200[col].values,
                b_300[col].iloc[:200].values,
                equal_nan=True,
                err_msg=f"Lookahead detected in indicator column {col}",
            )

    def test_no_lookahead_microstructure_estimators(self) -> None:
        book_300, _ = simulate_l2_tape(n_steps=300, seed=42)
        book_200 = book_300.iloc[:200].copy()

        # Microprice
        mp_200 = microprice(book_200).values
        mp_300 = microprice(book_300).iloc[:200].values
        np.testing.assert_allclose(mp_200, mp_300, equal_nan=True)

        # Depth imbalance
        di_200 = depth_imbalance(book_200).values
        di_300 = depth_imbalance(book_300).iloc[:200].values
        np.testing.assert_allclose(di_200, di_300, equal_nan=True)

        # Roll spread
        rs_200 = roll_spread(book_200["mid"], window=30).values
        rs_300 = roll_spread(book_300["mid"], window=30).iloc[:200].values
        np.testing.assert_allclose(rs_200, rs_300, equal_nan=True)

        # Amihud illiquidity
        ret_300 = book_300["mid"].pct_change()
        vol_300 = book_300["bid_sz_1"] + book_300["ask_sz_1"]
        ret_200 = ret_300.iloc[:200]
        vol_200 = vol_300.iloc[:200]
        am_200 = amihud_illiquidity(ret_200, vol_200, window=20).values
        am_300 = amihud_illiquidity(ret_300, vol_300, window=20).iloc[:200].values
        np.testing.assert_allclose(am_200, am_300, equal_nan=True)


class TestMicrostructureSpreadEstimators:
    """Verification of Roll, Corwin-Schultz, OBI, Microprice, and Kyle Lambda."""

    def test_roll_spread_recovers_s_on_alternating_bounce_series(self) -> None:
        """Constructed bounce series with dP = +-s/2 recovers s within 5%."""
        s_true = 0.50
        # Price change alternates +s/2, -s/2, yielding cov(dP_t, dP_{t-1}) = -s^2 / 4
        # Roll spread = 2 * sqrt(s^2 / 4) = s
        dp = np.array([s_true / 2.0, -s_true / 2.0] * 500)
        p = np.cumsum(dp) + 100.0

        est_s = roll_spread(p)
        assert abs(est_s - s_true) / s_true < 0.05

    def test_corwin_schultz_finite_and_non_negative(self) -> None:
        df = make_ohlcv("AAPL", start="2023-01-01", n_days=100, seed=77)
        cs = corwin_schultz(df["high"], df["low"])
        assert not cs.isna().all()
        assert (cs >= 0.0).all()
        assert np.isfinite(cs.values).all()

    def test_order_book_imbalance_bounds_and_zero_equality(self) -> None:
        assert order_book_imbalance(500.0, 500.0) == 0.0
        assert order_book_imbalance(100.0, 0.0) == 1.0
        assert order_book_imbalance(0.0, 100.0) == -1.0
        assert order_book_imbalance(0.0, 0.0) == 0.0

        b = pd.Series([100.0, 200.0, 0.0])
        a = pd.Series([100.0, 0.0, 300.0])
        obi = order_book_imbalance(b, a)
        assert list(obi.values) == [0.0, 1.0, -1.0]

    def test_microprice_strictly_between_best_bid_and_best_ask(self) -> None:
        book_df, _ = simulate_l2_tape(n_steps=100, seed=42)
        mp = microprice(book_df)
        b_px = book_df["bid_px_1"]
        a_px = book_df["ask_px_1"]

        assert (mp >= b_px - 1e-12).all()
        assert (mp <= a_px + 1e-12).all()

    def test_kyle_lambda_simulation_recovery(self) -> None:
        """Simulate dP = 0.002 * Q + noise -> lambda within 10% and t_stat > 10."""
        rng = np.random.default_rng(42)
        n = 5000
        q = rng.normal(0.0, 100.0, size=n)
        noise = rng.normal(0.0, 0.05, size=n)
        dp = 0.002 * q + noise

        lam, t_stat = estimate_kyle_lambda(dp, q)
        assert abs(lam - 0.002) / 0.002 < 0.10
        assert t_stat > 10.0


class TestTickEntropyAndVPIN:
    """Shannon entropy on runs and Volume-Synchronized Probability of Toxicity."""

    def test_shannon_entropy_alternating_vs_iid(self) -> None:
        """Alternating +-1 entropy < 0.1 bit; i.i.d. random (n=20000, word_len 2) > 1.9 bits."""
        # Alternating +-1 sequence (deterministic runs)
        alt_signs = np.array([1, -1] * 2000)
        h_alt = shannon_entropy_of_runs(alt_signs, word_len=2)
        assert h_alt < 0.1

        # i.i.d. random sequence (n=20000, word_len=2)
        rng = np.random.default_rng(42)
        rand_signs = rng.choice([-1, 1], size=20000)
        h_rand = shannon_entropy_of_runs(rand_signs, word_len=2)
        assert h_rand > 1.9

    def test_vpin_invariants_and_toxicity_separation(self) -> None:
        """VPIN in [0, 1]; all-buys tape VPIN approaches 1; balanced flow yields low VPIN."""
        # 1. All-buys tape
        n_trades = 1000
        df_buys = pd.DataFrame({
            "price": [100.0 + i * 0.01 for i in range(n_trades)],
            "size": [50.0] * n_trades,
            "side": [1] * n_trades,
        })
        vpin_buys = vpin_from_signed(df_buys, bucket_volume=100.0, n_buckets=20).dropna()
        final_buys_vpin = float(vpin_buys.iloc[-1])
        assert final_buys_vpin >= 0.95
        assert (vpin_buys >= 0.0).all() and (vpin_buys <= 1.0).all()

        # 2. Balanced flow tape (alternating buys and sells)
        df_bal = pd.DataFrame({
            "price": [100.0] * n_trades,
            "size": [50.0] * n_trades,
            "side": [1, -1] * (n_trades // 2),
        })
        vpin_bal = vpin_from_signed(df_bal, bucket_volume=100.0, n_buckets=20).dropna()
        final_bal_vpin = float(vpin_bal.iloc[-1])
        assert final_bal_vpin <= 0.05
        assert final_buys_vpin - final_bal_vpin > 0.85

    def test_l2_depth_imbalance_positive_spearman_ic(self) -> None:
        """Depth imbalance has positive Spearman IC with next-step return on L2 tape."""
        book_df, _ = simulate_l2_tape(n_steps=5000, seed=42)
        di = depth_imbalance(book_df, levels=5)
        next_ret = book_df["mid"].pct_change().shift(-1)

        valid = (~di.isna()) & (~next_ret.isna())
        corr, pval = spearmanr(di[valid], next_ret[valid])

        # Report value honestly
        print(f"\n[Proof Result] L2 Depth Imbalance Spearman IC: {corr:.4f} (p={pval:.2e})")
        assert corr > 0.20
        assert pval < 1e-50
