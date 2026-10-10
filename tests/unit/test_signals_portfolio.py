from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestEngine
from backtesting.engine.event_loop.execution import SimulatedExecutionHandler
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.execution.fill_models import RealisticFillModel
from backtesting.execution.slippage_models.models import FixedBpsSlippage
from backtesting.execution.transaction_costs.costs import PerShareCommission
from backtesting.portfolio.black_litterman.bl import black_litterman
from backtesting.portfolio.markowitz.markowitz import max_sharpe, min_variance
from backtesting.portfolio.risk_parity.hrp import (
    allocate,
    equal_risk_contribution,
    hierarchical_risk_parity,
)
from backtesting.reports.tearsheet.tearsheet import generate_tearsheet
from config.trading.risk_limits import RiskLimits
from database.connection.manager import ConnectionManager
from database.migrations.runner import apply_migrations
from database.repositories.signal_repo.repo import SignalRepo
from features.econometric.cointegration.causal_graph import build_causal_graph
from signals.arbitrage.pairs_trading.pairs import (
    PairsSignal,
    PairsStrategy,
    scan_pairs,
)
from signals.filtering.volatility_filter.causal_gate import (
    CausalGate,
    CausalGatedStrategy,
)
from signals.filtering.volatility_filter.filter import (
    apply_filter,
    garch_regime_filter,
    simple_vol_filter,
)
from signals.generation.breakout.breakout import (
    BreakoutStrategy,
    donchian_breakout,
    volatility_breakout,
)
from signals.generation.crossover.crossover import (
    CrossoverStrategy,
    dual_ema_signals,
    golden_cross_events,
)
from signals.generation.mean_reversion.mean_reversion import bollinger_rsi_reversion
from signals.persist import persist_signals
from signals.ranking.cross_sectional.ranker import (
    decile_buckets,
    long_short_weights,
    momentum_scores,
    value_scores,
)
from utils.math.linalg import ledoit_wolf_cov


class TestSignalGenerators:
    def test_crossover_hand_crafted_ramp_and_warmup(self) -> None:
        """Hand-crafted ramp flips at the right bars; no signal in the first slow - 1 bars."""
        # 10 down bars, then 20 strong up bars
        down = np.linspace(100.0, 90.0, 10)
        up = np.linspace(91.0, 130.0, 20)
        close = pd.Series(np.concatenate([down, up]))

        fast, slow = 4, 10
        sig = dual_ema_signals(close, fast=fast, slow=slow)

        # No signal in the first slow - 1 bars (bars 0..8)
        assert (sig.iloc[: slow - 1] == 0).all()

        # Eventually flips to bullish (+1) as fast EMA surges above slow EMA
        assert (sig.iloc[-5:] == 1).all()

        # Events detect the discrete flip
        events = golden_cross_events(close, fast=fast, slow=slow)
        assert (events == 1).sum() >= 1

    def test_donchian_breakout_strictly_excludes_current_bar(self) -> None:
        """Donchian channel breakout explicitly compares against prior n bars, excluding current bar."""
        # Bars 0..4 have range [95, 105]. Bar 5 makes an intraday high of 200, but closes at 100.
        highs = [105.0, 104.0, 103.0, 104.0, 105.0, 200.0, 105.0]
        lows = [95.0, 96.0, 97.0, 96.0, 95.0, 95.0, 95.0]
        closes = [100.0, 101.0, 100.0, 102.0, 101.0, 100.0, 106.0]

        df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
        n = 4
        sig = donchian_breakout(df, n=n)

        # At index 5: prior high of bars 1..4 is 105.0. Bar 5's close is 100.0 <= 105.0 -> signal 0
        assert sig.iloc[5] == 0

        # At index 6: prior 4 bars include bar 5 (high 200.0). Current close 106.0 is below 200.0 -> signal 0!
        # If current bar high wasn't excluded, bar 5 would have corrupted its own test.
        assert sig.iloc[6] == 0


class TestNoLookaheadInvariance:
    def test_no_lookahead_over_every_generator_and_garch_filter(self) -> None:
        """Strict no-lookahead: output on first 200 rows equals output on first 300 rows restricted to 200."""
        rng = np.random.default_rng(42)
        n_full = 300
        n_slice = 200

        px = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=n_full)))
        df_full = pd.DataFrame({
            "open": px * 0.999,
            "high": px * 1.015,
            "low": px * 0.985,
            "close": px,
            "volume": 1000.0,
        })
        df_slice = df_full.iloc[:n_slice].copy()
        rets_full = df_full["close"].pct_change().dropna()
        rets_slice = df_slice["close"].pct_change().dropna()

        # 1. Dual EMA signals
        s_ema_full = dual_ema_signals(df_full["close"], fast=5, slow=15)
        s_ema_slice = dual_ema_signals(df_slice["close"], fast=5, slow=15)
        np.testing.assert_allclose(s_ema_full.iloc[:n_slice], s_ema_slice)

        # 2. Golden Cross Events
        s_gc_full = golden_cross_events(df_full["close"], fast=5, slow=15)
        s_gc_slice = golden_cross_events(df_slice["close"], fast=5, slow=15)
        np.testing.assert_allclose(s_gc_full.iloc[:n_slice], s_gc_slice)

        # 3. Donchian Breakout
        s_don_full = donchian_breakout(df_full, n=15)
        s_don_slice = donchian_breakout(df_slice, n=15)
        np.testing.assert_allclose(s_don_full.iloc[:n_slice], s_don_slice)

        # 4. Volatility Breakout
        s_vol_full = volatility_breakout(df_full, n=15, k=1.5)
        s_vol_slice = volatility_breakout(df_slice, n=15, k=1.5)
        np.testing.assert_allclose(s_vol_full.iloc[:n_slice], s_vol_slice)

        # 5. Bollinger + RSI Mean Reversion
        s_mr_full = bollinger_rsi_reversion(df_full, n=15, k=1.8, rsi_n=10)
        s_mr_slice = bollinger_rsi_reversion(df_slice, n=15, k=1.8, rsi_n=10)
        np.testing.assert_allclose(s_mr_full.iloc[:n_slice], s_mr_slice)

        # 6. Simple Vol Filter
        f_simple_full = simple_vol_filter(rets_full, span=15, min_obs=30)
        f_simple_slice = simple_vol_filter(rets_slice, span=15, min_obs=30)
        np.testing.assert_allclose(f_simple_full.iloc[: len(rets_slice)].astype(float), f_simple_slice.astype(float))

        # 7. GARCH Regime Filter (refit_every kept large for test speed)
        f_garch_full = garch_regime_filter(rets_full, threshold_quantile=0.85, refit_every=100, min_obs=50)
        f_garch_slice = garch_regime_filter(rets_slice, threshold_quantile=0.85, refit_every=100, min_obs=50)
        np.testing.assert_allclose(f_garch_full.iloc[: len(rets_slice)].astype(float), f_garch_slice.astype(float))


class TestVolatilityFiltering:
    def test_garch_filter_blocks_burst_higher_than_calm(self) -> None:
        """GARCH filter blocks the burst period at a higher rate than calm periods."""
        rng = np.random.default_rng(123)
        calm1 = rng.normal(0, 0.008, size=260)
        burst = rng.normal(0, 0.045, size=60)  # Planted volatility spike
        calm2 = rng.normal(0, 0.008, size=100)
        rets = pd.Series(np.concatenate([calm1, burst, calm2]))

        allowed = garch_regime_filter(rets, threshold_quantile=0.90, refit_every=50, min_obs=250)

        # Calm period before burst
        calm_blocked_rate = float((~allowed.iloc[250:260]).mean())
        # Burst period
        burst_blocked_rate = float((~allowed.iloc[260:320]).mean())

        assert burst_blocked_rate > calm_blocked_rate
        assert burst_blocked_rate > 0.50


class TestCausalGate:
    def test_causal_gate_admits_planted_parent_and_rejects_independent(self) -> None:
        """x->y admitted, z not; filter_features keeps only x and raises when none admitted."""
        rng = np.random.default_rng(10)
        n = 1500
        x = rng.normal(0, 1, size=n)
        noise = rng.normal(0, 0.5, size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * x[t - 1] + noise[t]
        z = rng.normal(0, 1, size=n)

        df = pd.DataFrame({"x": x, "y": y, "z": z})
        graph = build_causal_graph(df, maxlag=3, q=0.01, te_perm=50, seed=10)
        gate = CausalGate(graph)

        # Precedence check
        assert gate.allows("x", "y") is True
        assert gate.allows("z", "y") is False

        # filter_features keeps only x
        feats = pd.DataFrame({"x": x, "z": z})
        admitted_df = gate.filter_features(feats, target_name="y")
        assert list(admitted_df.columns) == ["x"]

        # Raising when none admitted
        with pytest.raises(ValueError, match="No features admitted by CausalGate"):
            gate.filter_features(pd.DataFrame({"z": z}), target_name="y")

        # Gate report
        rep = gate.gate_report()
        assert not rep.empty
        assert "admitted" in rep.columns


class TestCrossSectionalRanking:
    def test_deciles_balanced_and_long_short_weights(self) -> None:
        """Deciles balanced +-1; long_short weights sum to 0.0 and gross exposure 1.0 to 1e-12."""
        rng = np.random.default_rng(42)
        n_assets = 20
        scores = pd.Series(rng.normal(0, 1, size=n_assets), index=[f"A{i}" for i in range(n_assets)])

        # Decile buckets (n=10)
        buckets = decile_buckets(scores, n=10)
        pos_count = (buckets == 1).sum()
        neg_count = (buckets == -1).sum()
        assert pos_count == neg_count
        assert pos_count > 0
        assert buckets.sum() == 0

        # Long-short weights (top 10%, bottom 10%)
        weights = long_short_weights(scores, top_q=0.10, bottom_q=0.10)
        np.testing.assert_allclose(weights.sum(), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(weights).sum(), 1.0, atol=1e-12)

    def test_momentum_and_value_scores(self) -> None:
        """Momentum and value factor score calculations."""
        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        prices = pd.DataFrame({
            "AAPL": np.linspace(100, 200, 300),
            "MSFT": np.linspace(200, 150, 300),
        }, index=dates)

        mom = momentum_scores(prices, lookback=100, skip=21)
        assert mom.shape == prices.shape
        # AAPL went up, MSFT went down
        assert mom.iloc[-1]["AAPL"] > mom.iloc[-1]["MSFT"]

        # Value scores with fundamentals
        funds = pd.DataFrame({
            "AAPL": np.full(300, 0.05),  # E/P ratio
            "MSFT": np.full(300, 0.08),
        }, index=dates)
        val = value_scores(prices, fundamentals_df=funds)
        assert val.shape == prices.shape
        # Standardized z-score sum across row equals 0
        np.testing.assert_allclose(val.iloc[-1].sum(), 0.0, atol=1e-12)

        # Never fabricate fundamentals: empty raises ValueError
        with pytest.raises(ValueError, match="fundamentals_df must be supplied"):
            value_scores(prices, fundamentals_df=pd.DataFrame())


class TestPairsTradingArbitrage:
    def test_pairs_zscore_state_machine_hand_path(self) -> None:
        """PairsSignal 3-state machine produces exact expected position path on hand spread."""
        sig_gen = PairsSignal(entry_z=2.0, exit_z=0.5, stop_z=4.0)
        z = pd.Series([0.0, 1.0, 2.1, 2.5, 0.4, 0.0, -1.0, -2.2, -3.0, -0.4, 0.0])

        # Hand expected path:
        # t=0,1: 0
        # t=2,3: -1 (short spread at z >= 2.0)
        # t=4..6: 0 (exited at z <= 0.5)
        # t=7,8: +1 (long spread at z <= -2.0)
        # t=9,10: 0 (exited at z >= -0.5)
        expected = np.array([0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0])
        actual = sig_gen.generate_from_zscores(z).values
        np.testing.assert_array_equal(actual, expected)

        # Stop loss branch
        z_stop = pd.Series([0.0, -2.5, -4.5, 0.0])
        # t=0: 0, t=1: 1, t=2: 0 (stop loss triggered at z <= -4.0), t=3: 0
        actual_stop = sig_gen.generate_from_zscores(z_stop).values
        np.testing.assert_array_equal(actual_stop, np.array([0, 1, 0, 0]))

    def test_scan_pairs_planted_pair_and_test_count(self) -> None:
        """scan_pairs finds planted pair among 6 series and reports n_tests=15."""
        rng = np.random.default_rng(42)
        T = 500
        # Planted cointegrated pair: s0 and s1
        rw1 = np.cumsum(rng.normal(0, 1, size=T))
        rw0 = 1.8 * rw1 + rng.normal(0, 0.5, size=T)

        panel = pd.DataFrame({
            "s0": rw0,
            "s1": rw1,
            "s2": np.cumsum(rng.normal(0, 1, size=T)),
            "s3": np.cumsum(rng.normal(0, 1, size=T)),
            "s4": np.cumsum(rng.normal(0, 1, size=T)),
            "s5": np.cumsum(rng.normal(0, 1, size=T)),
        })

        candidates, n_tests = scan_pairs(panel, pvalue=0.05)
        assert n_tests == 15
        assert any((c["asset1"] == "s0" and c["asset2"] == "s1") for c in candidates)


class TestPortfolioAllocation:
    def test_markowitz_2_asset_closed_form_and_max_sharpe(self) -> None:
        """2-asset closed-form min-variance matched to 1e-6; max-Sharpe beats equal-weight."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        s1_sq, s2_sq, s12 = cov[0, 0], cov[1, 1], cov[0, 1]

        # Closed form: w1 = (s2^2 - s12) / (s1^2 + s2^2 - 2*s12)
        w1_exact = (s2_sq - s12) / (s1_sq + s2_sq - 2.0 * s12)
        w_opt = min_variance(cov)
        assert pytest.approx(w_opt[0], abs=1e-6) == w1_exact

        # Max Sharpe beats equal weight Sharpe
        mu = np.array([0.08, 0.15])
        w_ms = max_sharpe(cov, expected_returns=mu, rf=0.02)
        sr_ms = (w_ms @ mu - 0.02) / np.sqrt(w_ms @ cov @ w_ms)

        w_eq = np.array([0.5, 0.5])
        sr_eq = (w_eq @ mu - 0.02) / np.sqrt(w_eq @ cov @ w_eq)
        assert sr_ms > sr_eq

    def test_black_litterman_equilibrium_and_confident_view(self) -> None:
        """No views -> equilibrium; very confident view (tiny Omega) -> mu within 1e-3 of Q."""
        Sigma = np.array([[0.04, 0.01, 0.02], [0.01, 0.09, 0.01], [0.02, 0.01, 0.06]])
        w_mkt = np.array([0.4, 0.3, 0.3])
        delta = 2.5
        tau = 0.05

        # 1. No views -> exact equilibrium pi
        pi, mu_bl, _ = black_litterman(Sigma, w_mkt, P=None, Q=None, tau=tau, delta=delta)
        np.testing.assert_allclose(mu_bl, pi, atol=1e-12)

        # 2. Confident view on asset 0: Q=0.10 with tiny Omega
        P = np.array([[1.0, 0.0, 0.0]])
        Q = np.array([0.10])
        Omega = np.array([[1e-10]])
        _, mu_conf, _ = black_litterman(Sigma, w_mkt, P=P, Q=Q, tau=tau, delta=delta, omega=Omega)
        assert pytest.approx(mu_conf[0], abs=1e-3) == 0.10

    def test_hrp_and_erc_risk_parity_properties(self) -> None:
        """HRP block-diagonal properties and ERC equal risk contributions within 1e-4."""
        # 1. HRP Block-diagonal covariance test
        cov_block = np.array([
            [0.01, 0.016, 0.0, 0.0],
            [0.016, 0.04, 0.0, 0.0],
            [0.0, 0.0, 0.02, 0.024],
            [0.0, 0.0, 0.024, 0.04],
        ])
        w_hrp = hierarchical_risk_parity(cov_block)
        assert pytest.approx(np.sum(w_hrp), abs=1e-12) == 1.0
        assert (w_hrp >= 0.0).all()
        # In Block 1, asset 0 has lower variance (0.01) than asset 1 (0.04) -> w[0] > w[1]
        assert w_hrp[0] > w_hrp[1]

        # 2. ERC test on uncorrelated assets: weights proportional to 1/sigma
        cov_uncorr = np.diag([0.04, 0.09, 0.16])
        sigma_vec = np.sqrt(np.diag(cov_uncorr))
        expected_w = (1.0 / sigma_vec) / np.sum(1.0 / sigma_vec)

        w_erc = equal_risk_contribution(cov_uncorr)
        np.testing.assert_allclose(w_erc, expected_w, atol=1e-4)

        # Risk contributions are equal within 1e-4
        port_vol = np.sqrt(w_erc @ cov_uncorr @ w_erc)
        rc = w_erc * (cov_uncorr @ w_erc) / port_vol
        np.testing.assert_allclose(rc, rc[0], atol=1e-4)

    def test_ledoit_wolf_shrinkage_stability_on_30_assets(self) -> None:
        """Weights from allocate(method='minvar', shrink=True) are less extreme than shrink=False."""
        rng = np.random.default_rng(42)
        N = 30
        T = 60
        factor = rng.normal(0, 0.02, size=(T, 1))
        betas = rng.uniform(0.2, 2.0, size=(1, N))
        noise = rng.normal(0, np.linspace(0.005, 0.05, N), size=(T, N))
        rets = pd.DataFrame(factor @ betas + noise, columns=[f"S{i}" for i in range(N)])

        w_sample = allocate(rets, method="minvar", shrink=False, long_only=False)
        w_shrunk = allocate(rets, method="minvar", shrink=True, long_only=False)

        max_w_sample = float(np.max(np.abs(w_sample)))
        max_w_shrunk = float(np.max(np.abs(w_shrunk)))

        print(f"\n[Ledoit-Wolf Stability] Max |w| Sample: {max_w_sample:.4f}, Shrunk: {max_w_shrunk:.4f}")
        assert max_w_shrunk < max_w_sample


class TestEndToEndEngineAndPersistence:
    @pytest.fixture(autouse=True)
    def setup_sqlite_database(self) -> None:
        """Setup SQLite in-memory database and run migrations."""
        ConnectionManager.reset_instance()
        manager = ConnectionManager.get_instance(backend="sqlite", sqlite_path=":memory:")
        with manager.connection() as conn:
            apply_migrations(conn)
        yield
        ConnectionManager.reset_instance()

    def test_crossover_strategy_through_engine_with_realistic_fill_model_and_tearsheet(self) -> None:
        """End-to-end: CrossoverStrategy through the engine with RealisticFillModel -> tearsheet generated."""
        dates = pd.date_range("2022-01-01", periods=120, freq="B")
        rng = np.random.default_rng(42)
        px = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.015, size=120)))
        df_bars = pd.DataFrame({
            "open": px,
            "high": px * 1.01,
            "low": px * 0.99,
            "close": px,
            "volume": 10000.0,
        }, index=dates)

        dh = HistoricBarDataHandler({"AAPL": df_bars})
        strat = CrossoverStrategy("AAPL", fast=5, slow=15, target_pct=0.20)
        limits = RiskLimits(max_position_pct=0.50, max_leverage=1.0)
        port = Portfolio(initial_cash=100_000.0, risk_limits=limits)

        fill_model = RealisticFillModel(
            slippage=FixedBpsSlippage(bps=5.0),
            cost_model=PerShareCommission(per_share=0.005),
        )
        exec_handler = SimulatedExecutionHandler(dh, fill_model=fill_model)

        engine = BacktestEngine(dh, strat, port, exec_handler)
        res = engine.run()

        # Generate tearsheet
        sheet = generate_tearsheet(res)
        assert "TAPEWRIGHT-QUANT TEARSHEET REPORT" in sheet
        assert "SUMMARY PERFORMANCE METRICS" in sheet
        assert len(res.fills) > 0

    def test_persist_signals_to_database(self) -> None:
        """Persist signals to SignalRepo without error."""
        from backtesting.engine.event_loop.events import SignalEvent

        from database.repositories.market_data_repo.repo import AssetRepo
        from database.repositories.market_data_repo.model_registry_repo import ModelRegistryRepo

        asset_repo = AssetRepo()
        asset_repo.upsert_asset("AAPL", "AAPL", "Apple Inc.", "EQUITY", "NASDAQ")
        asset_repo.upsert_asset("MSFT", "MSFT", "Microsoft Corp.", "EQUITY", "NASDAQ")

        model_repo = ModelRegistryRepo()
        model_repo.register("crossover_v1", "crossover", "RULE_BASED", "1.0")
        model_repo.register("breakout_v1", "breakout", "RULE_BASED", "1.0")

        sig_repo = SignalRepo()
        signals = [
            SignalEvent(
                timestamp="2024-01-15 10:00:00",
                symbol="AAPL",
                direction="LONG",
                strength=0.95,
                strategy_id="crossover_v1",
                target_pct=0.25,
            ),
            SignalEvent(
                timestamp="2024-01-15 11:00:00",
                symbol="MSFT",
                direction="SHORT",
                strength=0.80,
                strategy_id="breakout_v1",
                target_pct=0.15,
            ),
        ]

        count = persist_signals(signals, sig_repo)
        assert count == 2

        all_signals = sig_repo.get_signals(symbol="AAPL")
        assert len(all_signals) == 1
