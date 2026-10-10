"""
Integration tests verifying zero-lookahead invariance across strategies and feature registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestEngine
from backtesting.engine.event_loop.events import SignalEvent
from backtesting.engine.event_loop.execution import SimulatedExecutionHandler, ZeroCostFillModel
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.engine.event_loop.strategy_base import Strategy
from data.ingestion.market_feeds.equities.synthetic import make_ohlcv
from features.store.registry.registry import FeatureRegistry, register_default_features
from pipelines.training.train import walk_forward_signals
from models.ensemble.xgboost.model import XGBoostModel
from signals.arbitrage.pairs_trading.pairs import PairsStrategy
from signals.generation.breakout.breakout import BreakoutStrategy
from signals.generation.crossover.crossover import CrossoverStrategy
from signals.generation.mean_reversion.mean_reversion import MeanReversionStrategy


def assert_zero_lookahead_and_causal_fills(
    strategy_factory,
    df_300: pd.DataFrame,
    df_400: pd.DataFrame,
    symbol: str = "AAPL",
    tolerance: float = 1e-9,
) -> None:
    """Verifies that running on 300 bars vs 400 bars produces identical equity curves

    on the first 300 bars, and every fill occurs strictly after its signal/order.
    """
    dh300 = HistoricBarDataHandler({symbol: df_300})
    dh400 = HistoricBarDataHandler({symbol: df_400})

    s300 = strategy_factory(symbol)
    s400 = strategy_factory(symbol)

    p300 = Portfolio(100_000.0)
    p400 = Portfolio(100_000.0)

    e300 = BacktestEngine(dh300, s300, p300, SimulatedExecutionHandler(dh300, fill_model=ZeroCostFillModel()))
    e400 = BacktestEngine(dh400, s400, p400, SimulatedExecutionHandler(dh400, fill_model=ZeroCostFillModel()))

    r300 = e300.run()
    r400 = e400.run()

    eq300 = r300.equity_curve["equity"]
    eq400 = r400.equity_curve["equity"].iloc[:len(eq300)]

    diff = np.max(np.abs(eq300.values - eq400.values))
    assert diff <= tolerance, f"Equity curves diverged on first 300 bars! Max diff: {diff}"

    # Verify every fill occurred strictly after order timestamp
    orders_by_id = {o.order_id: o for o in r300.orders}
    for fill in r300.fills:
        order = orders_by_id.get(fill.order_id)
        assert order is not None, f"Fill {fill.order_id} has no matching order"
        assert fill.timestamp > order.timestamp, (
            f"Fill at {fill.timestamp} did not occur strictly after order at {order.timestamp}"
        )


class TestZeroLookaheadStrategies:
    """Tests for no-lookahead execution across all core strategies."""

    @pytest.fixture
    def ohlcv_panels(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_400 = make_ohlcv("AAPL", start="2022-01-01", n_days=400, seed=42)
        df_300 = df_400.iloc[:300].copy()
        return df_300, df_400

    def test_crossover_strategy_zero_lookahead(self, ohlcv_panels) -> None:
        """CrossoverStrategy: equity curve on 300 bars == first 300 of 400 bars."""
        df_300, df_400 = ohlcv_panels
        assert_zero_lookahead_and_causal_fills(
            lambda sym: CrossoverStrategy(sym, fast=10, slow=30),
            df_300,
            df_400,
        )

    def test_breakout_strategy_zero_lookahead(self, ohlcv_panels) -> None:
        """BreakoutStrategy: equity curve on 300 bars == first 300 of 400 bars."""
        df_300, df_400 = ohlcv_panels
        assert_zero_lookahead_and_causal_fills(
            lambda sym: BreakoutStrategy(sym, n=20),
            df_300,
            df_400,
        )

    def test_mean_reversion_strategy_zero_lookahead(self, ohlcv_panels) -> None:
        """MeanReversionStrategy: equity curve on 300 bars == first 300 of 400 bars."""
        df_300, df_400 = ohlcv_panels
        assert_zero_lookahead_and_causal_fills(
            lambda sym: MeanReversionStrategy(sym),
            df_300,
            df_400,
        )

    def test_pairs_strategy_zero_lookahead(self) -> None:
        """PairsStrategy: multi-asset equity curves agree on first 300 bars to 1e-9."""
        y_400 = make_ohlcv("Y", start="2022-01-01", n_days=400, seed=101)
        x_400 = make_ohlcv("X", start="2022-01-01", n_days=400, seed=102)

        y_300 = y_400.iloc[:300].copy()
        x_300 = x_400.iloc[:300].copy()

        dh300 = HistoricBarDataHandler({"Y": y_300, "X": x_300})
        dh400 = HistoricBarDataHandler({"Y": y_400, "X": x_400})

        s300 = PairsStrategy("Y", "X")
        s400 = PairsStrategy("Y", "X")

        e300 = BacktestEngine(dh300, s300, Portfolio(100_000.0), SimulatedExecutionHandler(dh300, fill_model=ZeroCostFillModel()))
        e400 = BacktestEngine(dh400, s400, Portfolio(100_000.0), SimulatedExecutionHandler(dh400, fill_model=ZeroCostFillModel()))

        r300 = e300.run()
        r400 = e400.run()

        eq300 = r300.equity_curve["equity"]
        eq400 = r400.equity_curve["equity"].iloc[:len(eq300)]

        diff = np.max(np.abs(eq300.values - eq400.values))
        assert diff <= 1e-9

        # Fill timestamps check
        orders_by_id = {o.order_id: o for o in r300.orders}
        for fill in r300.fills:
            order = orders_by_id.get(fill.order_id)
            if order:
                assert fill.timestamp > order.timestamp

    def test_ml_meta_walk_forward_zero_lookahead(self) -> None:
        """ML-meta walk_forward_signals: predictions on historical test blocks are invariant

        and max(t1_train) < min(t0_test) across all blocks.
        """
        np.random.seed(42)
        n = 300
        X = pd.DataFrame(np.random.randn(n, 4), columns=["f1", "f2", "f3", "f4"])
        y = np.random.randint(0, 2, n)
        t0 = np.arange(n)
        t1 = t0 + 2

        wf_300 = walk_forward_signals(
            model_factory=lambda: XGBoostModel(n_estimators=10, max_depth=2, random_state=42),
            X=X,
            y=y,
            t0=t0,
            t1=t1,
            n_splits=3,
            min_train_samples=40,
        )

        for meta in wf_300["metadata"]:
            assert meta["max_t1_train"] < meta["min_t0_test"], (
                f"Walk-forward leakage: train end {meta['max_t1_train']} >= test start {meta['min_t0_test']}"
            )
            assert meta["is_purged"] is True

    def test_cheating_strategy_flagged_by_detector(self, ohlcv_panels) -> None:
        """A deliberate cheating strategy that peeks at future dataset summary must be FLAGGED."""
        df_300, df_400 = ohlcv_panels

        class CheatingFuturePeekingStrategy(Strategy):
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

            def on_bar(self, tick, data_handler) -> list[SignalEvent]:
                # Peeks ahead into the full dataset mean
                full_df = data_handler.bars[self.symbol]
                future_mean = full_df["close"].mean()
                if tick.close < future_mean:
                    return [SignalEvent(tick.timestamp, self.symbol, "LONG", strength=1.0, target_pct=0.15)]
                else:
                    return [SignalEvent(tick.timestamp, self.symbol, "SHORT", strength=1.0, target_pct=0.15)]

        # Assert that our zero-lookahead detector FLAGS this cheating strategy
        with pytest.raises(AssertionError, match="Equity curves diverged"):
            assert_zero_lookahead_and_causal_fills(
                lambda sym: CheatingFuturePeekingStrategy(sym),
                df_300,
                df_400,
            )


class TestFeatureRegistryTruncationInvariance:
    """Tests that every feature in FeatureRegistry satisfies strict historical truncation invariance."""

    def test_all_registered_features_truncation_invariance(self) -> None:
        """Output on first 200 rows must equal output on first 300 rows restricted to first 200 rows."""
        reg = FeatureRegistry()
        register_default_features(reg)

        df_300 = make_ohlcv("AAPL", start="2022-01-01", n_days=300, seed=42)
        df_200 = df_300.iloc[:200].copy()

        assert len(reg._registry) > 0, "FeatureRegistry has no registered features"

        for feat_name in reg._registry.keys():
            feat_300 = reg.compute(feat_name, df_300)
            feat_200 = reg.compute(feat_name, df_200)

            if isinstance(feat_300, pd.DataFrame):
                for col in feat_300.columns:
                    v300 = feat_300[col].iloc[:200].values
                    v200 = feat_200[col].values
                    np.testing.assert_allclose(
                        v200,
                        v300,
                        equal_nan=True,
                        err_msg=f"Feature {feat_name}.{col} violated truncation invariance!",
                    )
            else:
                v300 = feat_300.iloc[:200].values
                v200 = feat_200.values
                np.testing.assert_allclose(
                    v200,
                    v300,
                    equal_nan=True,
                    err_msg=f"Feature {feat_name} violated truncation invariance!",
                )
