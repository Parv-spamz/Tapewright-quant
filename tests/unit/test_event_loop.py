from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestEngine
from backtesting.engine.event_loop.events import (
    FillEvent,
    OrderEvent,
    SignalEvent,
    TickEvent,
)
from backtesting.engine.event_loop.execution import (
    SimulatedExecutionHandler,
    ZeroCostFillModel,
)
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.engine.event_loop.strategy_base import (
    BuyAndHoldStrategy,
    FixedScheduleStrategy,
    Strategy,
)
from config.trading.risk_limits import RiskLimits
from database.connection.manager import ConnectionManager
from database.migrations.runner import apply_migrations
from database.repositories.order_repo.repo import OrderRepo


class TestDiscreteEventLoop:
    """Comprehensive test suite for the discrete event-driven backtesting engine."""

    def test_buy_and_hold_hand_5_bar_panel(self) -> None:
        """Buy & hold on a hand 5-bar panel: fill at bar 2's OPEN; final equity matches hand value exactly."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df_aapl = pd.DataFrame(
            {
                "open": [100.0, 102.0, 106.0, 108.0, 110.0],
                "high": [105.0, 108.0, 110.0, 112.0, 115.0],
                "low": [95.0, 101.0, 104.0, 107.0, 109.0],
                "close": [100.0, 106.0, 108.0, 110.0, 112.0],
                "volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            },
            index=dates,
        )

        handler = HistoricBarDataHandler({"AAPL": df_aapl})
        strategy = BuyAndHoldStrategy(symbol="AAPL", target_pct=0.5)
        portfolio = Portfolio(initial_cash=10000.0, risk_limits=RiskLimits(max_position_pct=1.0))
        exec_handler = SimulatedExecutionHandler(handler, fill_model=ZeroCostFillModel())

        engine = BacktestEngine(
            data_handler=handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=exec_handler,
        )

        res = engine.run()

        # Hand calculations:
        # Bar 0 (2024-01-01): Close = 100.0.
        # Strategy signals target_pct = 0.5 -> Target value = $5,000 -> 50 shares @ $100.
        # Bar 1 (2024-01-02, Bar 2 in 1-based): Pending order fills at OPEN = $102.0.
        # Cost = 50 * $102.0 = $5,100. Cash remaining = $10,000 - $5,100 = $4,900.
        # Bar 4 (2024-01-05): Close = $112.0.
        # Shares value = 50 * $112.0 = $5,600.
        # Final Equity = $4,900 + $5,600 = $10,500.0.

        assert len(res.fills) == 1
        fill = res.fills[0]
        assert fill.timestamp == dates[1]  # Filled on bar 2's timestamp
        assert fill.fill_price == 102.0  # Filled at bar 2's OPEN price
        assert fill.quantity == 50.0

        final_equity = float(res.equity_curve.iloc[-1]["equity"])
        final_cash = float(res.equity_curve.iloc[-1]["cash"])
        assert final_equity == pytest.approx(10500.0, abs=1e-5)
        assert final_cash == pytest.approx(4900.0, abs=1e-5)

    def test_accounting_buy_and_sell(self) -> None:
        """Accounting: BUY 10 @100 commission 1 -> cash = initial-1001; SELL 10 @110 commission 1 -> cash +1099, realized PnL 98."""
        initial_cash = 10000.0
        portfolio = Portfolio(initial_cash=initial_cash)

        # 1. BUY 10 @ 100, commission 1
        buy_fill = FillEvent(
            timestamp=pd.Timestamp("2024-01-01"),
            symbol="XYZ",
            side="BUY",
            quantity=10.0,
            fill_price=100.0,
            commission=1.0,
            slippage=0.0,
        )
        portfolio.on_fill(buy_fill)

        assert portfolio.cash == pytest.approx(initial_cash - 1001.0, abs=1e-6)
        assert portfolio.positions["XYZ"] == 10.0
        assert portfolio.cost_basis["XYZ"] == 100.0
        assert portfolio.realized_pnl == 0.0

        # 2. SELL 10 @ 110, commission 1
        sell_fill = FillEvent(
            timestamp=pd.Timestamp("2024-01-02"),
            symbol="XYZ",
            side="SELL",
            quantity=10.0,
            fill_price=110.0,
            commission=1.0,
            slippage=0.0,
        )
        portfolio.on_fill(sell_fill)

        # Cash becomes initial - 1001 + 1099
        expected_cash = (initial_cash - 1001.0) + 1099.0
        assert portfolio.cash == pytest.approx(expected_cash, abs=1e-6)
        # Realized PnL = (110 - 100) * 10 - 1 (buy comm) - 1 (sell comm) = 98
        assert portfolio.realized_pnl == pytest.approx(98.0, abs=1e-6)
        assert portfolio.positions["XYZ"] == 0.0

    def test_no_lookahead_in_get_latest_bars(self) -> None:
        """A strategy calls get_latest_bars(symbol, 2) on every bar; no returned timestamp is ever later than the current event."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 130, 30),
                "high": np.linspace(102, 132, 30),
                "low": np.linspace(98, 128, 30),
                "close": np.linspace(101, 131, 30),
                "volume": 1000.0,
            },
            index=dates,
        )

        class LookaheadAuditStrategy(Strategy):
            def __init__(self) -> None:
                self.checks_passed = 0

            def on_bar(self, tick: TickEvent, data_handler: HistoricBarDataHandler) -> list[SignalEvent]:
                bars = data_handler.get_latest_bars(tick.symbol, n=2)
                # No returned timestamp can exceed tick.timestamp
                for bar_time in bars.index:
                    assert bar_time <= tick.timestamp, f"Lookahead detected: {bar_time} > {tick.timestamp}"
                self.checks_passed += 1
                return []

        handler = HistoricBarDataHandler({"AAPL": df})
        strat = LookaheadAuditStrategy()
        portfolio = Portfolio(initial_cash=10000.0)
        exec_handler = SimulatedExecutionHandler(handler)

        engine = BacktestEngine(handler, strat, portfolio, exec_handler)
        engine.run()

        assert strat.checks_passed == 30

    def test_truncation_invariance(self) -> None:
        """Equity over the first 100 bars identical when run on 100 vs 200 bars."""
        np.random.seed(42)
        dates_200 = pd.date_range("2024-01-01", periods=200, freq="D")
        prices = 100.0 + np.cumsum(np.random.randn(200) * 0.5)

        df_200 = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices + 0.2,
                "volume": 10000.0,
            },
            index=dates_200,
        )
        df_100 = df_200.iloc[:100].copy()

        # Run on 100 bars
        h100 = HistoricBarDataHandler({"MOCK": df_100})
        s100 = BuyAndHoldStrategy(symbol="MOCK", target_pct=0.25)
        p100 = Portfolio(initial_cash=50000.0)
        e100 = SimulatedExecutionHandler(h100)
        res100 = BacktestEngine(h100, s100, p100, e100).run()

        # Run on 200 bars
        h200 = HistoricBarDataHandler({"MOCK": df_200})
        s200 = BuyAndHoldStrategy(symbol="MOCK", target_pct=0.25)
        p200 = Portfolio(initial_cash=50000.0)
        e200 = SimulatedExecutionHandler(h200)
        res200 = BacktestEngine(h200, s200, p200, e200).run()

        # Compare first 100 bars
        eq100 = res100.equity_curve["equity"].values
        eq200_sliced = res200.equity_curve.loc[res100.equity_curve.index]["equity"].values

        np.testing.assert_allclose(eq100, eq200_sliced, rtol=1e-9, atol=1e-9)

    def test_fill_timestamp_strictly_greater_than_signal_timestamp(self) -> None:
        """fill.timestamp > signal.timestamp for every order."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(10)],
                "high": [102.0 + i for i in range(10)],
                "low": [98.0 + i for i in range(10)],
                "close": [101.0 + i for i in range(10)],
                "volume": 5000.0,
            },
            index=dates,
        )

        schedule = [
            SignalEvent(timestamp=dates[0], symbol="SYM", direction="LONG", strength=1.0, strategy_id="sched", target_pct=0.2),
            SignalEvent(timestamp=dates[3], symbol="SYM", direction="EXIT", strength=1.0, strategy_id="sched"),
            SignalEvent(timestamp=dates[5], symbol="SYM", direction="SHORT", strength=1.0, strategy_id="sched", target_pct=0.2),
        ]

        handler = HistoricBarDataHandler({"SYM": df})
        strat = FixedScheduleStrategy(schedule)
        portfolio = Portfolio(initial_cash=10000.0)
        exec_handler = SimulatedExecutionHandler(handler)

        engine = BacktestEngine(handler, strat, portfolio, exec_handler)
        res = engine.run()

        assert len(res.fills) >= 2
        for fill in res.fills:
            # Find the order matching this fill
            matching_orders = [o for o in res.orders if o.order_id == fill.order_id]
            assert len(matching_orders) == 1
            order = matching_orders[0]
            assert fill.timestamp > order.timestamp, f"Fill {fill.timestamp} not strictly after order {order.timestamp}"

    def test_multi_asset_missing_days_no_stale_bar(self) -> None:
        """Multi-asset with different missing days: no stale bar used as fresh."""
        dates_a = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        # Asset B is missing Day 3 (2024-01-03)
        dates_b = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])

        df_a = pd.DataFrame(
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 101.0, "volume": 1000.0},
            index=dates_a,
        )
        df_b = pd.DataFrame(
            {"open": 200.0, "high": 205.0, "low": 195.0, "close": 201.0, "volume": 2000.0},
            index=dates_b,
        )

        handler = HistoricBarDataHandler({"A": df_a, "B": df_b})

        # Process through timeline
        ticks_at_day_3: list[TickEvent] = []
        while handler.continue_backtest:
            ticks = handler.update_bars()
            if handler.current_timestamp == pd.Timestamp("2024-01-03"):
                ticks_at_day_3 = ticks

        # Only Asset A should have emitted a tick at Day 3
        assert len(ticks_at_day_3) == 1
        assert ticks_at_day_3[0].symbol == "A"
        assert ticks_at_day_3[0].timestamp == pd.Timestamp("2024-01-03")

    def test_limit_order_touch_and_ttl_expiration(self) -> None:
        """Limit order: BUY limit below open never touched expires after ttl; touched fills at min(open, limit)."""
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        exec_handler = SimulatedExecutionHandler(fill_model=ZeroCostFillModel())

        # 1. Untouched limit order with ttl_bars=2
        untouched_order = OrderEvent(
            timestamp=dates[0],
            symbol="STK",
            side="BUY",
            quantity=100.0,
            order_type="LMT",
            limit_price=80.0,  # Below market
            ttl_bars=2,
        )
        exec_handler.on_order(untouched_order)

        # Bar 1: low is 95.0 > limit 80.0 -> not filled, bars_active = 1
        bar1 = TickEvent(timestamp=dates[1], symbol="STK", open=100.0, high=105.0, low=95.0, close=100.0, volume=1000.0)
        fills1 = exec_handler.on_bar(bar1)
        assert len(fills1) == 0
        assert len(exec_handler.pending_orders) == 1

        # Bar 2: low is 94.0 > limit 80.0 -> not filled, bars_active reaches ttl=2 -> expired
        bar2 = TickEvent(timestamp=dates[2], symbol="STK", open=100.0, high=105.0, low=94.0, close=100.0, volume=1000.0)
        fills2 = exec_handler.on_bar(bar2)
        assert len(fills2) == 0
        assert len(exec_handler.pending_orders) == 0  # Cancelled/expired

        # 2. Touched limit order filling at min(open, limit)
        touched_order = OrderEvent(
            timestamp=dates[2],
            symbol="STK",
            side="BUY",
            quantity=50.0,
            order_type="LMT",
            limit_price=96.0,
            ttl_bars=3,
        )
        exec_handler.on_order(touched_order)

        # Bar 3: open=98.0, low=94.0 <= limit 96.0 -> touched! Fill price = min(open=98, limit=96) = 96.0
        bar3 = TickEvent(timestamp=dates[3], symbol="STK", open=98.0, high=102.0, low=94.0, close=100.0, volume=1000.0)
        fills3 = exec_handler.on_bar(bar3)
        assert len(fills3) == 1
        assert fills3[0].fill_price == 96.0

        # 3. Favorable gap down: open=92.0 <= limit 96.0 -> fill price = min(open=92, limit=96) = 92.0
        gap_order = OrderEvent(
            timestamp=dates[3],
            symbol="STK",
            side="BUY",
            quantity=25.0,
            order_type="LMT",
            limit_price=96.0,
            ttl_bars=2,
        )
        exec_handler.on_order(gap_order)

        bar4 = TickEvent(timestamp=dates[4], symbol="STK", open=92.0, high=95.0, low=90.0, close=94.0, volume=1000.0)
        fills4 = exec_handler.on_bar(bar4)
        assert len(fills4) == 1
        assert fills4[0].fill_price == 92.0

    def test_risk_limit_rejection_and_zero_quantity_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        """Risk-limit rejection logged; OrderEvent quantity 0 raises."""
        # 1. Zero quantity raises ValueError
        with pytest.raises(ValueError, match="quantity must be strictly > 0"):
            OrderEvent(
                timestamp=pd.Timestamp("2024-01-01"),
                symbol="XYZ",
                side="BUY",
                quantity=0.0,
            )

        # 2. Risk-limit rejection logged
        portfolio = Portfolio(
            initial_cash=10000.0,
            risk_limits=RiskLimits(max_position_pct=0.20),
        )
        portfolio.current_prices["XYZ"] = 100.0

        # Signal requests 80% allocation, violating 20% limit
        excessive_signal = SignalEvent(
            timestamp=pd.Timestamp("2024-01-01"),
            symbol="XYZ",
            direction="LONG",
            strength=1.0,
            target_pct=0.80,
        )

        with caplog.at_level(logging.WARNING):
            orders = portfolio.on_signal(excessive_signal)

        assert orders == []
        assert any("Order rejected by risk limits" in record.message for record in caplog.records)

    def test_persist_to_db_orders_equals_fills(self) -> None:
        """persist_to_db: order_executions rows == number of fills."""
        # Set up SQLite in-memory database with migrations
        ConnectionManager.reset_instance()
        cm = ConnectionManager.get_instance(backend="sqlite", sqlite_path=":memory:")
        with cm.connection() as conn:
            apply_migrations(conn)

        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
                "high": [105.0, 106.0, 108.0, 110.0, 112.0, 115.0],
                "low": [98.0, 100.0, 102.0, 104.0, 106.0, 108.0],
                "close": [101.0, 103.0, 105.0, 107.0, 109.0, 111.0],
                "volume": 1000.0,
            },
            index=dates,
        )

        handler = HistoricBarDataHandler({"PERSIST_SYM": df})
        strat = BuyAndHoldStrategy(symbol="PERSIST_SYM", target_pct=0.15)
        portfolio = Portfolio(initial_cash=20000.0)
        exec_handler = SimulatedExecutionHandler(handler)

        engine = BacktestEngine(
            data_handler=handler,
            strategy=strat,
            portfolio=portfolio,
            execution_handler=exec_handler,
            persist_to_db=True,
            connection_manager=cm,
        )

        res = engine.run()

        # Query database via OrderRepo
        order_repo = OrderRepo(connection_manager=cm)
        db_orders = order_repo.list_orders(symbol="PERSIST_SYM")

        assert len(res.fills) > 0
        assert len(db_orders) == len(res.fills)

        ConnectionManager.reset_instance()
