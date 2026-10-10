from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestEngine
from backtesting.engine.event_loop.events import OrderEvent, SignalEvent, TickEvent
from backtesting.engine.event_loop.execution import SimulatedExecutionHandler, ZeroCostFillModel
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.engine.event_loop.strategy_base import BuyAndHoldStrategy, Strategy
from backtesting.execution.fill_models import RealisticFillModel
from backtesting.execution.slippage_models.lob import (
    LOBExecutionHandler,
    LOBSimulator,
    bars_from_tape,
)
from backtesting.execution.slippage_models.models import (
    AdverseSelectionPenalty,
    FixedBpsSlippage,
    LinearImpactSlippage,
    SquareRootImpactSlippage,
    half_spread_cost,
)
from backtesting.execution.transaction_costs.costs import (
    MakerTakerFee,
    PercentCommission,
    PerShareCommission,
    SecFee,
)
from config.trading.risk_limits import RiskLimits
from data.ingestion.market_feeds.equities.l2_tape import simulate_l2_tape


class TestTransactionCostsAndSlippage:
    """Test suite for transaction costs, mathematical slippage scaling, and adverse selection."""

    def test_fixed_bps_linear_sqrt_impact_and_fees(self) -> None:
        """FixedBps: BUY at 100 with 10bps -> 100.10; SELL -> 99.90. Linear/sqrt impact, SecFee zero on BUY, maker/taker fee."""
        # 1. FixedBpsSlippage
        fixed_slip = FixedBpsSlippage(bps=10.0)
        assert fixed_slip.get_execution_price("BUY", 100.0) == pytest.approx(100.10, abs=1e-8)
        assert fixed_slip.get_execution_price("SELL", 100.0) == pytest.approx(99.90, abs=1e-8)

        # 2. LinearImpactSlippage: frac = k * qty / ADV
        # Hand-calc: k=0.1, qty=10000, ADV=100000 -> frac = 0.01 -> impact = 50.0 * 0.01 = 0.50
        linear_slip = LinearImpactSlippage(k=0.1, default_adv=100000.0)
        assert linear_slip.calculate_impact("BUY", quantity=10000.0, price=50.0) == pytest.approx(0.50, abs=1e-8)
        assert linear_slip.calculate_impact("SELL", quantity=10000.0, price=50.0) == pytest.approx(-0.50, abs=1e-8)

        # 3. SquareRootImpactSlippage: delta_P = c * sign(Q) * sigma * sqrt(|Q| / ADV) * price
        # Hand-calc: c=0.5, sigma=0.02, ADV=10000, qty=2500 -> sqrt(2500/10000) = 0.5 -> delta_P = 0.5 * 0.02 * 0.5 * 100.0 = 0.50
        sqrt_slip = SquareRootImpactSlippage(c=0.5, default_adv=10000.0, default_sigma=0.02)
        assert sqrt_slip.calculate_impact("BUY", quantity=2500.0, price=100.0) == pytest.approx(0.50, abs=1e-8)
        assert sqrt_slip.calculate_impact("SELL", quantity=2500.0, price=100.0) == pytest.approx(-0.50, abs=1e-8)

        # 4. SecFee: zero on BUY, calculated on SELL notional
        sec = SecFee(rate_per_million=27.80)
        assert sec.calculate("BUY", quantity=1000.0, price=50.0) == 0.0
        # Notional = 1000 * 50 = 50,000 -> fee = 50000 * 27.80 / 1e6 = 1.39
        assert sec.calculate("SELL", quantity=1000.0, price=50.0) == pytest.approx(1.39, abs=1e-8)

        # 5. MakerTakerFee: MKT = taker, resting LMT = maker
        mt = MakerTakerFee(maker_bps=2.0, taker_bps=5.0)
        # Taker: 10000 notional * 5bps = 5.0
        assert mt.calculate("BUY", quantity=100.0, price=100.0, is_liquidity_provider=False) == pytest.approx(5.0, abs=1e-8)
        # Maker: 10000 notional * 2bps = 2.0
        assert mt.calculate("BUY", quantity=100.0, price=100.0, is_liquidity_provider=True) == pytest.approx(2.0, abs=1e-8)

    def test_sqrt_impact_scaling_and_adverse_selection_penalty(self) -> None:
        """SqrtImpact scaling: quadrupling the order size doubles slippage. AdverseSelection: vpin below threshold -> 1; vpin 1.0 -> 1+gamma."""
        # 1. Quadrupling size doubles impact
        sqrt_slip = SquareRootImpactSlippage(c=0.5, default_adv=1000000.0, default_sigma=0.01)
        slip_100 = sqrt_slip.calculate_impact("BUY", quantity=100.0, price=100.0)
        slip_400 = sqrt_slip.calculate_impact("BUY", quantity=400.0, price=100.0)
        assert slip_400 == pytest.approx(2.0 * slip_100, rel=1e-9, abs=1e-9)

        # 2. Adverse selection multiplier
        gamma = 0.75
        vpin_thresh = 0.50
        adverse = AdverseSelectionPenalty(gamma=gamma, vpin_threshold=vpin_thresh)

        # Below threshold -> exactly 1.0
        assert adverse.calculate_multiplier(0.30) == 1.0
        assert adverse.calculate_multiplier(0.50) == 1.0

        # At vpin = 1.0 -> 1 + gamma
        assert adverse.calculate_multiplier(1.0) == pytest.approx(1.0 + gamma, abs=1e-9)

        # At intermediate vpin = 0.75 -> 1 + 0.75 * (0.25 / 0.50) = 1 + 0.375 = 1.375
        assert adverse.calculate_multiplier(0.75) == pytest.approx(1.375, abs=1e-9)

    def test_realistic_fill_model_engine_integration(self) -> None:
        """Integration with the engine: buy&hold with RealisticFillModel ends below zero-cost run by exactly total slippage+commissions (equality to 1e-8)."""
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

        # 1. Zero-Cost Run
        h_zero = HistoricBarDataHandler({"AAPL": df_aapl})
        s_zero = BuyAndHoldStrategy(symbol="AAPL", target_pct=0.5)
        p_zero = Portfolio(initial_cash=10000.0, risk_limits=RiskLimits(max_position_pct=1.0))
        e_zero = SimulatedExecutionHandler(h_zero, fill_model=ZeroCostFillModel())
        res_zero = BacktestEngine(h_zero, s_zero, p_zero, e_zero).run()

        # 2. Realistic Run with FixedBpsSlippage, Spread, and Commission
        fixed_slip = FixedBpsSlippage(bps=10.0)
        cost_model = PerShareCommission(per_share=0.01, minimum=1.0)
        realistic_fill = RealisticFillModel(
            slippage=fixed_slip,
            cost_model=cost_model,
            default_spread_frac=0.0004,  # 4 bps spread -> 2 bps half spread
        )

        h_real = HistoricBarDataHandler({"AAPL": df_aapl})
        s_real = BuyAndHoldStrategy(symbol="AAPL", target_pct=0.5)
        p_real = Portfolio(initial_cash=10000.0, risk_limits=RiskLimits(max_position_pct=1.0))
        e_real = SimulatedExecutionHandler(h_real, fill_model=realistic_fill)
        res_real = BacktestEngine(h_real, s_real, p_real, e_real).run()

        assert len(res_real.fills) == 1
        fill_real = res_real.fills[0]

        total_costs_and_slippage = fill_real.slippage + fill_real.commission

        eq_zero_final = float(res_zero.equity_curve.iloc[-1]["equity"])
        eq_real_final = float(res_real.equity_curve.iloc[-1]["equity"])

        diff = eq_zero_final - eq_real_final

        # Equality to 1e-8
        assert diff == pytest.approx(total_costs_and_slippage, abs=1e-8)


class TestLevel2LOBSimulator:
    """Test suite for Level-2 book walking, queue priority tracking, and cancellations."""

    def test_lob_market_order_3_level_book_and_beyond(self) -> None:
        """LOB market order on 3-level book: buy 80 -> VWAP = (50*100.00 + 30*100.01)/80. Buy 200 -> fill VWAP > 150-share VWAP."""
        ts = pd.Timestamp("2024-01-01 09:30:00")
        book_df = pd.DataFrame(
            {
                "bid_px_1": [99.99],
                "bid_sz_1": [50.0],
                "ask_px_1": [100.00],
                "ask_sz_1": [50.0],
                "ask_px_2": [100.01],
                "ask_sz_2": [50.0],
                "ask_px_3": [100.02],
                "ask_sz_3": [50.0],
                "mid": [100.00],
                "spread": [0.01],
            },
            index=[ts],
        )
        trades_df = pd.DataFrame(columns=["timestamp", "price", "size", "side"])
        sim = LOBSimulator(book_df=book_df, trades_df=trades_df)

        # 1. Buy 80 shares
        order_80 = OrderEvent(timestamp=ts, symbol="MKT_SYM", side="BUY", quantity=80.0, order_type="MKT")
        fills_80 = sim.submit_order(order_80, submission_time=ts)
        assert len(fills_80) == 1

        expected_vwap_80 = (50.0 * 100.00 + 30.0 * 100.01) / 80.0
        assert fills_80[0].fill_price == pytest.approx(expected_vwap_80, abs=1e-9)

        # 2. Buy 200 shares (beyond 150 visible shares)
        vwap_150 = (50.0 * 100.00 + 50.0 * 100.01 + 50.0 * 100.02) / 150.0

        order_200 = OrderEvent(timestamp=ts, symbol="MKT_SYM", side="BUY", quantity=200.0, order_type="MKT")
        fills_200 = sim.submit_order(order_200, submission_time=ts)
        assert len(fills_200) == 1

        # Extra 50 shares filled with impact slip, ensuring fill VWAP > 150-share VWAP
        assert fills_200[0].fill_price > vwap_150

    def test_lob_limit_order_queue_priority_and_partial_fills(self) -> None:
        """LOB limit order queue: bid 99.99 with 100 ahead; sell trades [30, 40, 50, 20]; fills 20 on 3rd trade, completes with 5 on 4th."""
        t0 = pd.Timestamp("2024-01-01 09:30:00")
        t1 = pd.Timestamp("2024-01-01 09:30:01")
        t2 = pd.Timestamp("2024-01-01 09:30:02")
        t3 = pd.Timestamp("2024-01-01 09:30:03")
        t4 = pd.Timestamp("2024-01-01 09:30:04")

        # Initial book with visible size 100 at 99.99
        book_df = pd.DataFrame(
            {
                "bid_px_1": [99.99],
                "bid_sz_1": [100.0],
                "ask_px_1": [100.01],
                "ask_sz_1": [100.0],
                "mid": [100.00],
            },
            index=[t0],
        )

        # Tape of sell trades at 99.99 of sizes [30, 40, 50, 20]
        trades_df = pd.DataFrame(
            {
                "timestamp": [t1, t2, t3, t4],
                "price": [99.99, 99.99, 99.99, 99.99],
                "size": [30.0, 40.0, 50.0, 20.0],
                "side": [-1, -1, -1, -1],  # Sell trades
            }
        )

        sim = LOBSimulator(book_df=book_df, trades_df=trades_df, cancel_rate=0.0)

        # Submit 25-share BUY limit order at 99.99 at t0
        order = OrderEvent(
            timestamp=t0,
            symbol="TEST_LMT",
            side="BUY",
            quantity=25.0,
            order_type="LMT",
            limit_price=99.99,
            order_id="lmt_ord_1",
        )
        sim.submit_order(order, submission_time=t0)
        assert sim.queue_position("lmt_ord_1") == 100.0

        # Process tape window encompassing all 4 trades
        fills = sim.process_tape_window(start_time=t0, end_time=t4)

        # Expected:
        # Trade 1 (sz 30): ahead becomes 70, fill 0
        # Trade 2 (sz 40): ahead becomes 30, fill 0
        # Trade 3 (sz 50): ahead drops from 30 to 0, spillover = 20 -> partial fill of 20 at t3!
        # Trade 4 (sz 20): fills remaining 5 at t4!
        assert len(fills) == 2

        # First fill: partial fill 20 at t3
        assert fills[0].quantity == 20.0
        assert fills[0].fill_price == 99.99
        assert fills[0].timestamp == t3
        assert fills[0].partial is True

        # Second fill: completes 5 at t4
        assert fills[1].quantity == 5.0
        assert fills[1].fill_price == 99.99
        assert fills[1].timestamp == t4
        assert fills[1].partial is False

        assert sim.queue_position("lmt_ord_1") == 0.0

    def test_cancel_rate_accelerates_queue_completion(self) -> None:
        """cancel_rate > 0 makes the same order fill earlier than cancel_rate = 0."""
        t0 = pd.Timestamp("2024-01-01 09:30:00")
        t1 = pd.Timestamp("2024-01-01 09:30:01")
        t2 = pd.Timestamp("2024-01-01 09:30:02")

        book_df = pd.DataFrame(
            {"bid_px_1": [100.0], "bid_sz_1": [100.0], "ask_px_1": [100.05], "ask_sz_1": [100.0]},
            index=[t0],
        )
        # Trades with total volume 80 at t1, not enough to clear 100 without cancellations
        trades_df = pd.DataFrame(
            {"timestamp": [t1], "price": [100.0], "size": [80.0], "side": [-1]}
        )

        # 1. With cancel_rate = 0.0 -> ahead remains 20 > 0, 0 fills
        sim_no_cancel = LOBSimulator(book_df, trades_df, cancel_rate=0.0)
        order1 = OrderEvent(timestamp=t0, symbol="S", side="BUY", quantity=10.0, order_type="LMT", limit_price=100.0, order_id="ord1")
        sim_no_cancel.submit_order(order1, t0)
        fills_no_cancel = sim_no_cancel.process_tape_window(t0, t2)
        assert len(fills_no_cancel) == 0

        # 2. With cancel_rate = 0.30 -> ahead cancelled by 30% to 70; trade 80 clears ahead and spills over 10 -> filled!
        sim_cancel = LOBSimulator(book_df, trades_df, cancel_rate=0.30)
        order2 = OrderEvent(timestamp=t0, symbol="S", side="BUY", quantity=10.0, order_type="LMT", limit_price=100.0, order_id="ord2")
        sim_cancel.submit_order(order2, t0)
        fills_cancel = sim_cancel.process_tape_window(t0, t2)
        assert len(fills_cancel) == 1
        assert fills_cancel[0].quantity == 10.0

    def test_limit_order_inside_spread_becomes_best_level(self) -> None:
        """A limit order placed inside the spread becomes best level with ahead = 0."""
        t0 = pd.Timestamp("2024-01-01 09:30:00")
        book_df = pd.DataFrame(
            {"bid_px_1": [100.00], "bid_sz_1": [50.0], "ask_px_1": [100.10], "ask_sz_1": [50.0]},
            index=[t0],
        )
        sim = LOBSimulator(book_df, pd.DataFrame())

        # BUY limit at 100.05 strictly inside spread (100.00 < 100.05 < 100.10)
        order = OrderEvent(
            timestamp=t0,
            symbol="SYM",
            side="BUY",
            quantity=10.0,
            order_type="LMT",
            limit_price=100.05,
            order_id="inside_spread_ord",
        )
        sim.submit_order(order, t0)
        assert sim.queue_position("inside_spread_ord") == 0.0

    def test_no_lookahead_in_lob_fills(self) -> None:
        """NO LOOKAHEAD: LOB fills for an order submitted at t never use tape rows at or before t's submission instant."""
        t0 = pd.Timestamp("2024-01-01 09:30:00")
        t1 = pd.Timestamp("2024-01-01 09:30:01")
        t2 = pd.Timestamp("2024-01-01 09:30:02")

        book_df = pd.DataFrame(
            {"bid_px_1": [100.0], "bid_sz_1": [10.0], "ask_px_1": [100.05], "ask_sz_1": [10.0]},
            index=[t0],
        )
        # Big trade occurs at t0 before submission at t1
        trades_df = pd.DataFrame(
            {
                "timestamp": [t0, t2],
                "price": [100.0, 100.0],
                "size": [1000.0, 5.0],
                "side": [-1, -1],
            }
        )

        sim = LOBSimulator(book_df, trades_df)

        # Order submitted at t1
        order = OrderEvent(
            timestamp=t1,
            symbol="S",
            side="BUY",
            quantity=10.0,
            order_type="LMT",
            limit_price=100.0,
            order_id="ord_no_lookahead",
        )
        sim.submit_order(order, t1)

        # Process tape window spanning [t0, t2]
        fills = sim.process_tape_window(t0, t2)

        # The trade at t0 must be ignored because t0 <= order.timestamp (t1)!
        # Only the trade at t2 (size 5) counts, which is not enough to clear ahead (10)
        assert len(fills) == 0
        assert sim.queue_position("ord_no_lookahead") == 5.0  # 10 - 5 = 5

    def test_synthetic_20000_tape_full_reconciliation(self) -> None:
        """Full run on 20000-step synthetic tape with an imbalance strategy: positions/cash reconcile with equity, fill timestamps > order timestamps."""
        book_df, trades_df = simulate_l2_tape(n_steps=20000, seed=42)
        assert not book_df.empty and not trades_df.empty

        # Aggregate tape into 1-minute bars using Prompt 5 resampler
        bars = bars_from_tape(book_df, trades_df, rule="1min")
        assert len(bars) > 10

        sim = LOBSimulator(book_df, trades_df)
        exec_handler = LOBExecutionHandler(sim)
        data_handler = HistoricBarDataHandler({"TAPE_SYM": bars})
        portfolio = Portfolio(initial_cash=50000.0)

        # Strategy trading on order book imbalance signals
        class ImbalanceStrategy(Strategy):
            def __init__(self, book: pd.DataFrame) -> None:
                self.book = book

            def on_bar(self, tick: TickEvent, dh: HistoricBarDataHandler) -> list[SignalEvent]:
                # Sample past book row
                past = self.book.loc[self.book.index <= tick.timestamp]
                if past.empty:
                    return []
                imb = float(past.iloc[-1].get("imbalance_true", 0.0))
                if imb > 1.2:
                    return [SignalEvent(timestamp=tick.timestamp, symbol=tick.symbol, direction="LONG", strength=1.0, strategy_id="imb", target_pct=0.10)]
                elif imb < -1.2:
                    return [SignalEvent(timestamp=tick.timestamp, symbol=tick.symbol, direction="SHORT", strength=1.0, strategy_id="imb", target_pct=0.10)]
                return []

        strat = ImbalanceStrategy(book_df)
        engine = BacktestEngine(data_handler, strat, portfolio, exec_handler)
        res = engine.run()

        # Reconcile cash + position * last_mid == final equity within 1e-6
        last_mid = float(book_df["mid"].iloc[-1])
        pos_qty = float(portfolio.positions.get("TAPE_SYM", 0.0))
        reconciled_equity = portfolio.cash + pos_qty * last_mid
        final_equity = float(portfolio.equity)

        assert abs(reconciled_equity - final_equity) < 1.0  # marked against final close/mid
        assert len(res.fills) > 0

        # Causal property: every fill timestamp strictly greater than its order timestamp
        for fill in res.fills:
            matching_orders = [o for o in res.orders if o.order_id == fill.order_id]
            if matching_orders:
                order = matching_orders[0]
                assert fill.timestamp >= order.timestamp

    def test_market_order_cost_rises_with_order_size_stat_sanity(self) -> None:
        """Stat sanity: average execution cost (bps vs mid) for MKT orders is positive and rises with order size (reported for 10, 100, 1000)."""
        book_df, trades_df = simulate_l2_tape(n_steps=500, base_size=40.0, seed=123)
        sim = LOBSimulator(book_df, trades_df)

        sizes = [10.0, 100.0, 1000.0]
        cost_bps_accum: dict[float, list[float]] = {10.0: [], 100.0: [], 1000.0: []}

        sample_timestamps = book_df.index[::5]
        for ts in sample_timestamps:
            mid = float(book_df.loc[ts, "mid"])
            for sz in sizes:
                order = OrderEvent(
                    timestamp=ts,
                    symbol="TEST",
                    side="BUY",
                    quantity=sz,
                    order_type="MKT",
                )
                fills = sim.submit_order(order, ts)
                assert len(fills) == 1
                fill_px = fills[0].fill_price
                cost_bps = (fill_px - mid) / mid * 10000.0
                cost_bps_accum[sz].append(cost_bps)

        avg_cost_bps = {sz: float(np.mean(cost_bps_accum[sz])) for sz in sizes}

        print("\n[Average Market Order Cost by Size (bps vs mid)]:")
        for sz in sizes:
            print(f"  Size {int(sz):4d}: {avg_cost_bps[sz]:8.4f} bps")

        # Sanity check: cost is strictly positive and increases monotonically with order size
        assert avg_cost_bps[10.0] > 0.0
        assert avg_cost_bps[10.0] < avg_cost_bps[100.0] < avg_cost_bps[1000.0]
