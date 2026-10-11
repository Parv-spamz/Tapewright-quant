"""Metamorphic and differential integration tests for FinPulse-Apex / Tapewright-Quant.

Properties verified:
1. Price scale: multiplying every price by 10 leaves net returns identical to 1e-9 for fixed-fraction strategies with zero cost.
2. Symbol permutation: reordering symbols in the panel leaves portfolio equity identical to 1e-9.
3. Cost monotonicity: for crossover, breakout, mean_reversion on a fixed synthetic panel, final equity is non-increasing
   in the cost multiplier [0, 1, 2, 4] and strictly decreasing when trades occurred.
4. Differential: an independent naive loop inside the test (explicit next-open fills, no engine code) reproduces the
   engine's equity for the buy-when-fast-above-slow strategy on 20 seeds to 1e-9.
5. Causal gate monotonic: lowering q admits a subset of the edges admitted at the higher q (10 seeds).
6. Look-ahead mutant: a strategy using close.shift(-1) is flagged by the zero-lookahead detector from Prompt 18; a strategy
   using close.shift(1) is not.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd
import pytest

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestEngine
from backtesting.engine.event_loop.events import FillEvent, OrderEvent, SignalEvent, TickEvent
from backtesting.engine.event_loop.execution import FillModel, SimulatedExecutionHandler, ZeroCostFillModel
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.engine.event_loop.strategy_base import Strategy
from features.econometric.cointegration.causal_graph import build_causal_graph
from signals.generation.breakout.breakout import BreakoutStrategy
from signals.generation.crossover.crossover import CrossoverStrategy, dual_ema_signals
from signals.generation.mean_reversion.mean_reversion import MeanReversionStrategy
from tests.integration.test_zero_lookahead import assert_zero_lookahead_and_causal_fills


def _make_synthetic_ohlcv(symbol: str, n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible geometric Brownian motion synthetic OHLCV panel."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D", tz="UTC")
    ret = rng.normal(0.0005, 0.015, n_days)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1.0 + rng.uniform(0.001, 0.01, n_days))
    low = close * (1.0 - rng.uniform(0.001, 0.01, n_days))
    open_px = close * (1.0 + rng.normal(0, 0.002, n_days))
    vol = rng.uniform(1000.0, 5000.0, n_days)
    return pd.DataFrame(
        {"open": open_px, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


class ParametricCostFillModel(FillModel):
    """Execution fill model scaling commission and half-spread slippage by a cost multiplier."""

    def __init__(self, mult: float) -> None:
        self.mult = float(mult)

    def fill(self, order: OrderEvent, bar: TickEvent) -> FillEvent | None:
        if order.symbol != bar.symbol:
            return None
        px = float(bar.open)
        slip = 0.0005 * px * self.mult
        if order.side == "BUY":
            fill_px = px + slip
        else:
            fill_px = max(0.01, px - slip)
        comm = (order.quantity * fill_px) * 0.001 * self.mult
        return FillEvent(
            timestamp=bar.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_px,
            commission=comm,
            slippage=slip,
            order_id=order.order_id,
        )


class MultiSymbolCrossoverStrategy(Strategy):
    """Multi-asset trend following strategy evaluating crossover per instrument."""

    def __init__(
        self,
        symbols: list[str],
        fast: int = 5,
        slow: int = 15,
        target_pct: float = 0.10,
    ) -> None:
        self.symbols = symbols
        self.fast = fast
        self.slow = slow
        self.target_pct = float(target_pct)
        self.last_signals = {s: 0 for s in symbols}

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        if tick.symbol not in self.symbols:
            return []
        df = data_handler.get_latest_bars(tick.symbol, n=self.slow + 20)
        if len(df) < self.slow:
            return []
        sig_series = dual_ema_signals(df["close"], fast=self.fast, slow=self.slow)
        curr_sig = int(sig_series.iloc[-1])
        if curr_sig == self.last_signals[tick.symbol]:
            return []
        self.last_signals[tick.symbol] = curr_sig
        direction = "LONG" if curr_sig == 1 else "SHORT" if curr_sig == -1 else "EXIT"
        return [
            SignalEvent(
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                direction=direction,
                target_pct=self.target_pct,
            )
        ]


# ==============================================================================
# 1. Price Scale Metamorphic Invariance
# ==============================================================================


def test_price_scale_invariance() -> None:
    """Multiplying every price by 10 leaves net returns identical to 1e-9 for fixed-fraction strategies with zero cost."""
    df1 = _make_synthetic_ohlcv("SPY", n_days=100, seed=42)
    df10 = df1.copy()
    for col in ["open", "high", "low", "close"]:
        df10[col] = df10[col] * 10.0

    # Base price backtest
    dh1 = HistoricBarDataHandler({"SPY": df1})
    strat1 = CrossoverStrategy("SPY", fast=5, slow=15, target_pct=0.10)
    port1 = Portfolio(100_000.0)
    eng1 = BacktestEngine(dh1, strat1, port1, SimulatedExecutionHandler(dh1, ZeroCostFillModel()))
    res1 = eng1.run()

    # 10x scaled price backtest
    dh10 = HistoricBarDataHandler({"SPY": df10})
    strat10 = CrossoverStrategy("SPY", fast=5, slow=15, target_pct=0.10)
    port10 = Portfolio(100_000.0)
    eng10 = BacktestEngine(dh10, strat10, port10, SimulatedExecutionHandler(dh10, ZeroCostFillModel()))
    res10 = eng10.run()

    eq1 = res1.equity_curve["equity"].values
    eq10 = res10.equity_curve["equity"].values
    net_ret1 = (eq1 - 100_000.0) / 100_000.0
    net_ret10 = (eq10 - 100_000.0) / 100_000.0

    max_diff = float(np.max(np.abs(net_ret1 - net_ret10)))
    assert max_diff < 1e-9, f"Price scale returns diverged: {max_diff}"


# ==============================================================================
# 2. Symbol Permutation Invariance
# ==============================================================================


def test_symbol_permutation_invariance() -> None:
    """Reordering symbols in the panel leaves portfolio equity identical to 1e-9."""
    df_aapl = _make_synthetic_ohlcv("AAPL", n_days=60, seed=11)
    df_msft = _make_synthetic_ohlcv("MSFT", n_days=60, seed=22)

    # Run 1: Order ["AAPL", "MSFT"]
    dh1 = HistoricBarDataHandler({"AAPL": df_aapl, "MSFT": df_msft})
    st1 = MultiSymbolCrossoverStrategy(["AAPL", "MSFT"], fast=5, slow=15, target_pct=0.10)
    p1 = Portfolio(100_000.0)
    e1 = BacktestEngine(dh1, st1, p1, SimulatedExecutionHandler(dh1, ZeroCostFillModel()))
    r1 = e1.run()
    eq1 = r1.equity_curve["equity"].values

    # Run 2: Permuted order ["MSFT", "AAPL"]
    dh2 = HistoricBarDataHandler({"MSFT": df_msft, "AAPL": df_aapl})
    st2 = MultiSymbolCrossoverStrategy(["AAPL", "MSFT"], fast=5, slow=15, target_pct=0.10)
    p2 = Portfolio(100_000.0)
    e2 = BacktestEngine(dh2, st2, p2, SimulatedExecutionHandler(dh2, ZeroCostFillModel()))
    r2 = e2.run()
    eq2 = r2.equity_curve["equity"].values

    max_diff = float(np.max(np.abs(eq1 - eq2)))
    assert max_diff < 1e-9, f"Symbol permutation equity diverged: {max_diff}"


# ==============================================================================
# 3. Cost Monotonicity
# ==============================================================================


def test_cost_monotonicity() -> None:
    """For crossover, breakout, mean_reversion on a fixed synthetic panel, final equity is non-increasing

    in the cost multiplier [0,1,2,4] and strictly decreasing when trades occurred.
    """
    df_panel = _make_synthetic_ohlcv("SPY", n_days=100, seed=77)

    strategies = [
        ("crossover", lambda: CrossoverStrategy("SPY", fast=5, slow=15, target_pct=0.10)),
        ("breakout", lambda: BreakoutStrategy("SPY", n=10, target_pct=0.10)),
        ("mean_reversion", lambda: MeanReversionStrategy("SPY", n=10, k=1.0, rsi_n=5, target_pct=0.10)),
    ]

    for name, strat_factory in strategies:
        equities: list[float] = []
        trades_counts: list[int] = []

        for mult in [0, 1, 2, 4]:
            dh = HistoricBarDataHandler({"SPY": df_panel})
            port = Portfolio(100_000.0)
            strat = strat_factory()
            eng = BacktestEngine(dh, strat, port, SimulatedExecutionHandler(dh, ParametricCostFillModel(mult)))
            res = eng.run()

            final_eq = float(res.equity_curve["equity"].iloc[-1])
            equities.append(final_eq)
            trades_counts.append(len(res.trades))

        # Non-increasing condition
        assert equities[0] >= equities[1] >= equities[2] >= equities[3], (
            f"{name} final equity failed non-increasing: {equities}"
        )

        # Strictly decreasing when trades occurred
        if trades_counts[0] > 0:
            assert equities[0] > equities[1] > equities[2] > equities[3], (
                f"{name} final equity failed strictly decreasing with {trades_counts[0]} trades: {equities}"
            )


# ==============================================================================
# 4. Differential Naive Loop vs Backtest Engine
# ==============================================================================


def test_differential_naive_loop_reproduces_engine() -> None:
    """An independent naive loop inside the test (explicit next-open fills, no engine code)

    reproduces the engine's equity for the buy-when-fast-above-slow strategy on 20 seeds to 1e-9.
    """
    for seed in range(20):
        df_s = _make_synthetic_ohlcv("SPY", n_days=80, seed=seed)

        # 1. BacktestEngine execution
        dh_s = HistoricBarDataHandler({"SPY": df_s})
        strat_s = CrossoverStrategy("SPY", fast=5, slow=15, target_pct=0.10)
        port_s = Portfolio(100_000.0)
        eng_s = BacktestEngine(dh_s, strat_s, port_s, SimulatedExecutionHandler(dh_s, ZeroCostFillModel()))
        res_s = eng_s.run()
        eq_engine = res_s.equity_curve["equity"].values

        # 2. Naive loop (no engine code, explicit next-open fills)
        close_vals = df_s["close"].values
        open_vals = df_s["open"].values

        cash = 100_000.0
        pos = 0.0
        pending_qty = 0.0
        last_sig = 0
        eq_naive: list[float] = []

        for t in range(len(df_s)):
            # Explicit next-open fill of pending order
            if pending_qty != 0.0:
                fill_px = open_vals[t]
                cash -= pending_qty * fill_px
                pos += pending_qty
                pending_qty = 0.0

            # Mark to market at close
            current_equity = cash + pos * close_vals[t]
            eq_naive.append(current_equity)

            # Signal evaluation on exact past bars window
            window_start = max(0, t - 35 + 1)
            sub_close = pd.Series(close_vals[window_start : t + 1])
            if len(sub_close) >= 15:
                sig_series = dual_ema_signals(sub_close, fast=5, slow=15)
                curr_sig = int(sig_series.iloc[-1])
            else:
                curr_sig = 0

            if curr_sig != last_sig:
                last_sig = curr_sig
                if curr_sig == 1:
                    target_val = 0.10 * current_equity
                elif curr_sig == -1:
                    target_val = -0.10 * current_equity
                else:
                    target_val = 0.0

                cur_val = pos * close_vals[t]
                delta_val = target_val - cur_val
                if abs(delta_val) >= 1e-4:
                    d_qty = delta_val / close_vals[t]
                    if abs(d_qty) >= 1e-6:
                        pending_qty = d_qty

        diff = float(np.max(np.abs(eq_engine - np.array(eq_naive))))
        assert diff < 1e-9, f"Seed {seed}: differential loop diverged by {diff}"


# ==============================================================================
# 5. Causal Gate Monotonicity
# ==============================================================================


def test_causal_gate_monotonicity() -> None:
    """Lowering q admits a subset of the edges admitted at the higher q (10 seeds)."""
    for seed in range(10):
        rng = np.random.RandomState(seed)
        df_net = pd.DataFrame(
            {
                "X": np.cumsum(rng.normal(0, 1, 100)),
                "Y": np.cumsum(rng.normal(0, 1, 100)),
                "Z": np.cumsum(rng.normal(0, 1, 100)),
            },
            index=pd.date_range("2020-01-01", periods=100, freq="D"),
        )

        cg_high = build_causal_graph(df_net, maxlag=2, q=0.10, te_perm=20, seed=seed)
        cg_low = build_causal_graph(df_net, maxlag=2, q=0.01, te_perm=20, seed=seed)

        low_adj = cg_low.adjacency.values
        high_adj = cg_high.adjacency.values

        # Admitted edges at low q must be a subset of admitted edges at high q
        # Condition: (low_adj & ~high_adj) must be all False
        invalid_admissions = low_adj & (~high_adj)
        assert not invalid_admissions.any(), (
            f"Seed {seed}: lower q={0.01} admitted edge not admitted at higher q={0.10}"
        )


# ==============================================================================
# 6. Look-Ahead Mutant Detection
# ==============================================================================


def test_lookahead_mutant_detection() -> None:
    """A strategy using close.shift(-1) is flagged by the zero-lookahead detector from Prompt 18;

    a strategy using close.shift(1) is not.
    """
    df_400 = _make_synthetic_ohlcv("AAPL", n_days=400, seed=42)
    df_300 = df_400.iloc[:300].copy()

    class LookaheadCheatingStrategy(Strategy):
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
            # Illegal future peek using full panel shift(-1) forward mean
            full_df = data_handler.bars[self.symbol]
            future_mean = float(full_df["close"].shift(-1).mean())
            if tick.close < future_mean:
                return [SignalEvent(tick.timestamp, self.symbol, "LONG", target_pct=0.15)]
            else:
                return [SignalEvent(tick.timestamp, self.symbol, "SHORT", target_pct=0.15)]

    class CausalHonestStrategy(Strategy):
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
            # Strictly causal past-only window using shift(1)
            past_df = data_handler.get_latest_bars(self.symbol, n=10)
            if len(past_df) < 5:
                return []
            lagged = past_df["close"].shift(1)
            lagged_mean = float(lagged.dropna().mean())
            if tick.close > lagged_mean:
                return [SignalEvent(tick.timestamp, self.symbol, "LONG", target_pct=0.15)]
            else:
                return [SignalEvent(tick.timestamp, self.symbol, "SHORT", target_pct=0.15)]

    # 1. Mutant using close.shift(-1) is FLAGGED by detector
    with pytest.raises(AssertionError, match="Equity curves diverged"):
        assert_zero_lookahead_and_causal_fills(
            lambda sym: LookaheadCheatingStrategy(sym),
            df_300,
            df_400,
            symbol="AAPL",
        )

    # 2. Honest strategy using close.shift(1) is NOT flagged
    assert_zero_lookahead_and_causal_fills(
        lambda sym: CausalHonestStrategy(sym),
        df_300,
        df_400,
        symbol="AAPL",
    )
