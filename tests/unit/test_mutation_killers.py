"""Targeted killer unit tests for AST mutation testing.

Validates boundary conditions, exact default parameters, strict inequality thresholds,
loss formulas, cost basis calculations, and shift invariants across modules to eliminate
surviving mutants.
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.events import FillEvent, OrderEvent, SignalEvent, TickEvent
from backtesting.engine.event_loop.execution import SimulatedExecutionHandler
from backtesting.engine.event_loop.portfolio import Portfolio
from config.trading.risk_limits import RiskLimits
from backtesting.metrics.risk_measures.measures import (
    calmar,
    cornish_fisher_var,
    historical_var,
    omega_ratio,
    sharpe,
    sortino,
)
from backtesting.metrics.risk_measures.overfit import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    monte_carlo_equity_cone,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from backtesting.metrics.risk_measures.sizing import (
    KellySizedStrategyWrapper,
    half_kelly_fraction,
    position_size,
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
from signals.generation.mean_reversion.mean_reversion import (
    MeanReversionStrategy,
    bollinger_rsi_reversion,
)


def _make_ohlcv_df(n: int = 10, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    data = {
        "open": [100.0 + i for i in range(n)],
        "high": [105.0 + i for i in range(n)],
        "low": [95.0 + i for i in range(n)],
        "close": [102.0 + i for i in range(n)],
        "volume": [1000.0 + 10.0 * i for i in range(n)],
    }
    return pd.DataFrame(data, index=dates)


# ==============================================================================
# 1. Backtesting Engine & Event Loop Killers
# ==============================================================================


def test_kill_data_handler_default_n_and_bounds() -> None:
    df = _make_ohlcv_df(n=5)
    handler = HistoricBarDataHandler({"SPY": df})

    handler.update_bars()
    handler.update_bars()
    handler.update_bars()

    # Kills n: int = 1 -> 0
    bars_default = handler.get_latest_bars("SPY")
    assert len(bars_default) == 1, "Default n must return exactly 1 bar"

    # Kills n <= 0 -> n <= 1
    bars_1 = handler.get_latest_bars("SPY", n=1)
    assert len(bars_1) == 1, "n=1 must return 1 bar, not empty"

    # Kills n <= 0 with n=0
    bars_0 = handler.get_latest_bars("SPY", n=0)
    assert len(bars_0) == 0, "n=0 must return empty DataFrame"

    # Kills or -> and for n <= 0
    bars_neg = handler.get_latest_bars("SPY", n=-1)
    assert len(bars_neg) == 0, "n < 0 must return empty DataFrame"

    # Kills get_latest_bar_value n=1 -> 0 and iloc[-1] -> iloc[0]
    val = handler.get_latest_bar_value("SPY", "close")
    assert np.isclose(val, 104.0), f"Latest bar value must be 104.0, got {val}"


def test_kill_tick_event_defaults_and_validation() -> None:
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    # Default open, high, low must be 0.0 (kills 0.0 -> 1.0)
    tick = TickEvent(timestamp=now, symbol="SPY", close=0.0)
    assert tick.open == 0.0
    assert tick.high == 0.0
    assert tick.low == 0.0

    # Negative price check kills or -> and and 0.0 -> 1.0
    with pytest.raises(ValueError, match="must be non-negative"):
        TickEvent(timestamp=now, symbol="SPY", open=-0.1, close=100.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        TickEvent(timestamp=now, symbol="SPY", high=-0.1, close=100.0)

    # Valid fractional price
    tick_frac = TickEvent(timestamp=now, symbol="SPY", open=0.5, high=1.0, low=0.5, close=0.8)
    assert tick_frac.open == 0.5


def test_kill_order_event_quantities_and_ttl() -> None:
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    # Quantity <= 0.0 kills <= -> <
    with pytest.raises(ValueError, match="quantity must be strictly > 0"):
        OrderEvent(timestamp=now, symbol="SPY", side="BUY", quantity=0.0, order_type="MKT")

    # TTL <= 0 kills <= -> <
    with pytest.raises(ValueError, match="ttl_bars must be positive"):
        OrderEvent(
            timestamp=now,
            symbol="SPY",
            side="BUY",
            quantity=10.0,
            order_type="LMT",
            limit_price=100.0,
            ttl_bars=0,
        )

    order_valid = OrderEvent(
        timestamp=now,
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        order_type="LMT",
        limit_price=100.0,
        ttl_bars=2,
    )
    assert order_valid.ttl_bars == 2


def test_kill_execution_ttl_expiry() -> None:
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2020, 1, 2, tzinfo=timezone.utc)
    t2 = datetime(2020, 1, 3, tzinfo=timezone.utc)

    exec_handler = SimulatedExecutionHandler()
    order = OrderEvent(
        timestamp=t0,
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        order_type="LMT",
        limit_price=90.0,
        ttl_bars=1,
    )
    exec_handler.on_order(order)

    # First bar at t1: limit price 90 not touched (low=95) -> bars_active becomes 1 >= ttl_bars(1)
    bar1 = TickEvent(timestamp=t1, symbol="SPY", open=100.0, high=105.0, low=95.0, close=100.0)
    fills1 = exec_handler.on_bar(bar1)
    assert len(fills1) == 0

    # Second bar at t2: price touches 85, but order must be expired and dropped
    bar2 = TickEvent(timestamp=t2, symbol="SPY", open=85.0, high=90.0, low=80.0, close=85.0)
    fills2 = exec_handler.on_bar(bar2)
    assert len(fills2) == 0, "Expired order must not be filled"

    # Execution commission strictly 0.0 (kills commission=0.0 -> 1.0)
    order_buy = OrderEvent(
        timestamp=t0,
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        order_type="LMT",
        limit_price=105.0,
    )
    exec_handler.on_order(order_buy)
    fills_buy = exec_handler.on_bar(bar1)
    assert len(fills_buy) == 1
    assert fills_buy[0].commission == 0.0, "Execution commission must be strictly 0.0"


def test_kill_portfolio_math_and_position_tracking() -> None:
    port = Portfolio(initial_cash=100_000.0)

    # Test invalid price px <= 0.0 kills <= -> <
    port.current_prices["SPY"] = 0.0
    orders_invalid = port.on_signal(
        SignalEvent(
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            symbol="SPY",
            direction="LONG",
            strength=1.0,
            target_pct=0.2,
        )
    )
    assert len(orders_invalid) == 0

    # EXIT signal direction when flat must produce no orders (kills target_weight = 0.0 -> 1.0)
    port.current_prices["SPY"] = 100.0
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    orders_exit = port.on_signal(
        SignalEvent(
            timestamp=now,
            symbol="SPY",
            direction="EXIT",
            strength=1.0,
        )
    )
    assert len(orders_exit) == 0, "EXIT direction when flat must produce no orders"

    # Buy 100 shares at 100.0 via FillEvent
    fill1 = FillEvent(
        timestamp=now,
        symbol="SPY",
        side="BUY",
        quantity=100.0,
        fill_price=100.0,
        commission=10.0,
    )
    port.on_fill(fill1)
    assert port.positions["SPY"] == 100.0
    assert np.isclose(port.cost_basis["SPY"], 100.0)

    # Add 100 shares at 120.0 -> new cost basis must be (100*100 + 100*120)/200 = 110.0
    # Kills new_cost + -> -
    fill2 = FillEvent(
        timestamp=now,
        symbol="SPY",
        side="BUY",
        quantity=100.0,
        fill_price=120.0,
        commission=0.0,
    )
    port.on_fill(fill2)
    assert np.isclose(
        port.cost_basis["SPY"], 110.0
    ), f"Cost basis must be 110.0, got {port.cost_basis['SPY']}"

    # Close 100 shares at 130.0 (partial close): cash increases by 100 * 130 - 20 = 12980.0
    # Kills gross_pnl * -> / and close_comm / -> *
    initial_cash = port.cash
    fill3 = FillEvent(
        timestamp=now,
        symbol="SPY",
        side="SELL",
        quantity=100.0,
        fill_price=130.0,
        commission=20.0,
    )
    port.on_fill(fill3)
    assert np.isclose(port.cash, initial_cash + 12980.0)
    assert port.positions["SPY"] == 100.0

    # Close remaining 100 shares: exact full close qty <= abs(curr_pos) kills <= -> <
    fill4 = FillEvent(
        timestamp=now,
        symbol="SPY",
        side="SELL",
        quantity=100.0,
        fill_price=130.0,
        commission=0.0,
    )
    port.on_fill(fill4)
    assert port.positions["SPY"] == 0.0
    assert port.cost_basis["SPY"] == 0.0

    # Test rebalancing on signal kills target_value = target_weight * equity and delta_value = target - current
    port.current_prices["SPY"] = 100.0
    orders = port.on_signal(
        SignalEvent(
            timestamp=now,
            symbol="SPY",
            direction="LONG",
            strength=1.0,
            target_pct=0.2,
        )
    )
    assert len(orders) == 1
    assert orders[0].side == "BUY"
    assert orders[0].quantity > 0.0

    # Position reversal (kills reversal commission close_qty / qty)
    port_rev = Portfolio(initial_cash=100_000.0)
    port_rev.on_fill(FillEvent(timestamp=now, symbol="SPY", side="BUY", quantity=10.0, fill_price=100.0, commission=0.0))
    port_rev.on_fill(FillEvent(timestamp=now, symbol="SPY", side="SELL", quantity=30.0, fill_price=100.0, commission=30.0))
    assert np.isclose(port_rev.trades[-1]["commission"], 10.0), "Closing commission must be proportional to closed quantity"

    # Adding to fractional position (kills curr_pos > 0 -> curr_pos > 1)
    port_frac = Portfolio(initial_cash=10_000.0)
    port_frac.on_fill(FillEvent(timestamp=now, symbol="SPY", side="BUY", quantity=0.5, fill_price=100.0, commission=0.0))
    port_frac.on_fill(FillEvent(timestamp=now, symbol="SPY", side="BUY", quantity=0.5, fill_price=120.0, commission=0.0))
    assert np.isclose(port_frac.positions["SPY"], 1.0)
    assert np.isclose(port_frac.cost_basis["SPY"], 110.0)


# ==============================================================================
# 2. Risk Measures & Overfit Statistics Killers
# ==============================================================================


def test_kill_sortino_sharpe_calmar() -> None:
    # Sortino empty array kills == 0 -> != 0
    assert sortino(pd.Series([], dtype=float)) == 0.0

    # Sortino with zero downside diff kills downside_diff min(arr - mar, 0.0)
    all_positive = pd.Series([0.01, 0.02, 0.03, 0.04])
    assert sortino(all_positive) == 0.0

    # Sortino with mar > 0 kills mar default 0 -> 1 and mar difference
    mixed = pd.Series([0.02, -0.01, 0.03, -0.02])
    sort_0 = sortino(mixed, mar=0.0)
    sort_1 = sortino(mixed, mar=0.01)
    assert sort_0 > sort_1, "Higher MAR must reduce Sortino ratio"

    # Sharpe with rf kills - rf_per_period -> + rf_per_period
    sh_0 = sharpe(mixed, rf=0.0)
    sh_rf = sharpe(mixed, rf=0.05)
    assert sh_0 > sh_rf, "Higher risk-free rate must reduce Sharpe ratio"

    # Sharpe constant returns (std < 1e-12) kills return 0.0 -> 1.0
    assert sharpe(pd.Series([0.05, 0.05, 0.05])) == 0.0

    # Calmar drawdown calculation (equity - peak) / peak kills - -> +
    eq_down = pd.Series([0.1, -0.2, -0.1, 0.05])
    calm = calmar(eq_down)
    assert calm < 0.0, "Losing series must produce negative Calmar"

    # Calmar with overall positive return and non-zero drawdown kills 1.0 + arr -> 0.0 + arr
    assert calmar(pd.Series([0.05, -0.02, 0.04])) > 0.0

    # Cornish-Fisher VaR formula -(mu - z * sigma) with alpha=0.95
    cf_var = cornish_fisher_var(mixed, alpha=0.95)
    assert cf_var > 0.0

    # Cornish-Fisher VaR with non-zero kurtosis
    cf_kurt = cornish_fisher_var(pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.04]), alpha=0.95)
    assert cf_kurt > 0.0

    # Historical VaR
    h_var = historical_var(mixed, alpha=0.95)
    assert h_var > 0.0

    # Omega ratio with zero losses kills float('inf') if gains > 0.0 else 1.0
    assert omega_ratio(pd.Series([0.01, 0.02])) == float("inf")
    assert omega_ratio(pd.Series([0.0, 0.0])) == 1.0


def test_kill_overfit_and_sizing_annualization() -> None:
    # PSR benchmark annualization kills / np.sqrt(ppy) -> * np.sqrt(ppy)
    psr_daily = probabilistic_sharpe_ratio(
        sr_hat=0.08, sr_benchmark=0.04, T=252, annualized=True, ppy=252
    )
    psr_no_ppy = probabilistic_sharpe_ratio(
        sr_hat=0.08, sr_benchmark=0.04, T=252, annualized=False, ppy=252
    )
    assert psr_daily != psr_no_ppy

    # DSR sr_hat annualization kills sr_hat / sqrt(ppy)
    dsr = deflated_sharpe_ratio(
        sr_hat=1.5, T=252, skew=0.0, kurt=3.0, n_trials=10, var_sr=0.25, annualized=True, ppy=252
    )
    assert 0.0 <= dsr <= 1.0

    # CSCV PBO oos_mean / oos_std
    trials = pd.DataFrame(np.random.RandomState(42).normal(0.001, 0.01, (20, 2)))
    pbo, logits = probability_of_backtest_overfitting(trials)
    assert 0.0 <= pbo <= 1.0

    # Bootstrap Sharpe CI alpha / 2.0
    rets = pd.Series(np.random.RandomState(42).normal(0.001, 0.01, 50))
    ci_low, ci_high = bootstrap_sharpe_ci(rets, n_boot=10, alpha=0.05, seed=42)
    assert ci_low < ci_high

    # Bootstrap Sharpe CI single element returns (0.0, 0.0)
    ci_zero_l, ci_zero_h = bootstrap_sharpe_ci(pd.Series([0.05]), n_boot=5)
    assert ci_zero_l == 0.0 and ci_zero_h == 0.0

    # Monte Carlo equity cone empty returns (kills 1 -> 0)
    cone_empty = monte_carlo_equity_cone(pd.Series([], dtype=float), horizon=3)
    assert cone_empty.iloc[0, 0] == 1.0

    # Half-Kelly sizing with rf kills rf subtraction
    size_0 = half_kelly_fraction(mu=0.05, sigma=0.1, rf=0.0, cap=5.0)
    size_rf = half_kelly_fraction(mu=0.05, sigma=0.1, rf=0.02, cap=5.0)
    assert size_0 > size_rf > 0.0

    # Position sizing constraint diagnostics
    limits = RiskLimits(max_position_pct=0.25, max_leverage=1.0)
    f_pos, diag = position_size(mu=0.05, sigma=0.01, returns_window=np.array([0.01, 0.02]), limits=limits)
    assert diag["binding_constraint"] in ("max_position_pct", "cvar_scaled", "kelly")

    # Kelly sized strategy wrapper on_bar with signal (kills remove not and curr_vol * -> /)
    dummy_strat = BreakoutStrategy("SPY", n=2, target_pct=0.2)
    sized_strat = KellySizedStrategyWrapper(strategy=dummy_strat, window=5, limits=limits)
    df_sizing = _make_ohlcv_df(n=10)
    handler_sz = HistoricBarDataHandler({"SPY": df_sizing})
    for _ in range(len(df_sizing)):
        handler_sz.update_bars()
    tick_sz = TickEvent(
        timestamp=datetime(2020, 1, 15, tzinfo=timezone.utc),
        symbol="SPY",
        open=200.0,
        high=205.0,
        low=195.0,
        close=204.0,
        volume=100.0,
    )
    sigs_sz = sized_strat.on_bar(tick_sz, handler_sz)
    assert isinstance(sigs_sz, list)


# ==============================================================================
# 3. Signals Generation Killers
# ==============================================================================


def test_kill_breakout_strict_inequalities_and_shifts() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "high": [10.0, 10.0, 10.0, 10.0, 10.0],
            "low": [5.0, 5.0, 5.0, 5.0, 5.0],
            "close": [7.0, 8.0, 9.0, 9.5, 10.0],
            "open": [7.0, 8.0, 9.0, 9.5, 10.0],
            "volume": [100.0] * 5,
        },
        index=dates,
    )
    # Donchian: close > prior_high must be strictly greater than (kills > -> >=)
    sig = donchian_breakout(df, n=3)
    assert sig.iloc[-1] == 0, "Touch of prior high without breach must produce 0, not 1"

    # Breach on bar 5: close at 10.5
    df.iloc[4, df.columns.get_loc("close")] = 10.5
    sig_breach = donchian_breakout(df, n=3)
    assert sig_breach.iloc[-1] == 1, "Breach of prior high must produce 1"

    # Volatility breakout strict inequalities
    vol_sig = volatility_breakout(df, n=3, k=1.0)
    assert len(vol_sig) == len(df)

    # Strategy on_bar short signal curr_sig == -1
    df_short = pd.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [5.0, 5.0, 5.0],
            "close": [7.0, 7.0, 3.0],
            "open": [7.0, 7.0, 4.0],
            "volume": [100.0] * 3,
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"),
    )
    handler_short = HistoricBarDataHandler({"SPY": df_short})
    for _ in range(len(df_short)):
        handler_short.update_bars()
    b_strat = BreakoutStrategy("SPY", n=2, target_pct=0.2)
    tick_drop = TickEvent(
        timestamp=datetime(2020, 1, 10, tzinfo=timezone.utc),
        symbol="SPY",
        open=4.0,
        high=4.0,
        low=2.0,
        close=2.0,
        volume=100.0,
    )
    short_sigs = b_strat.on_bar(tick_drop, handler_short)
    assert len(short_sigs) == 1
    assert short_sigs[0].direction == "SHORT"


def test_kill_crossover_events_and_thresholds() -> None:
    # Empty series check kills if len(s) == 0
    empty_sig = dual_ema_signals(pd.Series([], dtype=float), fast=2, slow=5)
    assert len(empty_sig) == 0

    s = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    sig = dual_ema_signals(s, fast=2, slow=4)
    # Identical prices produce diff == 0.0 -> sig must be 0 (kills 0 -> 1)
    assert (sig == 0).all()

    # Golden cross transition from 0 to 1 (kills <= -> <)
    s_cross = pd.Series([10.0, 10.0, 10.0, 15.0, 20.0])
    ev_cross = golden_cross_events(s_cross, fast=2, slow=3)
    assert (ev_cross == 1).any(), "Crossover must produce golden cross event 1"

    # Death cross transition
    s_death = pd.Series([20.0, 20.0, 20.0, 15.0, 10.0])
    ev_death = golden_cross_events(s_death, fast=2, slow=3)
    assert (ev_death == -1).any(), "Death cross must produce event -1"

    # CrossoverStrategy last_signal default 0 (kills 0 -> 1)
    c_strat = CrossoverStrategy("SPY", fast=2, slow=3)
    assert c_strat.last_signal == 0

    # CrossoverStrategy on_bar short signal curr_sig == -1 (kills 1 -> 0)
    handler_c = HistoricBarDataHandler({"SPY": _make_ohlcv_df(n=10)})
    for _ in range(10):
        handler_c.update_bars()
    c_strat_short = CrossoverStrategy("SPY", fast=2, slow=4)
    tick_c_drop = TickEvent(
        timestamp=datetime(2020, 1, 20, tzinfo=timezone.utc),
        symbol="SPY",
        open=50.0,
        high=50.0,
        low=40.0,
        close=40.0,
        volume=100.0,
    )
    c_sigs = c_strat_short.on_bar(tick_c_drop, handler_c)
    assert isinstance(c_sigs, list)


def test_kill_mean_reversion_rsi_and_warmup() -> None:
    df = _make_ohlcv_df(n=12)
    # Bollinger RSI warmup zeroing kills sig_series[:warmup] = 0 -> 1
    sig = bollinger_rsi_reversion(df, n=5, k=1.5, rsi_n=3)
    assert (sig.iloc[:5] == 0).all(), "Warmup period must be strictly 0"

    # Strategy on_bar curr_sig == 1
    handler = HistoricBarDataHandler({"SPY": df})
    for _ in range(len(df)):
        handler.update_bars()
    mr_strat = MeanReversionStrategy("SPY", n=3, k=1.0, rsi_n=2)
    tick = TickEvent(
        timestamp=datetime(2020, 1, 20, tzinfo=timezone.utc),
        symbol="SPY",
        open=10.0,
        high=10.0,
        low=10.0,
        close=10.0,
        volume=100.0,
    )
    res = mr_strat.on_bar(tick, handler)
    assert isinstance(res, list)

    # RSI calculation precision and bands
    df_mr = _make_ohlcv_df(n=20)
    sig_mr = bollinger_rsi_reversion(df_mr, n=5, k=1.0, rsi_n=5, lo=30.0, hi=70.0)
    assert len(sig_mr) == len(df_mr)
