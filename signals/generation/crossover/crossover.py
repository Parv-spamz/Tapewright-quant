from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from backtesting.engine.event_loop.events import SignalEvent, TickEvent
from backtesting.engine.event_loop.strategy_base import Strategy


def dual_ema_signals(
    close: pd.Series | np.ndarray,
    fast: int = 12,
    slow: int = 26,
) -> pd.Series:
    """Generate dual Exponential Moving Average (EMA) trend signals in {-1, 0, 1}.

    Args:
        close: Price series of close prices.
        fast: Lookback span for the fast EMA.
        slow: Lookback span for the slow EMA (must be > fast).

    Returns:
        pd.Series: Signal series with values in {-1, 0, 1}.
            The first `slow - 1` bars strictly contain 0 (no signal).
    """
    if fast >= slow:
        raise ValueError(f"fast window ({fast}) must be strictly less than slow window ({slow})")

    s = pd.Series(close, dtype=float).copy()
    if len(s) == 0:
        return pd.Series([], dtype=int)

    # Compute exponential moving averages using pandas ewm (adjust=False matches standard financial EMA)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()

    # Raw state: +1 if fast > slow, -1 if fast < slow, 0 if equal
    diff = ema_fast - ema_slow
    sig = np.where(diff > 1e-12, 1, np.where(diff < -1e-12, -1, 0))

    # Mask first slow - 1 bars to enforce warmup constraint
    if len(sig) > 0:
        warmup = min(len(sig), slow - 1)
        sig[:warmup] = 0

    return pd.Series(sig, index=s.index, dtype=int)


def golden_cross_events(
    close: pd.Series | np.ndarray,
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """Identify discrete Golden Cross (+1) and Death Cross (-1) transition events.

    A Golden Cross event occurs when the fast EMA crosses above the slow EMA.
    A Death Cross event occurs when the fast EMA crosses below the slow EMA.

    Args:
        close: Price series of close prices.
        fast: Lookback span for the fast EMA (default 50).
        slow: Lookback span for the slow EMA (default 200).

    Returns:
        pd.Series: Series with +1 on golden cross, -1 on death cross, and 0 otherwise.
    """
    signals = dual_ema_signals(close, fast=fast, slow=slow)
    prev = signals.shift(1).fillna(0).astype(int)

    events = pd.Series(0, index=signals.index, dtype=int)
    # Transition to bullish state from non-bullish
    events[(prev <= 0) & (signals == 1)] = 1
    # Transition to bearish state from non-bearish
    events[(prev >= 0) & (signals == -1)] = -1

    return events


class CrossoverStrategy(Strategy):
    """Event-driven trend-following strategy using dual EMA crossover."""

    def __init__(
        self,
        symbol: str,
        fast: int = 12,
        slow: int = 26,
        target_pct: float = 0.20,
    ) -> None:
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.target_pct = float(target_pct)
        self.last_signal: int = 0

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        if tick.symbol != self.symbol:
            return []

        # Strictly no-lookahead: fetch past bars up to the current bar
        needed_bars = self.slow + 20
        df = data_handler.get_latest_bars(self.symbol, n=needed_bars)
        if len(df) < self.slow:
            return []

        sig_series = dual_ema_signals(df["close"], fast=self.fast, slow=self.slow)
        curr_sig = int(sig_series.iloc[-1])

        if curr_sig == self.last_signal:
            return []

        self.last_signal = curr_sig
        if curr_sig == 1:
            direction = "LONG"
            target_pct = self.target_pct
        elif curr_sig == -1:
            direction = "SHORT"
            target_pct = self.target_pct
        else:
            direction = "EXIT"
            target_pct = 0.0

        return [
            SignalEvent(
                timestamp=tick.timestamp,
                symbol=self.symbol,
                direction=direction,
                strength=1.0,
                strategy_id="crossover",
                target_pct=target_pct,
            )
        ]
