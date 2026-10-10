from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from backtesting.engine.event_loop.events import SignalEvent, TickEvent
from backtesting.engine.event_loop.strategy_base import Strategy


def donchian_breakout(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Generate Donchian Channel breakout signals comparing to the PRIOR n-bar high/low.

    Strictly excludes the current bar by shifting prior extremes by 1 period:
        prior_high = df['high'].shift(1).rolling(n).max()
        prior_low  = df['low'].shift(1).rolling(n).min()

    Signal:
        +1 if close > prior_high (bullish breakout above prior n-bar channel)
        -1 if close < prior_low  (bearish breakdown below prior n-bar channel)
         0 otherwise.

    First `n` bars evaluate to 0.

    Args:
        df: DataFrame containing at least 'high', 'low', 'close' columns.
        n: Rolling lookback channel window in bars.

    Returns:
        pd.Series: Integer signal series in {-1, 0, 1}.
    """
    if len(df) == 0:
        return pd.Series([], dtype=int)

    # Shift by 1 strictly excludes current bar
    prior_high = df["high"].shift(1).rolling(n).max()
    prior_low = df["low"].shift(1).rolling(n).min()

    close = df["close"]
    sig = np.where(close > prior_high, 1, np.where(close < prior_low, -1, 0))

    # Mask initial warmup
    sig_series = pd.Series(sig, index=df.index, dtype=int)
    sig_series.iloc[: min(len(sig_series), n)] = 0
    return sig_series


def volatility_breakout(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    """Generate volatility band breakout signals comparing to PRIOR n-bar mean and dispersion.

    Strictly shifts prior window by 1:
        prior_mean = df['close'].shift(1).rolling(n).mean()
        prior_std  = df['close'].shift(1).rolling(n).std(ddof=1)
        upper = prior_mean + k * prior_std
        lower = prior_mean - k * prior_std

    Signal:
        +1 if close > upper
        -1 if close < lower
         0 otherwise.

    Args:
        df: DataFrame containing at least 'close' column.
        n: Window length for volatility estimation.
        k: Volatility band multiplier.

    Returns:
        pd.Series: Integer signal series in {-1, 0, 1}.
    """
    if len(df) == 0:
        return pd.Series([], dtype=int)

    prior_mean = df["close"].shift(1).rolling(n).mean()
    prior_std = df["close"].shift(1).rolling(n).std(ddof=1)

    upper = prior_mean + k * prior_std
    lower = prior_mean - k * prior_std

    close = df["close"]
    sig = np.where(close > upper, 1, np.where(close < lower, -1, 0))

    sig_series = pd.Series(sig, index=df.index, dtype=int)
    sig_series.iloc[: min(len(sig_series), n)] = 0
    return sig_series


class BreakoutStrategy(Strategy):
    """Event-driven Donchian Channel breakout strategy."""

    def __init__(
        self,
        symbol: str,
        n: int = 20,
        target_pct: float = 0.20,
    ) -> None:
        self.symbol = symbol
        self.n = n
        self.target_pct = float(target_pct)
        self.last_signal: int = 0

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        if tick.symbol != self.symbol:
            return []

        # Strictly no-lookahead: fetch past bars up to the current bar
        needed_bars = self.n + 15
        df = data_handler.get_latest_bars(self.symbol, n=needed_bars)
        if len(df) <= self.n:
            return []

        sig_series = donchian_breakout(df, n=self.n)
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
                strategy_id="breakout",
                target_pct=target_pct,
            )
        ]
