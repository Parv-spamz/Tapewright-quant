from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from backtesting.engine.event_loop.events import SignalEvent, TickEvent
from backtesting.engine.event_loop.strategy_base import Strategy


def bollinger_rsi_reversion(
    df: pd.DataFrame,
    n: int = 20,
    k: float = 2.0,
    rsi_n: int = 14,
    lo: float = 30.0,
    hi: float = 70.0,
) -> pd.Series:
    """Generate mean-reversion signals combining Bollinger Bands with Relative Strength Index (RSI).

    Entry Conditions:
        +1 (Long): close < Lower Bollinger Band AND RSI < lo (Oversold condition)
        -1 (Short): close > Upper Bollinger Band AND RSI > hi (Overbought condition)
         0: Neutral / outside extreme bands.

    Warmup:
        The first `max(n, rsi_n)` bars strictly evaluate to 0.

    Args:
        df: DataFrame containing at least a 'close' column.
        n: Window length for Bollinger Bands (default 20).
        k: Bollinger Band standard deviation multiplier (default 2.0).
        rsi_n: Lookback period for RSI calculation (default 14).
        lo: RSI oversold threshold (default 30.0).
        hi: RSI overbought threshold (default 70.0).

    Returns:
        pd.Series: Integer signal series in {-1, 0, 1}.
    """
    if len(df) == 0:
        return pd.Series([], dtype=int)

    close = df["close"]

    # 1. Bollinger Bands
    sma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=1)
    upper_band = sma + k * std
    lower_band = sma - k * std

    # 2. RSI Calculation (Wilder smoothing via ewm with alpha=1/rsi_n)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / rsi_n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_n, adjust=False).mean()

    # Guard against zero division
    rs = np.where(avg_loss > 1e-12, avg_gain / avg_loss, np.where(avg_gain > 1e-12, 100.0, 1.0))
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 3. Mean Reversion Signal Logic
    is_oversold = (close < lower_band) & (rsi < lo)
    is_overbought = (close > upper_band) & (rsi > hi)

    sig = np.where(is_oversold, 1, np.where(is_overbought, -1, 0))

    sig_series = pd.Series(sig, index=df.index, dtype=int)
    warmup = max(n, rsi_n)
    sig_series.iloc[: min(len(sig_series), warmup)] = 0
    return sig_series


class MeanReversionStrategy(Strategy):
    """Event-driven Bollinger Band + RSI mean-reversion strategy."""

    def __init__(
        self,
        symbol: str,
        n: int = 20,
        k: float = 2.0,
        rsi_n: int = 14,
        lo: float = 30.0,
        hi: float = 70.0,
        target_pct: float = 0.20,
    ) -> None:
        self.symbol = symbol
        self.n = n
        self.k = k
        self.rsi_n = rsi_n
        self.lo = lo
        self.hi = hi
        self.target_pct = float(target_pct)
        self.last_signal: int = 0

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        if tick.symbol != self.symbol:
            return []

        needed_bars = max(self.n, self.rsi_n) + 20
        df = data_handler.get_latest_bars(self.symbol, n=needed_bars)
        if len(df) <= max(self.n, self.rsi_n):
            return []

        sig_series = bollinger_rsi_reversion(
            df,
            n=self.n,
            k=self.k,
            rsi_n=self.rsi_n,
            lo=self.lo,
            hi=self.hi,
        )
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
                strategy_id="mean_reversion",
                target_pct=target_pct,
            )
        ]
