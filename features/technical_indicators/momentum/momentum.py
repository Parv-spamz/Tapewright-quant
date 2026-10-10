"""Momentum technical indicators.

Includes:
- Relative Strength Index (RSI) with Wilder smoothing and initial simple mean
- Moving Average Convergence Divergence (MACD)
- Fast/Slow Stochastic Oscillator (%K, %D)
- Rate of Change (ROC)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder smoothing.

    First average is the simple mean of gains/losses over the first n changes.
    Subsequent values update with alpha = 1 / n.

    Args:
        series: Price series (e.g. close price).
        n: RSI lookback period (default 14).

    Returns:
        pd.Series: RSI values bounded in [0, 100], with NaN for rows 0 to n-1.
    """
    s = pd.Series(series)
    diff = s.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)

    rsi_values = np.full(len(s), np.nan, dtype=float)
    if len(s) <= n:
        return pd.Series(rsi_values, index=s.index, name="rsi")

    # Initial average gain and loss over first n changes (indices 1 to n)
    avg_gain = float(gain.iloc[1 : n + 1].mean())
    avg_loss = float(loss.iloc[1 : n + 1].mean())

    if avg_loss == 0.0:
        rsi_values[n] = 100.0 if avg_gain > 0.0 else 50.0
    elif avg_gain == 0.0:
        rsi_values[n] = 0.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[n] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder recurrence: next_avg = (prev_avg * (n - 1) + current) / n
    for i in range(n + 1, len(s)):
        g = float(gain.iloc[i])
        l = float(loss.iloc[i])
        avg_gain = (avg_gain * (n - 1) + g) / n
        avg_loss = (avg_loss * (n - 1) + l) / n

        if avg_loss == 0.0:
            rsi_values[i] = 100.0 if avg_gain > 0.0 else 50.0
        elif avg_gain == 0.0:
            rsi_values[i] = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi_values, index=s.index, name="rsi").clip(0.0, 100.0)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD).

    Args:
        series: Price series (e.g. close).
        fast: Fast EMA span (default 12).
        slow: Slow EMA span (default 26).
        signal: Signal line EMA span (default 9).

    Returns:
        pd.DataFrame with columns ['macd', 'signal', 'hist'].
    """
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist_line = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "hist": hist_line,
        },
        index=series.index,
    )


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    """Calculate Fast/Slow Stochastic Oscillator (%K, %D).

    Args:
        df: DataFrame containing 'high', 'low', 'close'.
        k: Lookback period for %K (default 14).
        d: Smoothing period for %D (default 3).

    Returns:
        pd.DataFrame with columns ['stoch_k', 'stoch_d'] in [0, 100].
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    low_k = low.rolling(window=k).min()
    high_k = high.rolling(window=k).max()

    spread = (high_k - low_k).replace(0.0, np.nan)
    stoch_k = (100.0 * (close - low_k) / spread).fillna(50.0).clip(0.0, 100.0)
    stoch_d = stoch_k.rolling(window=d).mean().clip(0.0, 100.0)

    return pd.DataFrame(
        {
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
        },
        index=df.index,
    )


def roc(series: pd.Series, n: int = 10) -> pd.Series:
    """Calculate Rate of Change (ROC).

    roc(n) = close / close.shift(n) - 1.

    Args:
        series: Price series.
        n: Shift lag (default 10).

    Returns:
        pd.Series: ROC values with first n rows as NaN.
    """
    return series / series.shift(n) - 1.0
