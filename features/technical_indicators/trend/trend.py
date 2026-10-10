"""Trend-following technical indicators.

Includes:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Supertrend with final-band carry-forward
- Average Directional Index (ADX) with Wilder smoothing
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int = 20) -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    Args:
        series: Price series.
        n: Rolling window length.

    Returns:
        pd.Series: SMA values with NaN in warm-up (first n-1 rows).
    """
    return series.rolling(window=n).mean()


def ema(series: pd.Series, n: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Uses alpha = 2 / (n + 1) with adjust=False.

    Args:
        series: Price series.
        n: Span of the EMA.

    Returns:
        pd.Series: EMA values matching pandas ewm(span=n, adjust=False).
    """
    return series.ewm(span=n, adjust=False).mean()


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Calculate Supertrend indicator with final-band carry-forward.

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns.
        n: ATR lookback period.
        mult: ATR multiplier for upper and lower bands.

    Returns:
        pd.DataFrame with columns ['supertrend', 'direction'].
        direction is +1 for uptrend, -1 for downtrend.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["supertrend", "direction"], index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing ATR
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr

    n_rows = len(df)
    final_upper = np.zeros(n_rows, dtype=float)
    final_lower = np.zeros(n_rows, dtype=float)
    supertrend_vals = np.zeros(n_rows, dtype=float)
    direction = np.zeros(n_rows, dtype=int)

    c = close.values
    bu = basic_upper.values
    bl = basic_lower.values

    # Initial condition at index 0
    direction[0] = 1
    final_upper[0] = bu[0]
    final_lower[0] = bl[0]
    supertrend_vals[0] = bl[0]

    for i in range(1, n_rows):
        # Final Upper Band carry-forward
        if bu[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = bu[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final Lower Band carry-forward
        if bl[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = bl[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Trend direction and supertrend line
        if direction[i - 1] == 1:
            if c[i] < final_lower[i]:
                direction[i] = -1
                supertrend_vals[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend_vals[i] = final_lower[i]
        else:
            if c[i] > final_upper[i]:
                direction[i] = 1
                supertrend_vals[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend_vals[i] = final_upper[i]

    return pd.DataFrame(
        {
            "supertrend": supertrend_vals,
            "direction": direction,
        },
        index=df.index,
    )


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Calculate Average Directional Index (ADX) with Wilder smoothing.

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns.
        n: Smoothing period (default 14).

    Returns:
        pd.DataFrame with columns ['adx', 'plus_di', 'minus_di'].
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["adx", "plus_di", "minus_di"], index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing with alpha = 1 / n
    tr_smooth = pd.Series(tr, index=df.index).ewm(alpha=1.0 / n, adjust=False).mean()
    pdm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False).mean()
    mdm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False).mean()

    # Directional indicators (+DI, -DI) in [0, 100]
    safe_tr = tr_smooth.replace(0.0, np.nan)
    plus_di = (100.0 * pdm_smooth / safe_tr).fillna(0.0).clip(0.0, 100.0)
    minus_di = (100.0 * mdm_smooth / safe_tr).fillna(0.0).clip(0.0, 100.0)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    di_diff = (plus_di - minus_di).abs()
    dx = (100.0 * di_diff / di_sum).fillna(0.0).clip(0.0, 100.0)

    adx_series = dx.ewm(alpha=1.0 / n, adjust=False).mean().clip(0.0, 100.0)

    return pd.DataFrame(
        {
            "adx": adx_series,
            "plus_di": plus_di,
            "minus_di": minus_di,
        },
        index=df.index,
    )
