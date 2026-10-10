"""Volatility technical indicators.

Includes:
- True Range (TR)
- Average True Range (ATR) with Wilder smoothing
- Bollinger Bands (mid, upper, lower, %B, Bandwidth) with ddof=0
- Keltner Channels (mid, upper, lower)
- Historical Volatility (annualized log returns standard deviation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range (TR).

    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|).
    For the first row, TR = High - Low.

    Args:
        df: DataFrame containing 'high', 'low', 'close'.

    Returns:
        pd.Series: True Range values.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rename("true_range")


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) using Wilder smoothing.

    Args:
        df: DataFrame containing 'high', 'low', 'close'.
        n: Smoothing period (default 14).

    Returns:
        pd.Series: ATR values (non-negative).
    """
    tr = true_range(df)
    # Wilder smoothing corresponds to ewm(alpha = 1 / n, adjust=False)
    atr_series = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    return atr_series.clip(lower=0.0).rename("atr")


def bollinger(
    series: pd.Series,
    n: int = 20,
    k: float = 2.0,
) -> pd.DataFrame:
    """Calculate Bollinger Bands with ddof=0.

    Args:
        series: Price series (e.g. close).
        n: Window length (default 20).
        k: Standard deviation multiplier (default 2.0).

    Returns:
        pd.DataFrame with columns ['mid', 'upper', 'lower', 'pctb', 'bandwidth'].
    """
    mid = series.rolling(window=n).mean()
    std = series.rolling(window=n).std(ddof=0)

    upper = mid + k * std
    lower = mid - k * std

    band_width = upper - lower
    safe_band_width = band_width.replace(0.0, np.nan)
    safe_mid = mid.replace(0.0, np.nan)

    pctb = ((series - lower) / safe_band_width).fillna(0.5)
    bandwidth = (band_width / safe_mid).fillna(0.0)

    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "pctb": pctb,
            "bandwidth": bandwidth,
        },
        index=series.index,
    )


def keltner(
    df: pd.DataFrame,
    n: int = 20,
    mult: float = 2.0,
) -> pd.DataFrame:
    """Calculate Keltner Channels.

    mid = EMA(close, n)
    upper = mid + mult * ATR(n)
    lower = mid - mult * ATR(n)

    Args:
        df: DataFrame with 'high', 'low', 'close'.
        n: Window length (default 20).
        mult: ATR multiplier (default 2.0).

    Returns:
        pd.DataFrame with columns ['mid', 'upper', 'lower'].
    """
    close = df["close"]
    mid = close.ewm(span=n, adjust=False).mean()
    atr_val = atr(df, n=n)

    upper = mid + mult * atr_val
    lower = mid - mult * atr_val

    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
        },
        index=df.index,
    )


def historical_vol(series: pd.Series, n: int = 21) -> pd.Series:
    """Calculate Annualized Historical Volatility.

    rolling std(ddof=1) of log returns * sqrt(252).

    Args:
        series: Price series (e.g. close).
        n: Rolling window period (default 21).

    Returns:
        pd.Series: Annualized historical volatility.
    """
    log_ret = np.log(series / series.shift(1))
    hv = log_ret.rolling(window=n).std(ddof=1) * np.sqrt(252.0)
    return hv.rename("historical_vol")
