"""Volume-based technical indicators.

Includes:
- On Balance Volume (OBV)
- Volume Weighted Average Price (VWAP) with intraday session reset
- Chaikin Money Flow (CMF)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On Balance Volume (OBV).

    Args:
        close: Close price series.
        volume: Volume series.

    Returns:
        pd.Series: Cumulative OBV values.
    """
    diff = close.diff()
    direction = pd.Series(
        np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0)),
        index=close.index,
    )
    # The first row has diff=NaN, direction=0
    return (direction * volume).cumsum().rename("obv")


def vwap(df: pd.DataFrame, session: str = "D") -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP).

    Resetting per calendar session on intraday DatetimeIndex,
    or cumulative if daily/unsupported index.

    Args:
        df: DataFrame containing 'high', 'low', 'close', 'volume'.
        session: Session resampling frequency (default 'D' for calendar day).

    Returns:
        pd.Series: VWAP values.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    pv = typical_price * vol

    if isinstance(df.index, pd.DatetimeIndex):
        session_groups = df.index.floor(session)
        cum_pv = pv.groupby(session_groups).cumsum()
        cum_vol = vol.groupby(session_groups).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_vol = vol.cumsum()

    safe_vol = cum_vol.replace(0.0, np.nan)
    vwap_series = (cum_pv / safe_vol).fillna(typical_price)
    return vwap_series.rename("vwap")


def chaikin_money_flow(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Calculate Chaikin Money Flow (CMF).

    MFM = ((Close - Low) - (High - Close)) / (High - Low). If High == Low, MFM = 0.
    MFV = MFM * Volume.
    CMF = sum(MFV, n) / sum(Volume, n).

    Args:
        df: DataFrame containing 'high', 'low', 'close', 'volume'.
        n: Rolling window period (default 20).

    Returns:
        pd.Series: CMF values bounded in [-1, 1].
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    vol = df["volume"]

    hl_spread = high - low
    mfm = np.where(hl_spread == 0.0, 0.0, ((close - low) - (high - close)) / hl_spread)

    mfv = pd.Series(mfm, index=df.index) * vol
    rolling_mfv = mfv.rolling(window=n).sum()
    rolling_vol = vol.rolling(window=n).sum().replace(0.0, np.nan)

    cmf = (rolling_mfv / rolling_vol).fillna(0.0).clip(-1.0, 1.0)
    return cmf.rename("cmf")
