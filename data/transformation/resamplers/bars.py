from typing import Any
import numpy as np
import pandas as pd


def resample_time_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame into larger time bars based on pandas offset rule.

    Aggregations:
    - open: first
    - high: max
    - low: min
    - close: last
    - volume: sum
    """
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pd.DatetimeIndex")

    resampled = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    # Drop empty buckets where no trading occurred
    return resampled.dropna(subset=["close"])


def ticks_to_bars(ticks_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Convert tick trade stream into OHLCV time bars."""
    if ticks_df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = ticks_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("ticks_df must have a 'timestamp' column or a DatetimeIndex.")

    size_col = "size" if "size" in df.columns else "volume"

    resampled = df.resample(rule).agg(
        {
            "price": ["first", "max", "min", "last"],
            size_col: "sum",
        }
    )
    resampled.columns = ["open", "high", "low", "close", "volume"]
    return resampled.dropna(subset=["close"])


def volume_bars(
    ticks_df: pd.DataFrame,
    volume_threshold: float,
    return_remainder: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, float]:
    """Sample volume bars where each closed bar contains cumulative volume >= threshold.

    Guarantees no tick dropped or double counted:
    sum(closed_bars['volume']) + remainder == sum(ticks_df['size'])
    """
    if volume_threshold <= 0:
        raise ValueError(f"volume_threshold must be strictly positive, got {volume_threshold}")

    if ticks_df.empty:
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty_df.attrs["remainder"] = 0.0
        return (empty_df, 0.0) if return_remainder else empty_df

    df = ticks_df.copy()
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"]).values
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.values
    else:
        timestamps = np.arange(len(df))

    prices = df["price"].values
    size_col = "size" if "size" in df.columns else "volume"
    sizes = df[size_col].values

    bar_ts: list[Any] = []
    bar_open: list[float] = []
    bar_high: list[float] = []
    bar_low: list[float] = []
    bar_close: list[float] = []
    bar_vol: list[float] = []

    cur_vol = 0.0
    cur_open = 0.0
    cur_high = -np.inf
    cur_low = np.inf
    cur_close = 0.0
    cur_ts = timestamps[0]

    for t in range(len(prices)):
        p = float(prices[t])
        s = float(sizes[t])
        ts = timestamps[t]

        if cur_vol == 0.0:
            cur_open = p
            cur_high = p
            cur_low = p
            cur_ts = ts

        cur_high = max(cur_high, p)
        cur_low = min(cur_low, p)
        cur_close = p
        cur_vol += s

        if cur_vol >= volume_threshold:
            bar_ts.append(cur_ts)
            bar_open.append(cur_open)
            bar_high.append(cur_high)
            bar_low.append(cur_low)
            bar_close.append(cur_close)
            bar_vol.append(cur_vol)

            # Reset accumulator
            cur_vol = 0.0
            cur_high = -np.inf
            cur_low = np.inf

    remainder = cur_vol
    bars_df = pd.DataFrame(
        {
            "open": bar_open,
            "high": bar_high,
            "low": bar_low,
            "close": bar_close,
            "volume": bar_vol,
        },
        index=pd.DatetimeIndex(bar_ts) if len(bar_ts) > 0 and isinstance(bar_ts[0], (np.datetime64, pd.Timestamp)) else bar_ts,
    )
    bars_df.attrs["remainder"] = remainder

    if return_remainder:
        return bars_df, remainder
    return bars_df


def dollar_bars(
    ticks_df: pd.DataFrame,
    dollar_threshold: float,
    return_remainder: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, float]:
    """Sample dollar bars where each closed bar contains cumulative dollar volume (price * size) >= threshold.

    Guarantees no tick dropped or double counted:
    sum(closed_bars['dollar_volume']) + remainder == sum(ticks_df['price'] * ticks_df['size'])
    """
    if dollar_threshold <= 0:
        raise ValueError(f"dollar_threshold must be strictly positive, got {dollar_threshold}")

    if ticks_df.empty:
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "dollar_volume"])
        empty_df.attrs["remainder"] = 0.0
        return (empty_df, 0.0) if return_remainder else empty_df

    df = ticks_df.copy()
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"]).values
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.values
    else:
        timestamps = np.arange(len(df))

    prices = df["price"].values
    size_col = "size" if "size" in df.columns else "volume"
    sizes = df[size_col].values
    dollars = prices * sizes

    bar_ts: list[Any] = []
    bar_open: list[float] = []
    bar_high: list[float] = []
    bar_low: list[float] = []
    bar_close: list[float] = []
    bar_vol: list[float] = []
    bar_dollar: list[float] = []

    cur_dollars = 0.0
    cur_vol = 0.0
    cur_open = 0.0
    cur_high = -np.inf
    cur_low = np.inf
    cur_close = 0.0
    cur_ts = timestamps[0]

    for t in range(len(prices)):
        p = float(prices[t])
        s = float(sizes[t])
        d = float(dollars[t])
        ts = timestamps[t]

        if cur_dollars == 0.0:
            cur_open = p
            cur_high = p
            cur_low = p
            cur_ts = ts

        cur_high = max(cur_high, p)
        cur_low = min(cur_low, p)
        cur_close = p
        cur_vol += s
        cur_dollars += d

        if cur_dollars >= dollar_threshold:
            bar_ts.append(cur_ts)
            bar_open.append(cur_open)
            bar_high.append(cur_high)
            bar_low.append(cur_low)
            bar_close.append(cur_close)
            bar_vol.append(cur_vol)
            bar_dollar.append(cur_dollars)

            # Reset accumulator
            cur_dollars = 0.0
            cur_vol = 0.0
            cur_high = -np.inf
            cur_low = np.inf

    remainder = cur_dollars
    bars_df = pd.DataFrame(
        {
            "open": bar_open,
            "high": bar_high,
            "low": bar_low,
            "close": bar_close,
            "volume": bar_vol,
            "dollar_volume": bar_dollar,
        },
        index=pd.DatetimeIndex(bar_ts) if len(bar_ts) > 0 and isinstance(bar_ts[0], (np.datetime64, pd.Timestamp)) else bar_ts,
    )
    bars_df.attrs["remainder"] = remainder

    if return_remainder:
        return bars_df, remainder
    return bars_df


def make_synthetic_ticks(n: int, seed: int | None = None) -> pd.DataFrame:
    """Helper to generate realistic trade tick sequences."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02 09:30:00", periods=n, freq="500ms")
    dp = rng.normal(loc=0.0001, scale=0.02, size=n)
    prices = np.round(100.0 + np.cumsum(dp), 2)
    sizes = rng.integers(10, 200, size=n)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "price": prices,
            "size": sizes,
        }
    )
