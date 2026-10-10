import numpy as np
import pandas as pd


def mad_filter(
    series: pd.Series,
    window: int = 20,
    threshold: float = 5.0,
) -> pd.Series:
    """Filter outliers using rolling median and Median Absolute Deviation (MAD).

    Flags points where: |x_t - median_{past}| / (1.4826 * MAD_{past}) > threshold
    Uses PAST data only (center=False, current point excluded from its own window).
    """
    if len(series) < window + 1:
        return pd.Series(False, index=series.index)

    # Shift by 1 so the current point is excluded from its own lookback window
    past = series.shift(1)
    roll_med = past.rolling(window=window, min_periods=window).median()

    def _calc_mad(w: np.ndarray) -> float:
        m = np.median(w)
        return float(np.median(np.abs(w - m)))

    roll_mad = past.rolling(window=window, min_periods=window).apply(_calc_mad, raw=True)
    scale = 1.4826 * roll_mad

    # Prevent division by zero
    scale = scale.replace(0.0, np.nan)
    denom = scale.fillna(1e-8)
    denom = np.where(denom < 1e-8, 1e-8, denom)

    diff = np.abs(series - roll_med)
    z_score = diff / denom

    flags = z_score > threshold
    return pd.Series(flags.fillna(False), index=series.index, dtype=bool)


def remove_bad_ticks(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Identify and scrub bad price ticks from OHLCV data.

    Flagged closes are replaced by the previous valid close (forward fill).
    Returns:
        tuple[pd.DataFrame, pd.Series]: (clean_df, mask_of_flagged_ticks)
    """
    clean_df = df.copy()
    close_col = "close" if "close" in clean_df.columns else "Close"

    mask = mad_filter(clean_df[close_col], window=window, threshold=threshold)

    if mask.any():
        clean_df.loc[mask, close_col] = np.nan
        clean_df[close_col] = clean_df[close_col].ffill()

        # If the first element is flagged and ffill leaves NaN, bfill
        if clean_df[close_col].isna().any():
            clean_df[close_col] = clean_df[close_col].bfill()

        # Re-synchronize high and low with cleaned close if needed
        if "high" in clean_df.columns and "low" in clean_df.columns:
            clean_df.loc[mask, "high"] = clean_df.loc[mask, close_col]
            clean_df.loc[mask, "low"] = clean_df.loc[mask, close_col]
            if "open" in clean_df.columns:
                clean_df["high"] = clean_df[["high", "open", "close"]].max(axis=1)
                clean_df["low"] = clean_df[["low", "open", "close"]].min(axis=1)

    return clean_df, mask


def detect_price_spikes(
    close: pd.Series,
    pct_threshold: float = 0.10,
) -> pd.Series:
    """Detect isolated price spikes that make a sudden move > pct_threshold and reverse next bar."""
    ret_in = close.pct_change()
    ret_out = close.pct_change().shift(-1)

    # Spike: large move in, opposite sign move out with substantial reversal
    is_large_move = ret_in.abs() > pct_threshold
    is_reversal = (ret_in * ret_out < 0) & (ret_out.abs() >= 0.5 * ret_in.abs())

    spikes = is_large_move & is_reversal
    return pd.Series(spikes.fillna(False), index=close.index, dtype=bool)
