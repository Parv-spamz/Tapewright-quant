from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def underwater_curve(equity: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Calculate the underwater curve (percentage drawdown from high-water mark).

    Underwater(t) = (Equity(t) - Peak(t)) / Peak(t) <= 0.0

    Args:
        equity: Time series or array of portfolio equity values.

    Returns:
        pd.Series | np.ndarray: Underwater drawdown values.
    """
    is_series = isinstance(equity, pd.Series)
    vals = np.asarray(equity, dtype=float)
    if len(vals) == 0:
        return pd.Series(dtype=float) if is_series else np.array([], dtype=float)

    peaks = np.maximum.accumulate(vals)
    # Guard against zero or negative peak
    safe_peaks = np.where(peaks > 0.0, peaks, 1.0)
    dd = (vals - safe_peaks) / safe_peaks

    if is_series:
        return pd.Series(dd, index=equity.index)
    return dd


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Calculate the maximum peak-to-trough drawdown (reported as a negative number, e.g., -0.25).

    Args:
        equity: Time series or array of equity values.

    Returns:
        float: Maximum drawdown (<= 0.0).
    """
    uw = underwater_curve(equity)
    if len(uw) == 0:
        return 0.0
    return float(np.min(uw))


def drawdown_periods(equity: pd.Series | np.ndarray) -> pd.DataFrame:
    """Decompose an equity curve into distinct peak-to-recovery drawdown periods.

    Identifies drawdown start (peak), trough (valley), end (recovery), depth,
    total duration (in bars), and recovery duration (in bars). Correctly handles
    open / unrecovered drawdown periods at the end of the series.

    Args:
        equity: Time series or array of equity values.

    Returns:
        pd.DataFrame: DataFrame with columns:
            ['start', 'trough', 'end', 'depth', 'duration', 'recovery_duration']
    """
    cols = ["start", "trough", "end", "depth", "duration", "recovery_duration"]
    if len(equity) == 0:
        return pd.DataFrame(columns=cols)

    s = pd.Series(equity) if not isinstance(equity, pd.Series) else equity
    vals = s.values.astype(float)
    idx = s.index
    T = len(vals)

    peaks = np.maximum.accumulate(vals)
    safe_peaks = np.where(peaks > 0.0, peaks, 1.0)
    dd = (vals - safe_peaks) / safe_peaks

    records: list[dict[str, Any]] = []
    in_drawdown = False
    start_idx = 0
    trough_idx = 0
    trough_dd = 0.0

    current_peak_val = vals[0]
    current_peak_idx = 0

    for t in range(T):
        val = vals[t]
        if val >= current_peak_val:
            if in_drawdown:
                # Drawdown has fully recovered at bar t
                records.append({
                    "start": idx[start_idx],
                    "trough": idx[trough_idx],
                    "end": idx[t],
                    "depth": float(trough_dd),
                    "duration": int(t - start_idx),
                    "recovery_duration": int(t - trough_idx),
                })
                in_drawdown = False

            current_peak_val = val
            current_peak_idx = t
        else:
            if not in_drawdown:
                in_drawdown = True
                start_idx = current_peak_idx
                trough_idx = t
                trough_dd = dd[t]
            else:
                if dd[t] < trough_dd:
                    trough_dd = dd[t]
                    trough_idx = t

    # Handle open / unrecovered period at the tail
    if in_drawdown:
        records.append({
            "start": idx[start_idx],
            "trough": idx[trough_idx],
            "end": None,
            "depth": float(trough_dd),
            "duration": int((T - 1) - start_idx),
            "recovery_duration": None,
        })

    return pd.DataFrame(records, columns=cols)


def max_drawdown_duration(equity: pd.Series | np.ndarray) -> int:
    """Calculate the maximum drawdown duration in bars across all drawdown episodes."""
    df = drawdown_periods(equity)
    if df.empty:
        return 0
    return int(df["duration"].max())


def recovery_factor(equity: pd.Series | np.ndarray) -> float:
    """Calculate Recovery Factor: Cumulative Return / abs(Maximum Drawdown)."""
    vals = np.asarray(equity, dtype=float)
    if len(vals) <= 1:
        return 0.0

    mdd = abs(max_drawdown(vals))
    if mdd <= 1e-14:
        return 0.0

    total_return = (vals[-1] - vals[0]) / vals[0]
    return float(total_return / mdd)
