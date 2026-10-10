from datetime import date, datetime
from typing import Any
import pandas as pd


def _normalize_date_index(d: Any) -> pd.Timestamp:
    """Normalize input date/timestamp to a pd.Timestamp."""
    if isinstance(d, pd.Timestamp):
        return d.normalize()
    if isinstance(d, datetime):
        return pd.Timestamp(d).normalize()
    if isinstance(d, date):
        return pd.Timestamp(d)
    return pd.Timestamp(str(d).split("T")[0].split(" ")[0])


def adjust_for_splits(
    df: pd.DataFrame,
    splits: dict[str | date | datetime | pd.Timestamp, float],
) -> pd.DataFrame:
    """Adjust historical prices and volumes for stock splits.

    For each split date with ratio:
    - Pre-split prices (open, high, low, close) are divided by ratio
    - Pre-split volume is multiplied by ratio
    - Total dollar volume (price * volume) is strictly preserved.
    """
    if df.empty or not splits:
        return df.copy()

    adj_df = df.copy()
    price_cols = [c for c in ["open", "high", "low", "close", "Open", "High", "Low", "Close"] if c in adj_df.columns]
    vol_cols = [c for c in ["volume", "Volume"] if c in adj_df.columns]

    df_index_normalized = pd.DatetimeIndex([_normalize_date_index(idx) for idx in adj_df.index])

    for split_date, ratio in sorted(splits.items(), key=lambda x: _normalize_date_index(x[0])):
        if ratio <= 0:
            raise ValueError(f"Split ratio must be positive, got {ratio}")

        target_date = _normalize_date_index(split_date)
        mask = df_index_normalized < target_date

        if mask.any():
            for pcol in price_cols:
                adj_df.loc[mask, pcol] = adj_df.loc[mask, pcol] / ratio
            for vcol in vol_cols:
                adj_df.loc[mask, vcol] = adj_df.loc[mask, vcol] * ratio

    return adj_df


def adjust_for_dividends(
    df: pd.DataFrame,
    dividends: dict[str | date | datetime | pd.Timestamp, float],
) -> pd.DataFrame:
    """Adjust historical prices for cash dividends using backward multiplicative factor.

    For each dividend at ex-date:
    factor = 1.0 - (dividend_amount / previous_close)
    All price bars occurring strictly BEFORE the ex-date are scaled by factor.
    """
    if df.empty or not dividends:
        return df.copy()

    adj_df = df.copy()
    close_col = "close" if "close" in adj_df.columns else "Close"
    price_cols = [c for c in ["open", "high", "low", "close", "Open", "High", "Low", "Close"] if c in adj_df.columns]

    df_index_normalized = pd.DatetimeIndex([_normalize_date_index(idx) for idx in adj_df.index])

    for ex_date, amt in sorted(dividends.items(), key=lambda x: _normalize_date_index(x[0])):
        if amt < 0:
            raise ValueError(f"Dividend amount cannot be negative, got {amt}")

        target_date = _normalize_date_index(ex_date)
        prior_mask = df_index_normalized < target_date

        if not prior_mask.any():
            continue

        # Previous close is the last available close strictly before the ex-date
        prev_close_val = adj_df.loc[prior_mask, close_col].iloc[-1]
        if prev_close_val <= 0:
            continue

        factor = 1.0 - (amt / prev_close_val)

        for pcol in price_cols:
            adj_df.loc[prior_mask, pcol] = adj_df.loc[prior_mask, pcol] * factor

    return adj_df
