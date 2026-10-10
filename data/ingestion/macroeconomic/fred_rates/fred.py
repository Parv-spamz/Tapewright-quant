import csv
import io
import logging
import urllib.error
import urllib.request
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from data.ingestion.market_feeds.equities.yahoo import FeedError


def parse_fred_csv(raw_text: str) -> pd.Series:
    """Parse FRED CSV string into a pandas Series with DatetimeIndex.

    Replaces missing marker '.' with np.nan.
    """
    if not raw_text.strip():
        return pd.Series(dtype=float)

    try:
        reader = csv.reader(io.StringIO(raw_text))
        header = next(reader, None)
        if not header or len(header) < 2:
            return pd.Series(dtype=float)

        series_name = header[1].strip()
        dates: list[pd.Timestamp] = []
        values: list[float] = []

        for row in reader:
            if not row or len(row) < 2:
                continue
            date_str = row[0].strip()
            val_str = row[1].strip()

            try:
                dt = pd.to_datetime(date_str)
                dates.append(dt)
            except Exception as e:
                logger.warning("Failed to parse FRED date '%s': %s", date_str, e)
                continue

            if val_str == "." or not val_str:
                values.append(np.nan)
            else:
                try:
                    values.append(float(val_str))
                except ValueError:
                    values.append(np.nan)

        return pd.Series(data=values, index=pd.DatetimeIndex(dates), name=series_name, dtype=float)
    except Exception as exc:
        raise FeedError(f"Failed to parse FRED CSV: {exc}") from exc


def fetch_series(series_id: str, timeout: float = 10.0) -> pd.Series:
    """Fetch an economic time series from the St. Louis Federal Reserve (FRED) via CSV."""
    clean_id = series_id.strip().upper()
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={clean_id}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeedError(f"Failed to fetch FRED series {series_id}: {exc}") from exc

    return parse_fred_csv(raw_text)


def yield_curve_spread(dgs10: pd.Series, dgs2: pd.Series) -> pd.Series:
    """Calculate the 10Y-2Y Treasury yield curve spread (DGS10 - DGS2).

    Aligns on dates, forward-fills gaps up to max 3 days, and drops remaining NaNs.
    """
    aligned = pd.DataFrame({"10": dgs10, "2": dgs2}).sort_index()

    # Forward fill gaps up to max 3 days
    filled = aligned.ffill(limit=3).dropna()

    spread = filled["10"] - filled["2"]
    spread.name = "yield_spread"
    return spread


def inversion_flag(spread: pd.Series) -> pd.Series:
    """Return a boolean Series indicating yield curve inversion (spread < 0)."""
    return (spread < 0).rename("is_inverted")


def inversion_episodes(spread: pd.Series) -> pd.DataFrame:
    """Identify contiguous historical episodes of yield curve inversion.

    Returns:
        pd.DataFrame with columns: [start, end, days, min_spread]
    """
    if spread.empty:
        return pd.DataFrame(columns=["start", "end", "days", "min_spread"])

    is_inv = spread < 0
    if not is_inv.any():
        return pd.DataFrame(columns=["start", "end", "days", "min_spread"])

    # Group contiguous True regions
    change_points = is_inv != is_inv.shift(1)
    episode_ids = change_points.cumsum()

    records: list[dict[str, Any]] = []

    for eid, sub_mask in is_inv.groupby(episode_ids):
        if not sub_mask.iloc[0]:
            continue

        sub_spread = spread.loc[sub_mask.index]
        start_dt = sub_spread.index[0]
        end_dt = sub_spread.index[-1]
        days_span = (end_dt - start_dt).days + 1
        min_val = float(sub_spread.min())

        records.append(
            {
                "start": start_dt,
                "end": end_dt,
                "days": days_span,
                "min_spread": min_val,
            }
        )

    return pd.DataFrame(records)
