from datetime import date, datetime, timedelta, timezone
from typing import Any
import pandas as pd

from config.trading.calendar import is_trading_day


def _normalize_to_date(val: Any) -> date:
    """Normalize input date/datetime/string to a datetime.date object."""
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, (str, pd.Timestamp)):
        dt = pd.to_datetime(val)
        return dt.date()
    raise TypeError(f"Cannot convert type {type(val)} to datetime.date")


def to_utc(ts: datetime | str | pd.Timestamp | int | float) -> datetime:
    """Convert a timestamp to a timezone-aware UTC datetime.

    Accepts datetime objects (naive or aware), ISO strings, pd.Timestamp,
    or unix timestamps in seconds/milliseconds.
    """
    if isinstance(ts, (int, float)):
        # Auto-detect seconds vs milliseconds
        unit = "s" if abs(ts) < 1e11 else "ms"
        return pd.to_datetime(ts, unit=unit, utc=True).to_pydatetime()

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    # Use pandas for robust parsing of strings, Timestamps, etc.
    parsed = pd.to_datetime(ts, utc=True)
    return parsed.to_pydatetime()


def business_days_between(
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    include_end: bool = False,
) -> int:
    """Calculate the number of business days (Monday-Friday) between two dates.

    By default, calculates [start_date, end_date) half-open.
    If include_end=True, calculates [start_date, end_date] closed.
    If start_date > end_date, returns the negative count.
    """
    start = _normalize_to_date(start_date)
    end = _normalize_to_date(end_date)

    if start == end:
        return 1 if (include_end and start.weekday() < 5) else 0

    if start > end:
        return -business_days_between(end, start, include_end=include_end)

    cur = start
    count = 0
    while cur < end:
        if cur.weekday() < 5:
            count += 1
        cur += timedelta(days=1)

    if include_end and end.weekday() < 5:
        count += 1

    return count


def add_business_days(start_date: date | datetime | str, n: int) -> date:
    """Add n business days (Monday-Friday) to start_date.

    If n is negative, moves backwards in time.
    """
    cur = _normalize_to_date(start_date)
    if n == 0:
        return cur

    step = 1 if n > 0 else -1
    remaining = abs(n)

    while remaining > 0:
        cur += timedelta(days=step)
        if cur.weekday() < 5:
            remaining -= 1

    return cur


def trading_days_between(
    a: date | datetime | str,
    b: date | datetime | str,
    calendar: str = "NYSE",
    include_end: bool = False,
) -> int:
    """Count the number of active trading days between dates a and b using the specified calendar.

    By default, counts [a, b) half-open.
    If include_end=True, counts [a, b] closed.
    If a > b, returns the negative count.
    """
    start = _normalize_to_date(a)
    end = _normalize_to_date(b)

    if start == end:
        return 1 if (include_end and is_trading_day(start, calendar=calendar)) else 0

    if start > end:
        return -trading_days_between(end, start, calendar=calendar, include_end=include_end)

    cur = start
    count = 0
    while cur < end:
        if is_trading_day(cur, calendar=calendar):
            count += 1
        cur += timedelta(days=1)

    if include_end and is_trading_day(end, calendar=calendar):
        count += 1

    return count
