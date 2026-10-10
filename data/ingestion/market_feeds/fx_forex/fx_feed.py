import json
import urllib.error
import urllib.request
import pandas as pd

from data.ingestion.market_feeds.equities.yahoo import FeedError


def parse_fx_rates(raw_text: str) -> pd.DataFrame:
    """Parse Foreign Exchange rates JSON into a pandas DataFrame."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FeedError(f"Failed to decode FX rates JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise FeedError(f"Expected dict from FX feed, got {type(data)}")

    rates = data.get("rates", {})
    if not rates:
        return pd.DataFrame()

    # Check if rates is keyed by date: {"2024-01-02": {"EUR": 0.91, "JPY": 142.0}}
    first_val = next(iter(rates.values()))
    if isinstance(first_val, dict):
        records = []
        dates = []
        for d_str, c_dict in rates.items():
            dates.append(pd.to_datetime(d_str))
            records.append({k: float(v) for k, v in c_dict.items()})
        df = pd.DataFrame(records, index=pd.DatetimeIndex(dates))
        return df.sort_index()

    # Single-date response: {"date": "2024-01-02", "rates": {"EUR": 0.91, "JPY": 142.0}}
    date_str = data.get("date", "2024-01-01")
    row = {k: float(v) for k, v in rates.items()}
    return pd.DataFrame([row], index=pd.DatetimeIndex([pd.to_datetime(date_str)]))


def fetch_fx_rates(
    base: str = "USD",
    symbols: list[str] | str = "EUR,JPY",
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Fetch Foreign Exchange rates from open public ECB/Frankfurter API."""
    sym_str = ",".join(symbols) if isinstance(symbols, (list, tuple)) else symbols
    url = f"https://api.frankfurter.app/latest?from={base}&to={sym_str}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeedError(f"Failed to fetch FX rates for base {base}: {exc}") from exc

    return parse_fx_rates(raw_text)
