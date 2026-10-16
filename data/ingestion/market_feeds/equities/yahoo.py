import json
import urllib.error
import urllib.request
from typing import Any
import pandas as pd


class FeedError(Exception):
    """Raised when an external market data or news feed fetch fails."""
    pass


def parse_yahoo_chart(raw_text: str) -> pd.DataFrame:
    """Parse Yahoo Finance v8 chart JSON text into an OHLCV DataFrame."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FeedError(f"Failed to decode Yahoo chart JSON: {exc}") from exc

    chart = data.get("chart", {})
    if chart.get("error"):
        raise FeedError(f"Yahoo chart returned error: {chart['error']}")

    results = chart.get("result")
    if not results or not isinstance(results, list):
        raise FeedError("Yahoo chart returned no result.")

    result_data = results[0]
    timestamps = result_data.get("timestamp")
    if not timestamps:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    indicators = result_data.get("indicators", {})
    quote_list = indicators.get("quote", [])
    if not quote_list:
        raise FeedError("No quote data found in Yahoo chart indicators.")

    quotes = quote_list[0]
    dt_index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)

    df = pd.DataFrame(
        {
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", []),
        },
        index=dt_index,
    )
    # Drop missing rows
    df = df.dropna(subset=["close"]).astype(float)
    return df


def fetch_yahoo_chart(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Fetch historical chart OHLCV data from Yahoo Finance using standard library urllib."""
    clean_symbol = symbol.strip().upper()
    
    if start is not None or end is not None:
        try:
            start_ts = pd.to_datetime(start, utc=True) if start else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=730)
            end_ts = pd.to_datetime(end, utc=True) if end else pd.Timestamp.now(tz="UTC")
            p1 = int(start_ts.timestamp())
            p2 = int(end_ts.timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_symbol}?interval=1d&period1={p1}&period2={p2}"
        except Exception:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_symbol}?interval=1d&range=2y"
    else:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_symbol}?interval=1d&range=2y"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeedError(f"Failed to fetch Yahoo chart for {symbol}: {exc}") from exc

    return parse_yahoo_chart(raw_text)
