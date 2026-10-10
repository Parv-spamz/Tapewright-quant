import json
import urllib.error
import urllib.request
import pandas as pd

from data.ingestion.market_feeds.equities.yahoo import FeedError


def parse_binance_klines(raw_text: str) -> pd.DataFrame:
    """Parse Binance public klines REST JSON into an OHLCV DataFrame."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FeedError(f"Failed to decode Binance klines JSON: {exc}") from exc

    if not isinstance(data, list):
        raise FeedError(f"Expected list of klines, got {type(data)}: {data}")

    if not data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    timestamps: list[pd.Timestamp] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []

    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        timestamps.append(pd.to_datetime(item[0], unit="ms", utc=True))
        opens.append(float(item[1]))
        highs.append(float(item[2]))
        lows.append(float(item[3]))
        closes.append(float(item[4]))
        volumes.append(float(item[5]))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=pd.DatetimeIndex(timestamps),
    )
    return df


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    limit: int = 500,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Fetch public cryptocurrency klines from Binance public REST API."""
    clean_symbol = symbol.replace("-", "").replace("/", "").upper()
    url = f"https://api.binance.com/api/v3/klines?symbol={clean_symbol}&interval={interval}&limit={limit}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeedError(f"Failed to fetch Binance klines for {symbol}: {exc}") from exc

    return parse_binance_klines(raw_text)
