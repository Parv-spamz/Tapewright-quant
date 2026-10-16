from datetime import date, datetime
from typing import Any
import pandas as pd

from data.ingestion.market_feeds.crypto.ccxt_feed import fetch_binance_klines
from data.ingestion.market_feeds.equities.synthetic import make_ohlcv
from data.ingestion.market_feeds.equities.yahoo import FeedError, fetch_yahoo_chart
from utils.logging.logger import get_logger

logger = get_logger("market_provider")


def get_prices(
    symbol: str,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    source: str = "auto",
) -> pd.DataFrame:
    """Retrieve OHLCV market data for a symbol across a specified date range.

    If source is 'auto', 'real', 'live', 'yahoo', or 'crypto', attempts to fetch
    live market data from Yahoo Finance or Binance.
    If the live feed fails (e.g. offline, rate-limited), it gracefully falls back
    to deterministic synthetic generation, setting df.attrs['source'] = 'synthetic'
    and logging a warning so simulated data is never mistaken for live quotes.
    """
    clean_source = source.strip().lower()
    clean_symbol = symbol.strip().upper()

    start_str = str(start) if start is not None else "2023-01-01"
    end_str = str(end) if end is not None else str(pd.Timestamp.now().strftime("%Y-%m-%d"))

    if clean_source in ("auto", "live", "real", "yahoo", "crypto", "binance"):
        try:
            if clean_source in ("crypto", "binance") or clean_symbol.endswith("USDT") or clean_symbol in ("BTC", "ETH", "SOL"):
                crypto_pair = clean_symbol if clean_symbol.endswith("USDT") else f"{clean_symbol}USDT"
                df = fetch_binance_klines(crypto_pair, limit=1000)
            else:
                df = fetch_yahoo_chart(clean_symbol, start=start_str, end=end_str)

            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            df.index = df.index.normalize()

            if start is not None:
                start_ts = pd.to_datetime(start)
                if getattr(start_ts, "tz", None) is not None:
                    start_ts = start_ts.tz_localize(None)
                start_ts = start_ts.normalize()
                df = df[df.index >= start_ts]
            if end is not None:
                end_ts = pd.to_datetime(end)
                if getattr(end_ts, "tz", None) is not None:
                    end_ts = end_ts.tz_localize(None)
                end_ts = end_ts.normalize()
                df = df[df.index <= end_ts]

            if not df.empty:
                df.attrs["source"] = "real"
                return df
            else:
                logger.warning(f"Live feed returned empty DataFrame for {clean_symbol}. Falling back to synthetic.")
        except (FeedError, Exception) as exc:
            logger.warning(
                f"Live market feed failed for {clean_symbol} ({exc}). "
                "Falling back to synthetic data generator."
            )

    # Synthetic fallback
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    cal_days = max(15, (end_dt - start_dt).days)
    n_days = max(10, int(cal_days * (252.0 / 365.0)) + 5)

    df_synth = make_ohlcv(
        symbol=clean_symbol,
        start=start_str,
        n_days=n_days,
        seed=abs(hash(clean_symbol)) % 100_000,
        model="gbm",
    )
    df_synth = df_synth[df_synth.index >= pd.to_datetime(start_str)]
    df_synth = df_synth[df_synth.index <= pd.to_datetime(end_str)]

    df_synth.attrs["source"] = "synthetic"
    return df_synth
