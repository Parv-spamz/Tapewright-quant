"""Strict real market data ingestion, provenance archiving, and partition storage.

Supported universes:
- Equities (Liquid ETFs): SPY, QQQ, IWM, EFA, TLT, IEF, GLD, XLE, XLF, XLK, XLV, XLY
- Crypto: BTCUSDT, ETHUSDT
- Macro (FRED): DGS10, DGS2, VIXCLS

All network data is fetched using Python standard library networking only,
content-addressed in research/raw_archive/, indexed by cryptographic manifests,
and partitioned into research/store_devval and research/vault without synthetic fallbacks.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import random
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from config.trading.calendar import is_trading_day
from data.caching.parquet_store.store import ParquetStore
from data.validation.schema_rules.rules import validate_ohlcv
from integrity.ledger import Ledger
from integrity.provenance import (
    DatasetManifest,
    ProvenanceError,
    archive_raw,
    looks_synthetic_gbm,
    parsed_hash,
    realism_fingerprint,
)
from research.splits import CRYPTO_SPLITS, EQUITY_SPLITS, split_of

logger = logging.getLogger(__name__)

EQUITY_UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "GLD", "XLE", "XLF", "XLK", "XLV", "XLY"
]
CRYPTO_UNIVERSE = ["BTCUSDT", "ETHUSDT"]
MACRO = ["DGS10", "DGS2", "VIXCLS"]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HTTP_TIMEOUT_SECONDS = 30


class FeedError(Exception):
    """Raised when an external feed fails, returns corrupt data, or violates null thresholds."""


def _http_get(url: str, headers: dict[str, str] | None = None) -> bytes:
    """Perform HTTP GET using standard library urllib with user agent and timeout."""
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_yahoo_json(data: bytes | str) -> pd.DataFrame:
    """Parse Yahoo Finance v8 chart JSON response.

    Applies exchange-local date adjustment via gmtoffset, calculates
    total-return adjusted OHLC (adj_factor = adjclose / close), enforces
    null thresholds (< 0.5%), and returns clean DatetimeIndex DataFrame.
    Note: Returns are on a total-return basis without look-ahead in return calculations.
    """
    if isinstance(data, str):
        payload = json.loads(data)
    else:
        payload = json.loads(data.decode("utf-8"))

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise FeedError(f"Yahoo API returned error: {error}")

    result = chart.get("result")
    if not result:
        raise FeedError("Yahoo API response missing 'result' array")

    res0 = result[0]
    meta = res0.get("meta", {})
    gmtoffset = meta.get("gmtoffset", 0)

    timestamps = res0.get("timestamp", [])
    if not timestamps:
        raise FeedError("Yahoo API returned empty timestamps array")

    indicators = res0.get("indicators", {})
    quotes = indicators.get("quote", [{}])[0]
    adjquotes = indicators.get("adjclose", [{}])[0]

    opens = quotes.get("open", [])
    highs = quotes.get("high", [])
    lows = quotes.get("low", [])
    closes = quotes.get("close", [])
    volumes = quotes.get("volume", [])
    adjcloses = adjquotes.get("adjclose", closes)

    n_raw = len(timestamps)
    records: list[dict[str, Any]] = []
    null_count = 0

    for i in range(n_raw):
        ts = timestamps[i]
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        l = lows[i] if i < len(lows) else None
        c = closes[i] if i < len(closes) else None
        v = volumes[i] if i < len(volumes) else None
        ac = adjcloses[i] if i < len(adjcloses) else None

        if None in (o, h, l, c, ac) or pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c) or pd.isna(ac):
            null_count += 1
            continue

        local_dt = pd.to_datetime(ts + gmtoffset, unit="s").normalize()
        records.append({
            "timestamp": local_dt,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(v) if v is not None and not pd.isna(v) else 0.0,
            "adjclose": float(ac),
        })

    if n_raw > 0 and (null_count / n_raw) > 0.005:
        raise FeedError(f"Yahoo feed contains {null_count}/{n_raw} null rows ({null_count / n_raw:.2%}), exceeding 0.5% limit")

    if not records:
        raise FeedError("No valid rows remaining after filtering null entries")

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]

    # Calculate adj_factor and adjust OHLC on total-return basis
    adj_factor = df["adjclose"] / df["close"]
    df["open"] = df["open"] * adj_factor
    df["high"] = df["high"] * adj_factor
    df["low"] = df["low"] * adj_factor
    df["close"] = df["adjclose"]
    df = df.drop(columns=["adjclose"])

    # High / low physical sanity bounds after float multiplication
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    return df[["open", "high", "low", "close", "volume"]]


def parse_binance_klines(data: bytes | str | list[Any]) -> pd.DataFrame:
    """Parse Binance daily klines array.

    Keeps columns: open, high, low, close, volume, quote_volume, n_trades, taker_buy_base, taker_buy_quote.
    Converts string numerics to float and indexes by midnight UTC datetime.
    """
    if isinstance(data, (bytes, str)):
        raw_list = json.loads(data if isinstance(data, str) else data.decode("utf-8"))
    else:
        raw_list = data

    if not raw_list:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for item in raw_list:
        open_time_ms = item[0]
        dt = pd.to_datetime(open_time_ms, unit="ms", utc=True).tz_convert(None).normalize()
        records.append({
            "timestamp": dt,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
            "quote_volume": float(item[7]),
            "n_trades": float(item[8]),
            "taker_buy_base": float(item[9]),
            "taker_buy_quote": float(item[10]),
        })

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    return df


def parse_fred_csv(data: bytes | str) -> pd.DataFrame:
    """Parse FRED CSV string / bytes. Replaces '.' with NaN and drops missing values."""
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or len(header) < 2:
        raise FeedError("Invalid or empty FRED CSV header")

    series_col = header[1].strip()
    records: list[dict[str, Any]] = []

    for row in reader:
        if not row or len(row) < 2:
            continue
        date_str, val_str = row[0].strip(), row[1].strip()
        if val_str == "." or not val_str:
            continue
        try:
            val_float = float(val_str)
            records.append({
                "timestamp": pd.to_datetime(date_str).normalize(),
                "value": val_float,
            })
        except ValueError as e:
            logger.warning("Failed to parse FRED observation %r: %s", val_str, e)
            continue

    if not records:
        raise FeedError(f"FRED series '{series_col}' returned 0 parseable observations")

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    return df


def load_user_csv(path: Path | str, symbol: str) -> tuple[pd.DataFrame, bytes, str]:
    """Load Yahoo-style manual user CSV with Date,Open,High,Low,Close,Adj Close,Volume."""
    fpath = Path(path).resolve()
    if not fpath.is_file():
        raise FileNotFoundError(f"User CSV not found: {fpath}")

    raw_bytes = fpath.read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()

    df = pd.read_csv(io.BytesIO(raw_bytes))
    lower_map = {str(c).lower().replace(" ", ""): c for c in df.columns}

    required = ["date", "open", "high", "low", "close", "adjclose", "volume"]
    for r in required:
        if r not in lower_map:
            raise FeedError(f"User CSV missing required column '{r}'")

    date_col = lower_map["date"]
    df["timestamp"] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.set_index("timestamp").sort_index()

    o = df[lower_map["open"]].astype(float)
    h = df[lower_map["high"]].astype(float)
    l = df[lower_map["low"]].astype(float)
    c = df[lower_map["close"]].astype(float)
    ac = df[lower_map["adjclose"]].astype(float)
    v = df[lower_map["volume"]].astype(float)

    adj_factor = ac / c
    df_clean = pd.DataFrame({
        "open": o * adj_factor,
        "high": (h * adj_factor),
        "low": (l * adj_factor),
        "close": ac,
        "volume": v,
    }, index=df.index)

    df_clean["high"] = df_clean[["high", "open", "close"]].max(axis=1)
    df_clean["low"] = df_clean[["low", "open", "close"]].min(axis=1)

    return df_clean, raw_bytes, raw_sha


# ---------------------------------------------------------------------------
# Ingestion Orchestrators
# ---------------------------------------------------------------------------

def fetch_yahoo_symbol(
    symbol: str,
    start_unix: int = 1104537600,  # 2005-01-01
    end_unix: int | None = None,
    archive_dir: Path | str = "research/raw_archive",
) -> tuple[pd.DataFrame, list[str], bytes]:
    """Fetch daily OHLCV from Yahoo Finance API, archiving the raw response page."""
    if end_unix is None:
        end_unix = int(time.time())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start_unix}&period2={end_unix}&interval=1d&events=div%2Csplits"
    )

    raw_data = _http_get(url)
    page_sha = archive_raw(raw_data, archive_dir)
    df = parse_yahoo_json(raw_data)
    return df, [page_sha], raw_data


def fetch_binance_symbol(
    symbol: str,
    start_ms: int = 1502928000000,  # 2017-08-17
    archive_dir: Path | str = "research/raw_archive",
    fetch_func: Any = None,
) -> tuple[pd.DataFrame, list[str], bytes, str]:
    """Fetch daily klines from Binance API with pagination, archiving every raw page."""
    bases = ["https://data-api.binance.vision", "https://api.binance.com"]
    raw_pages: list[str] = []
    all_raw_bytes: list[bytes] = []
    all_frames: list[pd.DataFrame] = []
    used_base = ""

    for base in bases:
        try:
            curr_start = start_ms
            pages_for_base: list[str] = []
            bytes_for_base: list[bytes] = []
            frames_for_base: list[pd.DataFrame] = []

            while True:
                url = f"{base}/api/v3/klines?symbol={symbol}&interval=1d&startTime={curr_start}&limit=1000"
                if fetch_func is not None:
                    page_bytes = fetch_func(url)
                else:
                    page_bytes = _http_get(url)

                if not page_bytes or page_bytes.strip() == b"[]":
                    break

                page_sha = archive_raw(page_bytes, archive_dir)
                pages_for_base.append(page_sha)
                bytes_for_base.append(page_bytes)

                frame = parse_binance_klines(page_bytes)
                if frame.empty:
                    break
                frames_for_base.append(frame)

                # Next page starts 1 ms after last bar's end (86400000 ms per day)
                last_ts = frame.index[-1]
                last_ms = int(last_ts.timestamp() * 1000) + 86400000
                if last_ms <= curr_start or len(frame) < 1000:
                    break
                curr_start = last_ms

            if frames_for_base:
                used_base = base
                raw_pages = pages_for_base
                all_raw_bytes = bytes_for_base
                all_frames = frames_for_base
                break

        except Exception as e:
            logger.warning("Binance base %s failed for %s: %s", base, symbol, e)
            continue

    if not all_frames:
        raise FeedError(f"All Binance endpoints failed for symbol {symbol}")

    combined = pd.concat(all_frames, axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    concatenated_raw = b"".join(all_raw_bytes)

    return combined, raw_pages, concatenated_raw, used_base


def fetch_fred_series(
    series: str,
    archive_dir: Path | str = "research/raw_archive",
) -> tuple[pd.DataFrame, list[str], bytes]:
    """Fetch macroeconomic series CSV from FRED, archiving the raw response page."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2005-01-01"
    raw_data = _http_get(url)
    page_sha = archive_raw(raw_data, archive_dir)
    df = parse_fred_csv(raw_data)
    return df, [page_sha], raw_data


# ---------------------------------------------------------------------------
# Invariant Validation
# ---------------------------------------------------------------------------

def validate_fetched_dataset(
    df: pd.DataFrame,
    symbol: str,
    is_equity: bool,
    fp: dict[str, Any],
) -> None:
    """Validate physical invariants, NYSE calendar alignment, and lack of synthetic patterns."""
    if is_equity:
        # 1. Structural OHLCV validation
        report = validate_ohlcv(df[["open", "high", "low", "close", "volume"]], allowed_gap_days=5)
        if not report.ok:
            raise ProvenanceError(f"OHLCV validation failed for equity {symbol}: {report.errors}")

        # 2. No bars on NYSE holidays
        for dt in df.index:
            if not is_trading_day(dt, calendar="NYSE"):
                raise ProvenanceError(f"Equity {symbol} contains observation on non-trading day / NYSE holiday: {dt}")

        # 3. Missing trading days <= 1.0%
        first_date = df.index[0].date()
        last_date = df.index[-1].date()
        expected_days = 0
        curr = first_date
        while curr <= last_date:
            if is_trading_day(curr, calendar="NYSE"):
                expected_days += 1
            curr = date.fromordinal(curr.toordinal() + 1)

        actual_days = len(df)
        missing_count = max(0, expected_days - actual_days)
        missing_pct = missing_count / max(1, expected_days)
        if missing_pct > 0.01:
            raise ProvenanceError(
                f"Equity {symbol} missing {missing_count}/{expected_days} trading days ({missing_pct:.2%}), exceeding 1.0% limit"
            )

    # 4. Anti-synthetic verification
    if looks_synthetic_gbm(fp):
        raise ProvenanceError(f"Dataset for {symbol} exhibits synthetic GBM characteristics ({fp})")


# ---------------------------------------------------------------------------
# Storage & Partitioning
# ---------------------------------------------------------------------------

def store_dataset_splits(
    symbol: str,
    df: pd.DataFrame,
    is_crypto: bool,
    store_devval_path: Path | str = "research/store_devval",
    vault_path: Path | str = "research/vault",
) -> tuple[int, int]:
    """Partition DataFrame into dev/val (research/store_devval) and lockbox (research/vault)."""
    splits_map = CRYPTO_SPLITS if is_crypto else EQUITY_SPLITS

    devval_rows = []
    lockbox_rows = []

    for dt, row in df.iterrows():
        split = split_of(dt, splits_map)
        if split in ("dev", "val"):
            devval_rows.append(row)
        elif split == "lockbox":
            lockbox_rows.append(row)

    store_devval = ParquetStore(store_devval_path)
    vault = ParquetStore(vault_path)

    if devval_rows:
        devval_df = pd.DataFrame(devval_rows)
        devval_df.index = pd.DatetimeIndex([r.name for r in devval_rows])
        store_devval.write(symbol, devval_df, "1d")

    if lockbox_rows:
        lockbox_df = pd.DataFrame(lockbox_rows)
        lockbox_df.index = pd.DatetimeIndex([r.name for r in lockbox_rows])
        vault.write(symbol, lockbox_df, "1d")

    return len(devval_rows), len(lockbox_rows)


# ---------------------------------------------------------------------------
# Orchestrator & CLI Entrypoints
# ---------------------------------------------------------------------------

def fetch_all(
    archive_dir: Path | str = "research/raw_archive",
    manifest_dir: Path | str = "research/manifests",
    store_devval_path: Path | str = "research/store_devval",
    vault_path: Path | str = "research/vault",
    ledger_path: Path | str = "research/ledger/ledger.jsonl",
) -> list[dict[str, Any]]:
    """Fetch all equities, crypto, and macro datasets with strict provenance manifests."""
    manifest_path = Path(manifest_dir).resolve()
    manifest_path.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(ledger_path)

    results: list[dict[str, Any]] = []

    # 1. Equities
    for sym in EQUITY_UNIVERSE:
        ts_utc = datetime.now(timezone.utc).isoformat()
        try:
            df, raw_pages, raw_bytes = fetch_yahoo_symbol(sym, archive_dir=archive_dir)
            raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            p_sha = parsed_hash(df)
            returns = df["close"].pct_change().dropna()
            fp = realism_fingerprint(returns)

            validate_fetched_dataset(df, sym, is_equity=True, fp=fp)
            devval_n, vault_n = store_dataset_splits(
                sym, df, is_crypto=False, store_devval_path=store_devval_path, vault_path=vault_path
            )

            manifest = DatasetManifest.create(
                source_kind="real_api",
                source_name=f"yahoo.{sym}",
                params={"symbol": sym, "interval": "1d"},
                fetched_at_utc=ts_utc,
                n_rows=len(df),
                raw_pages=raw_pages,
                raw_sha256=raw_sha,
                parsed_sha256=p_sha,
                fingerprint=fp,
                url=f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                first_ts=str(df.index[0]),
                last_ts=str(df.index[-1]),
            )
            m_file = manifest_path / f"{manifest.dataset_id}.json"
            m_file.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")

            ledger.append(
                kind="data_fetch",
                status="PASS",
                tag=f"fetch.equity.{sym}",
                payload={
                    "dataset_id": manifest.dataset_id,
                    "symbol": sym,
                    "n_rows": len(df),
                    "first_ts": manifest.first_ts,
                    "last_ts": manifest.last_ts,
                    "devval_rows": devval_n,
                    "vault_rows": vault_n,
                },
            )
            results.append({
                "id": manifest.dataset_id,
                "symbol": sym,
                "rows": len(df),
                "first_date": str(df.index[0].date()),
                "last_date": str(df.index[-1].date()),
                "status": "PASS",
            })
            print(f"{sym:<10} | {manifest.dataset_id[:16]} | rows={len(df):<5} | {df.index[0].date()} to {df.index[-1].date()} | PASS")

        except Exception as e:
            ledger.append(
                kind="data_fetch",
                status="NOT_RUN",
                tag=f"fetch.equity.{sym}",
                payload={"symbol": sym, "error": str(e)},
            )
            results.append({"id": "N/A", "symbol": sym, "rows": 0, "status": f"NOT_RUN({e})"})
            print(f"{sym:<10} | NOT_RUN({e})")

    # 2. Crypto
    for sym in CRYPTO_UNIVERSE:
        ts_utc = datetime.now(timezone.utc).isoformat()
        try:
            df, raw_pages, raw_bytes, used_base = fetch_binance_symbol(sym, archive_dir=archive_dir)
            raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            p_sha = parsed_hash(df)
            returns = df["close"].pct_change().dropna()
            fp = realism_fingerprint(returns)

            validate_fetched_dataset(df, sym, is_equity=False, fp=fp)
            devval_n, vault_n = store_dataset_splits(
                sym, df, is_crypto=True, store_devval_path=store_devval_path, vault_path=vault_path
            )

            manifest = DatasetManifest.create(
                source_kind="real_api",
                source_name=f"binance.{sym}",
                params={"symbol": sym, "interval": "1d", "base": used_base},
                fetched_at_utc=ts_utc,
                n_rows=len(df),
                raw_pages=raw_pages,
                raw_sha256=raw_sha,
                parsed_sha256=p_sha,
                fingerprint=fp,
                url=f"{used_base}/api/v3/klines",
                first_ts=str(df.index[0]),
                last_ts=str(df.index[-1]),
            )
            m_file = manifest_path / f"{manifest.dataset_id}.json"
            m_file.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")

            ledger.append(
                kind="data_fetch",
                status="PASS",
                tag=f"fetch.crypto.{sym}",
                payload={
                    "dataset_id": manifest.dataset_id,
                    "symbol": sym,
                    "n_rows": len(df),
                    "first_ts": manifest.first_ts,
                    "last_ts": manifest.last_ts,
                    "devval_rows": devval_n,
                    "vault_rows": vault_n,
                },
            )
            results.append({
                "id": manifest.dataset_id,
                "symbol": sym,
                "rows": len(df),
                "first_date": str(df.index[0].date()),
                "last_date": str(df.index[-1].date()),
                "status": "PASS",
            })
            print(f"{sym:<10} | {manifest.dataset_id[:16]} | rows={len(df):<5} | {df.index[0].date()} to {df.index[-1].date()} | PASS")

        except Exception as e:
            ledger.append(
                kind="data_fetch",
                status="NOT_RUN",
                tag=f"fetch.crypto.{sym}",
                payload={"symbol": sym, "error": str(e)},
            )
            results.append({"id": "N/A", "symbol": sym, "rows": 0, "status": f"NOT_RUN({e})"})
            print(f"{sym:<10} | NOT_RUN({e})")

    # 3. Macro
    for series in MACRO:
        ts_utc = datetime.now(timezone.utc).isoformat()
        try:
            df, raw_pages, raw_bytes = fetch_fred_series(series, archive_dir=archive_dir)
            raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            p_sha = parsed_hash(df)
            diffs = df["value"].diff().dropna()
            fp = realism_fingerprint(diffs)

            # Store macro into devval
            store_devval = ParquetStore(store_devval_path)
            store_devval.write(series, df, "1d")

            manifest = DatasetManifest.create(
                source_kind="real_api",
                source_name=f"fred.{series}",
                params={"series": series},
                fetched_at_utc=ts_utc,
                n_rows=len(df),
                raw_pages=raw_pages,
                raw_sha256=raw_sha,
                parsed_sha256=p_sha,
                fingerprint=fp,
                url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
                first_ts=str(df.index[0]),
                last_ts=str(df.index[-1]),
            )
            m_file = manifest_path / f"{manifest.dataset_id}.json"
            m_file.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")

            ledger.append(
                kind="data_fetch",
                status="PASS",
                tag=f"fetch.macro.{series}",
                payload={
                    "dataset_id": manifest.dataset_id,
                    "series": series,
                    "n_rows": len(df),
                    "first_ts": manifest.first_ts,
                    "last_ts": manifest.last_ts,
                },
            )
            results.append({
                "id": manifest.dataset_id,
                "symbol": series,
                "rows": len(df),
                "first_date": str(df.index[0].date()),
                "last_date": str(df.index[-1].date()),
                "status": "PASS",
            })
            print(f"{series:<10} | {manifest.dataset_id[:16]} | rows={len(df):<5} | {df.index[0].date()} to {df.index[-1].date()} | PASS")

        except Exception as e:
            ledger.append(
                kind="data_fetch",
                status="NOT_RUN",
                tag=f"fetch.macro.{series}",
                payload={"series": series, "error": str(e)},
            )
            results.append({"id": "N/A", "symbol": series, "rows": 0, "status": f"NOT_RUN({e})"})
            print(f"{series:<10} | NOT_RUN({e})")

    return results


def spotcheck(
    manifest_dir: Path | str = "research/manifests",
    store_devval_path: Path | str = "research/store_devval",
    out_file: Path | str = "research/reports/spotcheck.txt",
) -> None:
    """Generate deterministic random spotcheck reports on ingested datasets."""
    m_dir = Path(manifest_dir).resolve()
    out_path = Path(out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = ParquetStore(store_devval_path)

    lines: list[str] = [
        "=" * 80,
        "TAPEWRIGHT REAL DATA SPOTCHECK REPORT",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "Instructions: Compare sample values by eye against public chart sources.",
        "=" * 80,
        "",
    ]

    manifest_files = sorted(m_dir.glob("*.json"))
    for mf in manifest_files:
        data = json.loads(mf.read_text(encoding="utf-8"))
        ds_id = data.get("dataset_id", "")
        source_name = data.get("source_name", "")
        raw_sha = data.get("raw_sha256", "0" * 64)
        sym = source_name.split(".")[-1]

        lines.append(f"DATASET: {source_name} (ID: {ds_id[:16]}...)")
        lines.append(f"  Rows: {data.get('n_rows')}, Range: {data.get('first_ts')} to {data.get('last_ts')}")

        try:
            df = store.read(sym, "1d")
        except Exception:
            lines.append("  (Not available in store_devval)")
            lines.append("")
            continue

        if df.empty:
            lines.append("  (Empty DataFrame in store)")
            lines.append("")
            continue

        seed_val = int(raw_sha[:8], 16)
        rng = random.Random(seed_val)
        sample_indices = rng.sample(range(len(df)), min(5, len(df)))
        sample_indices.sort()

        lines.append("  Random Sample Dates (Seed from raw_sha256):")
        for idx in sample_indices:
            row = df.iloc[idx]
            dt_str = str(df.index[idx].date())
            if "close" in df.columns:
                lines.append(
                    f"    {dt_str}: O={row.get('open', 0):.4f}, H={row.get('high', 0):.4f}, "
                    f"L={row.get('low', 0):.4f}, C={row.get('close', 0):.4f}, V={row.get('volume', 0):.0f}"
                )
            elif "value" in df.columns:
                lines.append(f"    {dt_str}: Value={row.get('value', 0):.4f}")

        # Top 3 up / down days for SPY and BTCUSDT
        if sym in ("SPY", "BTCUSDT") and "close" in df.columns:
            pct_chg = df["close"].pct_change().dropna()
            top_up = pct_chg.nlargest(3)
            top_down = pct_chg.nsmallest(3)

            lines.append(f"  Top 3 Up Days for {sym} (dev/val):")
            for dt, val in top_up.items():
                lines.append(f"    {dt.date()}: +{val:.2%}")

            lines.append(f"  Top 3 Down Days for {sym} (dev/val):")
            for dt, val in top_down.items():
                lines.append(f"    {dt.date()}: {val:.2%}")

        lines.append("")

    lines.append("=" * 80)
    report_text = "\n".join(lines) + "\n"
    out_path.write_text(report_text, encoding="utf-8")
    print(report_text)


def main() -> int:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch_all()
        return 0
    elif len(sys.argv) > 1 and sys.argv[1] == "spotcheck":
        spotcheck()
        return 0
    print("Usage: python -m research.real_data [fetch|spotcheck]")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
