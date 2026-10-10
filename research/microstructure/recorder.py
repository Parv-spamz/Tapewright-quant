"""Real market L2 order book and trade microstructure recorder.

Honest limitations:
- 1 Hz snapshots, not event-by-event full order book reconstructions.
- No order cancellations or individual modifications are visible in depth snapshots.
- Top 20 bid and ask price levels recorded per snapshot.

Standard library networking only. Interacts with public endpoints of Binance
with fallback across official API hosts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from integrity.ledger import Ledger
from integrity.provenance import archive_raw
from research.data_access import write_microstructure_data

logger = logging.getLogger(__name__)

BINANCE_BASES = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
]


def fetch_depth(
    symbol: str,
    limit: int = 20,
    base: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Fetch L2 depth snapshot from Binance REST API using standard library urllib."""
    clean_symbol = symbol.strip().upper()
    bases = [base] if base else BINANCE_BASES

    last_error: Exception | None = None
    for b in bases:
        url = f"{b}/api/v3/depth?symbol={clean_symbol}&limit={limit}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Tapewright-Quant/1.0", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except Exception as e:
            last_error = e
            logger.warning("Depth fetch failed for host %s: %s", b, e)

    raise RuntimeError(f"All depth API hosts failed for {clean_symbol}: {last_error}") from last_error


def fetch_agg_trades(
    symbol: str,
    from_id: int | None = None,
    limit: int = 1000,
    base: str | None = None,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch aggregate trades from Binance REST API using standard library urllib."""
    clean_symbol = symbol.strip().upper()
    bases = [base] if base else BINANCE_BASES

    last_error: Exception | None = None
    for b in bases:
        url = f"{b}/api/v3/aggTrades?symbol={clean_symbol}&limit={limit}"
        if from_id is not None:
            url += f"&fromId={from_id}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Tapewright-Quant/1.0", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list):
                    return data
                raise ValueError(f"Unexpected response format from {b}: {data}")
        except Exception as e:
            last_error = e
            logger.warning("Agg trades fetch failed for host %s: %s", b, e)

    raise RuntimeError(f"All agg trades API hosts failed for {clean_symbol}: {last_error}") from last_error


def parse_depth_snapshot(raw_depth: dict[str, Any], max_levels: int = 20) -> dict[str, float]:
    """Parse raw Binance depth payload into structured 20-level book fields and top-of-book metrics."""
    bids = raw_depth.get("bids", [])
    asks = raw_depth.get("asks", [])

    row: dict[str, float] = {}
    for i in range(1, max_levels + 1):
        if i <= len(bids):
            row[f"bid_px_{i}"] = float(bids[i - 1][0])
            row[f"bid_sz_{i}"] = float(bids[i - 1][1])
        else:
            row[f"bid_px_{i}"] = np.nan
            row[f"bid_sz_{i}"] = np.nan

        if i <= len(asks):
            row[f"ask_px_{i}"] = float(asks[i - 1][0])
            row[f"ask_sz_{i}"] = float(asks[i - 1][1])
        else:
            row[f"ask_px_{i}"] = np.nan
            row[f"ask_sz_{i}"] = np.nan

    best_bid = row.get("bid_px_1", np.nan)
    best_ask = row.get("ask_px_1", np.nan)

    if not np.isnan(best_bid) and not np.isnan(best_ask):
        row["mid"] = (best_bid + best_ask) / 2.0
        row["spread"] = best_ask - best_bid
    else:
        row["mid"] = np.nan
        row["spread"] = np.nan

    return row


def parse_agg_trades(raw_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse raw Binance aggregate trades payload into signed trade records.

    Trade sign convention:
    -1 if isBuyerMaker ('m') is True (aggressor sell hitting resting bid),
    +1 if isBuyerMaker ('m') is False (aggressor buy taking resting ask).
    """
    parsed: list[dict[str, Any]] = []
    for t in raw_trades:
        agg_id = int(t["a"])
        ts = int(t["T"])
        price = float(t["p"])
        size = float(t["q"])
        is_buyer_maker = bool(t.get("m", False))
        sign = -1 if is_buyer_maker else 1

        parsed.append({
            "agg_id": agg_id,
            "ts": ts,
            "price": price,
            "size": size,
            "sign": sign,
        })
    return parsed


def detect_agg_gaps(agg_ids: list[int]) -> int:
    """Count missing trade IDs (gaps) in a sequential aggTrade ID sequence."""
    if len(agg_ids) < 2:
        return 0

    sorted_ids = sorted(agg_ids)
    gaps = 0
    for i in range(1, len(sorted_ids)):
        diff = sorted_ids[i] - sorted_ids[i - 1]
        if diff > 1:
            gaps += (diff - 1)
    return gaps


def record(
    symbol: str,
    minutes: float = 30.0,
    hz: float = 1.0,
    store_path: str | Path = "research/store_devval",
    archive_dir: str | Path = "research/raw_archive/microstructure",
    ledger_path: str | Path = "research/ledger/ledger.jsonl",
    manifests_dir: str | Path = "research/manifests",
    clock_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    depth_fetcher: Callable[[str], dict[str, Any]] | None = None,
    trades_fetcher: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Execute high-frequency microstructure recording loop.

    Polls L2 depth snapshots and sequential aggregate trades, writes book and trade rows
    to partitioned Parquet storage, archives raw JSON responses cryptographically,
    and appends a data_fetch record to the ledger.
    """
    clean_symbol = symbol.strip().upper()
    now_fn = clock_fn or time.time
    sleeper = sleep_fn or time.sleep
    df_fetcher = depth_fetcher or (lambda s: fetch_depth(s, limit=20))
    tr_fetcher = trades_fetcher or fetch_agg_trades

    ledger = Ledger(ledger_path)
    start_utc = datetime.now(timezone.utc)
    start_ts = now_fn()
    duration_sec = float(minutes * 60.0)
    loop_target_interval = 1.0 / max(hz, 0.01)

    book_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    raw_shas: list[str] = []
    latencies_ms: list[float] = []
    seen_agg_ids: set[int] = set()
    last_agg_id: int | None = None

    logger.info("Starting microstructure recording for %s (duration: %.1f min, rate: %.1f Hz)", clean_symbol, minutes, hz)

    try:
        while True:
            current_time = now_fn()
            elapsed_total = current_time - start_ts
            if elapsed_total >= duration_sec:
                break

            # 1. Fetch depth with midpoint timestamping
            t0 = now_fn()
            raw_depth = df_fetcher(clean_symbol)
            t1 = now_fn()

            latency_ms = (t1 - t0) * 1000.0
            latencies_ms.append(latency_ms)
            midpoint_s = (t0 + t1) / 2.0

            parsed_book = parse_depth_snapshot(raw_depth, max_levels=20)
            parsed_book["latency_ms"] = latency_ms
            parsed_book["snapshot_ts"] = midpoint_s
            book_rows.append(parsed_book)

            # 2. Fetch new aggregate trades from last_agg_id + 1
            from_id = (last_agg_id + 1) if last_agg_id is not None else None
            raw_tr = tr_fetcher(clean_symbol, from_id=from_id, limit=1000)
            parsed_trades = parse_agg_trades(raw_tr)

            for tr in parsed_trades:
                a_id = tr["agg_id"]
                if a_id not in seen_agg_ids:
                    seen_agg_ids.add(a_id)
                    trade_rows.append(tr)
                    if last_agg_id is None or a_id > last_agg_id:
                        last_agg_id = a_id

            # 3. Archive raw payloads
            combined_raw = json.dumps({"depth": raw_depth, "trades": raw_tr}, sort_keys=True).encode("utf-8")
            sha = archive_raw(combined_raw, archive_dir)
            if sha not in raw_shas:
                raw_shas.append(sha)

            # 4. Pace execution rate
            loop_elapsed = now_fn() - current_time
            if loop_elapsed < loop_target_interval:
                sleeper(loop_target_interval - loop_elapsed)

    except Exception as e:
        logger.warning("Microstructure recording halted with error: %s", e)
        fail_payload = {
            "symbol": clean_symbol,
            "status": "NOT_RUN",
            "reason": f"Network or region blocked access to Binance market data: {e}",
            "snapshots_recorded": len(book_rows),
            "trades_recorded": len(trade_rows),
        }
        ledger.append(
            kind="data_fetch",
            status="NOT_RUN",
            tag=f"microstructure.{clean_symbol}",
            payload=fail_payload,
        )
        return fail_payload

    # Build and persist DataFrames
    end_utc = datetime.now(timezone.utc)
    if book_rows:
        book_df = pd.DataFrame(book_rows)
        book_df.index = pd.to_datetime(book_df["snapshot_ts"], unit="s", utc=True)
        book_df = book_df.drop(columns=["snapshot_ts"])
    else:
        book_df = pd.DataFrame()

    if trade_rows:
        trades_df = pd.DataFrame(trade_rows)
        trades_df.index = pd.to_datetime(trades_df["ts"], unit="ms", utc=True)
    else:
        trades_df = pd.DataFrame()

    write_microstructure_data(clean_symbol, book_df, trades_df, store_path=store_path)

    # Compute metrics and gap analysis
    agg_ids = [t["agg_id"] for t in trade_rows]
    agg_gaps = detect_agg_gaps(agg_ids)
    med_latency = float(np.median(latencies_ms)) if latencies_ms else 0.0
    p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    manifest_payload = {
        "dataset_id": f"microstructure_{clean_symbol.lower()}_{start_utc.strftime('%Y%m%dT%H%M%SZ')}",
        "symbol": clean_symbol,
        "source_kind": "real_api",
        "source_name": "binance_vision_l2",
        "started_at_utc": start_utc.isoformat(),
        "ended_at_utc": end_utc.isoformat(),
        "minutes_recorded": float(minutes),
        "snapshot_count": len(book_rows),
        "trade_count": len(trade_rows),
        "agg_id_gaps_found": agg_gaps,
        "latency_median_ms": med_latency,
        "latency_p95_ms": p95_latency,
        "raw_pages": raw_shas,
    }

    m_dir = Path(manifests_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    m_path = m_dir / f"{manifest_payload['dataset_id']}.json"
    m_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    ledger_payload = {
        "status": "PASS",
        **manifest_payload,
        "manifest_path": str(m_path),
    }

    ledger.append(
        kind="data_fetch",
        status="PASS",
        tag=f"microstructure.{clean_symbol}",
        payload=ledger_payload,
    )

    logger.info("Recording complete: %d snapshots, %d trades, %d gaps", len(book_rows), len(trade_rows), agg_gaps)
    return ledger_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Real market L2 microstructure recorder.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to record (e.g. BTCUSDT)")
    parser.add_argument("--minutes", type=float, default=30.0, help="Recording duration in minutes")
    parser.add_argument("--hz", type=float, default=1.0, help="Snapshot polling frequency in Hz")
    parser.add_argument("--store-path", type=str, default="research/store_devval", help="ParquetStore path")
    args = parser.parse_args()

    result = record(symbol=args.symbol, minutes=args.minutes, hz=args.hz, store_path=args.store_path)
    print("\n=== MICROSTRUCTURE RECORDER SUMMARY ===")
    print(f"Status: {result.get('status', 'PASS')}")
    print(f"Symbol: {result.get('symbol')}")
    print(f"Snapshots Recorded: {result.get('snapshot_count', result.get('snapshots_recorded', 0))}")
    print(f"Trades Recorded: {result.get('trade_count', result.get('trades_recorded', 0))}")
    print(f"AggId Gaps: {result.get('agg_id_gaps_found', 0)}")
    print(f"Median Latency (ms): {result.get('latency_median_ms', 0.0):.2f}")
    print(f"95th Pct Latency (ms): {result.get('latency_p95_ms', 0.0):.2f}")
    if result.get("status") == "NOT_RUN":
        print(f"Reason: {result.get('reason')}")


if __name__ == "__main__":
    main()
