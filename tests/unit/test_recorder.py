"""Unit tests for real L2 order book and trade microstructure recorder.

Enforces:
- Parser correctness on hand-crafted payloads
- Trade sign convention (+1 for buyer taker, -1 for buyer maker)
- Sequence gap detection in aggregate trades
- Pagination from_id logic with mock fetcher
- Deterministic 5-tick simulation producing 5 book rows and zero duplicate trades
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from integrity.ledger import Ledger
from research.microstructure.recorder import (
    detect_agg_gaps,
    parse_agg_trades,
    parse_depth_snapshot,
    record,
)


def test_parse_depth_snapshot_hand_payload() -> None:
    """parse_depth_snapshot extracts 20 levels, mid price, and spread with precision."""
    hand_payload = {
        "lastUpdateId": 1234567,
        "bids": [
            ["50000.00", "1.500"],
            ["49995.00", "2.100"],
            ["49990.00", "0.800"],
        ],
        "asks": [
            ["50010.00", "0.500"],
            ["50015.00", "1.200"],
            ["50020.00", "3.000"],
        ],
    }

    parsed = parse_depth_snapshot(hand_payload, max_levels=20)

    assert parsed["bid_px_1"] == 50000.0
    assert parsed["bid_sz_1"] == 1.5
    assert parsed["bid_px_2"] == 49995.0
    assert parsed["bid_sz_2"] == 2.1
    assert parsed["bid_px_3"] == 49990.0
    assert parsed["bid_sz_3"] == 0.8

    assert parsed["ask_px_1"] == 50010.0
    assert parsed["ask_sz_1"] == 0.5
    assert parsed["ask_px_2"] == 50015.0
    assert parsed["ask_sz_2"] == 1.2

    # Midpoint and spread
    assert parsed["mid"] == 50005.0
    assert parsed["spread"] == 10.0

    # Levels beyond available bids/asks must be NaN
    assert np.isnan(parsed["bid_px_4"])
    assert np.isnan(parsed["bid_sz_4"])
    assert np.isnan(parsed["bid_px_20"])
    assert np.isnan(parsed["ask_px_20"])


def test_trade_sign_convention_hand_payload() -> None:
    """Trade sign is strictly -1 when isBuyerMaker is True, and +1 when False."""
    hand_trades = [
        {"a": 1001, "p": "50000.00", "q": "0.10", "T": 1600000000000, "m": True},
        {"a": 1002, "p": "50010.00", "q": "0.25", "T": 1600000001000, "m": False},
        {"a": 1003, "p": "50005.00", "q": "1.00", "T": 1600000002000, "m": True},
    ]

    parsed = parse_agg_trades(hand_trades)

    assert len(parsed) == 3
    assert parsed[0]["sign"] == -1
    assert parsed[0]["agg_id"] == 1001
    assert parsed[1]["sign"] == 1
    assert parsed[1]["agg_id"] == 1002
    assert parsed[2]["sign"] == -1
    assert parsed[2]["agg_id"] == 1003


def test_gap_detection_on_hand_trade_list() -> None:
    """detect_agg_gaps accurately detects missing aggregate trade IDs."""
    # Sequence with no gaps
    seq_continuous = [500, 501, 502, 503, 504]
    assert detect_agg_gaps(seq_continuous) == 0

    # Sequence with missing IDs: 502 missing (1), 505 and 506 missing (2) -> total 3
    seq_with_gaps = [500, 501, 503, 504, 507]
    assert detect_agg_gaps(seq_with_gaps) == 3

    # Empty and single-element sequences have zero gaps
    assert detect_agg_gaps([]) == 0
    assert detect_agg_gaps([42]) == 0


def test_pagination_from_id_logic_with_fake_fetcher() -> None:
    """Recorder passes from_id = last_agg_id + 1 to fetch subsequent aggregate trades."""
    call_log: list[dict[str, Any]] = []

    def fake_trades_fetcher(symbol: str, from_id: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        call_log.append({"symbol": symbol, "from_id": from_id, "limit": limit})
        if from_id is None:
            return [{"a": 100, "p": "50000.0", "q": "0.1", "T": 1000, "m": False}]
        if from_id == 101:
            return [{"a": 101, "p": "50001.0", "q": "0.2", "T": 2000, "m": True}]
        return []

    # First fetch without from_id
    res1 = fake_trades_fetcher("BTCUSDT", from_id=None)
    parsed1 = parse_agg_trades(res1)
    last_id = parsed1[-1]["agg_id"]
    assert last_id == 100

    # Second fetch requesting from last_id + 1
    res2 = fake_trades_fetcher("BTCUSDT", from_id=last_id + 1)
    parsed2 = parse_agg_trades(res2)
    assert len(parsed2) == 1
    assert parsed2[0]["agg_id"] == 101

    assert len(call_log) == 2
    assert call_log[0]["from_id"] is None
    assert call_log[1]["from_id"] == 101


def test_recorder_simulated_5_ticks_produces_5_rows_no_duplicate_trades(tmp_path: Path) -> None:
    """A 5-tick simulated run against synthetic fetchers produces exactly 5 book rows and zero duplicate trades."""
    current_sim_time = 1000.0
    tick_counter = 0

    def fake_clock() -> float:
        return current_sim_time

    def fake_sleep(duration: float) -> None:
        nonlocal current_sim_time, tick_counter
        current_sim_time += duration
        tick_counter += 1

    def fake_depth_fetcher(symbol: str) -> dict[str, Any]:
        return {
            "lastUpdateId": 1000 + tick_counter,
            "bids": [[f"{50000.0 - tick_counter:.2f}", "1.0"]],
            "asks": [[f"{50010.0 + tick_counter:.2f}", "1.0"]],
        }

    # Trade generator providing overlapping or sequential trades
    def fake_trades_fetcher(symbol: str, from_id: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        base_id = 5000 if from_id is None else from_id
        return [
            {"a": base_id, "p": "50005.0", "q": "0.05", "T": int(current_sim_time * 1000), "m": False},
            {"a": base_id + 1, "p": "50006.0", "q": "0.08", "T": int(current_sim_time * 1000 + 50), "m": True},
        ]

    store_dir = tmp_path / "store"
    archive_dir = tmp_path / "archive"
    manifests_dir = tmp_path / "manifests"
    ledger_path = tmp_path / "ledger.jsonl"

    result = record(
        symbol="BTCUSDT",
        minutes=5.0 / 60.0,  # 5 seconds duration
        hz=1.0,
        store_path=store_dir,
        archive_dir=archive_dir,
        ledger_path=ledger_path,
        manifests_dir=manifests_dir,
        clock_fn=fake_clock,
        sleep_fn=fake_sleep,
        depth_fetcher=fake_depth_fetcher,
        trades_fetcher=fake_trades_fetcher,
    )

    assert result["status"] == "PASS"
    assert result["snapshot_count"] == 5
    # Each tick emits 2 distinct trade IDs: total 10 unique trades
    assert result["trade_count"] == 10
    assert result["agg_id_gaps_found"] == 0

    # Verify ledger record
    ledger = Ledger(ledger_path)
    records = ledger.query()
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "data_fetch"
    assert rec["status"] == "PASS"
    assert rec["tag"] == "microstructure.BTCUSDT"
    assert rec["payload"]["snapshot_count"] == 5

    # Verify manifest file exists on disk
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["snapshot_count"] == 5
    assert manifest_data["trade_count"] == 10
