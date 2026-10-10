"""Verification stage: Ledger cryptographic chain integrity."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from integrity.ledger import Ledger, compute_record_hash

NAME = "ledger_chain"


def _fix_numeric_keys(obj: Any) -> Any:
    """Restore integer dictionary keys if string keys are numeric digits."""
    if isinstance(obj, dict):
        return {
            int(k) if isinstance(k, str) and k.isdigit() else k: _fix_numeric_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_fix_numeric_keys(item) for item in obj]
    return obj


def run(args: argparse.Namespace) -> tuple[str, str]:
    ledger_path = Path("research/ledger/ledger.jsonl")
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        return "PASS", "Ledger is empty / initialized"

    ledger = Ledger(ledger_path)
    records = ledger._read_all_records()
    if not records:
        return "PASS", "Ledger is empty"

    prev_hash = "0" * 64
    prev_dt: datetime | None = None

    for idx, rec in enumerate(records):
        seq = rec.get("seq")
        if seq != idx:
            return "FAIL", f"Non-contiguous sequence at index {idx}: expected {idx}, got {seq}"

        rec_prev_hash = rec.get("prev_hash")
        if rec_prev_hash != prev_hash:
            return "FAIL", f"Broken prev_hash link at seq {seq}: expected {prev_hash}, got {rec_prev_hash}"

        recorded_hash = rec.get("hash")
        no_hash = {k: v for k, v in rec.items() if k != "hash"}
        expected_hash = compute_record_hash(no_hash)
        if recorded_hash != expected_hash:
            # Check if payload dict originally contained numeric keys
            fixed_no_hash = _fix_numeric_keys(no_hash)
            fixed_expected = compute_record_hash(fixed_no_hash)
            if recorded_hash != fixed_expected:
                return "FAIL", f"Tampered record hash at seq {seq}: expected {expected_hash}, got {recorded_hash}"

        ts_str = rec.get("ts_utc")
        if not ts_str:
            return "FAIL", f"Missing timestamp at seq {seq}"
        try:
            curr_dt = datetime.fromisoformat(ts_str)
        except Exception as e:
            return "FAIL", f"Malformed timestamp at seq {seq}: {e}"

        if prev_dt is not None and curr_dt < prev_dt:
            return "FAIL", f"Decreasing timestamp at seq {seq}: {curr_dt.isoformat()} < {prev_dt.isoformat()}"

        prev_dt = curr_dt
        prev_hash = recorded_hash

    return "PASS", f"Cryptographic hash chain valid ({len(records)} records verified)"
