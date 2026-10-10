"""Verification stage: Ledger cryptographic chain integrity."""

from __future__ import annotations

import argparse
from pathlib import Path

from integrity.ledger import Ledger

NAME = "ledger_chain"


def run(args: argparse.Namespace) -> tuple[str, str]:
    ledger_path = Path("research/ledger/ledger.jsonl")
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        return "PASS", "Ledger is empty / initialized"

    ledger = Ledger(ledger_path)
    ok, bad_seq, msg = ledger.verify_chain()
    if ok:
        return "PASS", f"Cryptographic hash chain valid ({msg})"
    return "FAIL", f"Chain broken at seq {bad_seq}: {msg}"
