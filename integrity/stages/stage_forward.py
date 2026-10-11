"""Verification stage: Forward test signal-outcome invariants and time-barrier integrity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import pandas as pd

from integrity.ledger import Ledger

NAME = "forward"


def run(args: argparse.Namespace) -> tuple[str, str]:
    """Verify that all forward signals strictly precede target bar open times and match outcomes."""
    ledger_arg = getattr(args, "ledger", None)
    ledger_path = Path(ledger_arg) if ledger_arg else (Path(".").resolve() / "research" / "ledger" / "ledger.jsonl")

    if not ledger_path.exists():
        return "PASS", "Ledger does not exist yet; zero forward records to audit"

    ledger = Ledger(ledger_path)
    records = ledger.query()

    signals = [r for r in records if r.get("kind") == "forward_signal" and r.get("status") == "PASS"]
    outcomes = [r for r in records if r.get("kind") == "forward_outcome" and r.get("status") == "PASS"]

    # 1. Invariant: Every forward_signal timestamp strictly precedes target_bar_open
    for sig in signals:
        p = sig.get("payload", {})
        emitted_str = p.get("emitted_at_utc")
        target_open_str = p.get("target_bar_open")
        if not emitted_str or not target_open_str:
            return "FAIL", f"Forward signal record {sig.get('seq')} missing timestamps"

        t_emitted = pd.to_datetime(emitted_str)
        t_open = pd.to_datetime(target_open_str)
        if t_emitted >= t_open:
            return "FAIL", f"Forward signal at {t_emitted} was logged after target bar open at {t_open}"

    # 2. Invariant: Every forward_outcome has a matching earlier forward_signal
    signal_keys = {
        (
            s.get("payload", {}).get("spec_hash"),
            str(s.get("payload", {}).get("target_bar_open")),
        ): s
        for s in signals
    }

    for out in outcomes:
        p = out.get("payload", {})
        spec_hash = p.get("spec_hash")
        target_open = str(p.get("target_bar_open"))
        key = (spec_hash, target_open)

        if key not in signal_keys:
            return "FAIL", f"Forward outcome for {spec_hash[:12]} at {target_open} has no preceding forward_signal"

        sig = signal_keys[key]
        sig_seq = sig.get("seq", 0)
        out_seq = out.get("seq", 0)
        if sig_seq >= out_seq:
            return "FAIL", f"Forward outcome seq {out_seq} appeared before its signal seq {sig_seq}"

    return "PASS", f"Verified {len(signals)} forward signals and {len(outcomes)} forward outcomes; zero time-barrier violations"
