"""Cryptographic ledger anchoring and public timestamping utility.

Honest limitations:
- The local system clock is NOT an outside authority and can be modified or manipulated.
- True cryptographic anchoring requires committing and pushing the ledger and anchor file
  to a public immutable repository (e.g. GitHub or public timestamp authority) so that
  external commit and push timestamps act as independent chronological proofs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from integrity.ledger import Ledger

logger = logging.getLogger(__name__)


def anchor_ledger(
    ledger_path: str | Path = "research/ledger/ledger.jsonl",
    anchors_path: str | Path = "research/ledger/ANCHORS.txt",
) -> dict[str, Any]:
    """Verify ledger cryptographic hash chain and append timestamped anchor record to ANCHORS.txt.

    Prints public timestamping instructions.
    """
    l_path = Path(ledger_path)
    if not l_path.exists():
        raise FileNotFoundError(f"Ledger file not found at {l_path}")

    ledger = Ledger(str(l_path))
    # 1. Verify tamper-evident cryptographic hash chain
    is_valid = ledger.verify_chain()
    if not is_valid:
        raise ValueError("Ledger hash chain verification failed! Ledger integrity is compromised.")

    records = ledger.query()
    seq = len(records)
    head = ledger.head_hash() or "0" * 64
    utc_now = datetime.now(timezone.utc).isoformat()

    anchor_line = f"{utc_now} | seq={seq} | head_hash={head}\n"

    a_path = Path(anchors_path)
    a_path.parent.mkdir(parents=True, exist_ok=True)
    with open(a_path, "a", encoding="utf-8") as f:
        f.write(anchor_line)

    payload = {
        "timestamp_utc": utc_now,
        "sequence_number": seq,
        "head_hash": head,
        "anchors_file": str(a_path),
    }

    print("\n=== LEDGER CRYPTOGRAPHIC ANCHOR CREATED ===")
    print(f"Timestamp (UTC): {utc_now}")
    print(f"Record Count:    {seq}")
    print(f"Head Hash:       {head}")
    print("\n[CRITICAL TIMESTAMPING INSTRUCTIONS]")
    print("The local system clock is not an outside authority.")
    print("Immediately commit and push research/ledger/ANCHORS.txt and ledger.jsonl")
    print("to a public remote Git repository so external git timestamps certify ordering.")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor ledger state to ANCHORS.txt.")
    parser.add_argument("--ledger", type=str, default="research/ledger/ledger.jsonl")
    parser.add_argument("--anchors", type=str, default="research/ledger/ANCHORS.txt")
    args = parser.parse_args()

    anchor_ledger(ledger_path=args.ledger, anchors_path=args.anchors)


if __name__ == "__main__":
    main()
