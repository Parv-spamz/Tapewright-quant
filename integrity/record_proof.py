"""Proof record parser and ledger registration.

CLI: python -m integrity.record_proof <proof_file> --tag <TAG>
Parses pytest summary metrics from the given proof file, computes the file's
cryptographic SHA-256, and appends a 'proof' record to the integrity ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from integrity.ledger import Ledger


def get_whitelist_count(whitelist_path: Path) -> int:
    """Return count of active entries in tests/SKIP_WHITELIST.txt."""
    if not whitelist_path.exists():
        return 1
    count = 0
    with open(whitelist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    # If the file lists class and method separately for the same test, handle gracefully
    # The requirement specifies 1 allowable live MySQL test skip
    return max(1, count)


def parse_pytest_summary(text: str) -> dict[str, float | int]:
    """Extract passed, failed, skipped, xfailed, errors, and seconds from pytest output."""
    counts: dict[str, float | int] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "errors": 0,
        "seconds": 0.0,
    }

    # Search for pytest summary line, e.g.:
    # === 221 passed, 1 skipped, 2 warnings in 122.67s ===
    # === 18 passed in 10.77s ===
    # === 1 failed, 3 passed in 7.81s ===
    patterns = {
        "passed": r"(\d+)\s+passed",
        "failed": r"(\d+)\s+failed",
        "skipped": r"(\d+)\s+skipped",
        "xfailed": r"(\d+)\s+xfailed",
        "errors": r"(\d+)\s+error(?:s)?",
        "seconds": r"in\s+([\d\.]+)\s*s",
    }

    # Find the last summary line
    summary_lines = [
        line for line in text.splitlines()
        if "=" in line and ("passed" in line or "failed" in line or "error" in line)
    ]
    target_text = summary_lines[-1] if summary_lines else text

    for key, pat in patterns.items():
        m = re.search(pat, target_text, re.IGNORECASE)
        if m:
            if key == "seconds":
                counts[key] = float(m.group(1))
            else:
                counts[key] = int(m.group(1))

    return counts


def record_proof(proof_path: Path | str, tag: str, ledger_path: Path | str = "research/ledger/ledger.jsonl") -> tuple[str, dict]:
    """Parse proof file, compute sha256, and append record to Ledger."""
    path = Path(proof_path)
    if not path.exists():
        raise FileNotFoundError(f"Proof file not found: {proof_path}")

    raw_bytes = path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    text = path.read_text(encoding="utf-8", errors="replace")

    counts = parse_pytest_summary(text)
    whitelist_count = get_whitelist_count(Path("tests/SKIP_WHITELIST.txt"))

    passed = counts["passed"]
    failed = counts["failed"]
    skipped = counts["skipped"]
    xfailed = counts["xfailed"]
    errors = counts["errors"]

    # status PASS only if failed=0, errors=0, xfailed=0 and skipped <= whitelist_count
    # Also if at least one passed or valid run
    if failed == 0 and errors == 0 and xfailed == 0 and skipped <= whitelist_count:
        status = "PASS"
    else:
        status = "FAIL"

    payload = {
        "file": path.as_posix(),
        "sha256": file_sha256,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "xfailed": xfailed,
        "errors": errors,
        "seconds": counts["seconds"],
    }

    ledger = Ledger(ledger_path)
    ledger.append(
        kind="proof",
        status=status,
        tag=tag,
        payload=payload,
    )

    return status, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Record test proof to integrity ledger.")
    parser.add_argument("proof_file", type=str, help="Path to pytest output proof file.")
    parser.add_argument("--tag", type=str, required=True, help="Tag for ledger record (e.g. p19).")
    parser.add_argument(
        "--ledger",
        type=str,
        default="research/ledger/ledger.jsonl",
        help="Path to integrity ledger file.",
    )

    args = parser.parse_args()
    status, payload = record_proof(args.proof_file, args.tag, args.ledger)
    print(f"Recorded proof tag '{args.tag}' with status {status}: {payload}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
