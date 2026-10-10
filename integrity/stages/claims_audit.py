"""Verification stage: Claims audit verification."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

NAME = "claims_audit"


def run(args: argparse.Namespace) -> tuple[str, str]:
    root = Path(".").resolve()
    script_path = root / "scripts" / "claims_audit.py"
    if not script_path.exists():
        return "NOT_RUN", "scripts/claims_audit.py not found"

    # Execute claims_audit.py to refresh report
    res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, cwd=root)
    if res.returncode != 0:
        return "FAIL", f"claims_audit.py execution failed: {res.stderr}"

    report_path = root / "research" / "reports" / "claims_audit.txt"
    if not report_path.exists():
        return "FAIL", "claims_audit report not generated"

    lines = report_path.read_text(encoding="utf-8").splitlines()
    missing_count = sum(1 for line in lines if "MISSING" in line)
    untested_count = sum(1 for line in lines if "UNTESTED" in line)

    if missing_count > 0 or untested_count > 0:
        return "FAIL", f"Claims audit findings: {missing_count} MISSING, {untested_count} UNTESTED"

    return "PASS", "All claims verified with tests"
