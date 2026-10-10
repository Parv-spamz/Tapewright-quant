"""Verification stage: Test integrity rules R1 through R7."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

NAME = "test_integrity"


def run(args: argparse.Namespace) -> tuple[str, str]:
    root = Path(".").resolve()
    script = root / "scripts" / "check_test_integrity.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=root)
    if res.returncode != 0:
        return "FAIL", f"Test integrity check failed: {res.stdout.strip() or res.stderr.strip()}"
    return "PASS", "Rules R1-R7 verified with 0 violations"
