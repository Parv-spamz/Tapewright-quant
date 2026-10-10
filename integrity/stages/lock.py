"""Verification stage: Test lock verification."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

NAME = "lock"


def run(args: argparse.Namespace) -> tuple[str, str]:
    lock_sha = getattr(args, "lock_sha", None)
    if not lock_sha:
        return "NOT_RUN", "No --lock-sha argument provided"

    root = Path(".").resolve()
    script = root / "scripts" / "lock_tests.py"
    res = subprocess.run(
        [sys.executable, str(script), "--verify", str(lock_sha)],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if res.returncode == 0:
        return "PASS", f"Lock matches {lock_sha[:8]}... and all files intact"
    return "FAIL", f"Lock verification failed: {res.stdout.strip() or res.stderr.strip()}"
