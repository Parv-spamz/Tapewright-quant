"""Verification stage: Pytest suite execution and failure/skip auditing."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from integrity.record_proof import parse_pytest_summary

NAME = "pytest"


def run(args: argparse.Namespace) -> tuple[str, str]:
    root = Path(".").resolve()
    quick = getattr(args, "quick", False)

    cmd = [sys.executable, "-m", "pytest"]
    if quick:
        cmd.extend(["tests/unit/test_integrity.py", "tests/unit/test_integrity_checker.py", "-q"])
    else:
        cmd.extend(["-q", "-rs"])

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    output = res.stdout + "\n" + res.stderr
    counts = parse_pytest_summary(output)

    failed = counts["failed"]
    errors = counts["errors"]
    xfailed = counts["xfailed"]
    skipped = counts["skipped"]

    if failed > 0 or errors > 0:
        return "FAIL", f"Pytest failed: {failed} failed, {errors} errors"
    if xfailed > 0:
        return "FAIL", f"Pytest contains prohibited xfail: {xfailed}"
    if skipped > 1:
        return "FAIL", f"Pytest contains unauthorized skips: {skipped} (whitelist allows max 1)"

    suite_type = "quick suite" if quick else "full suite"
    return "PASS", f"{suite_type} passed: {counts['passed']} passed, {skipped} skipped in {counts['seconds']:.2f}s"
