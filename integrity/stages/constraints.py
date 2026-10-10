"""Verification stage: Architectural constraints, syntax, and TypeScript prohibition."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

NAME = "constraints"


def run(args: argparse.Namespace) -> tuple[str, str]:
    root = Path(".").resolve()

    # 1. Syntax check
    syntax_script = root / "scripts" / "check_syntax.py"
    res_syntax = subprocess.run([sys.executable, str(syntax_script)], capture_output=True, text=True, cwd=root)
    if res_syntax.returncode != 0:
        return "FAIL", f"Syntax validation failed: {res_syntax.stdout.strip()}"

    # 2. Constraints check
    constraints_script = root / "scripts" / "check_constraints.py"
    res_constraints = subprocess.run([sys.executable, str(constraints_script)], capture_output=True, text=True, cwd=root)
    if res_constraints.returncode != 0:
        return "FAIL", f"Constraints check failed: {res_constraints.stdout.strip()}"

    # 3. TypeScript sweep
    ts_exts = (".ts", ".tsx", ".mts", ".cts")
    ts_configs = ("tsconfig.json", "package.json")
    for r, dirs, files in os.walk(root):
        if any(d in r for d in [".git", "__pycache__", ".venv", "venv", ".pytest_cache"]):
            continue
        if "node_modules" in dirs:
            return "FAIL", f"Found node_modules directory in {r}"
        for f in files:
            fl = f.lower()
            if fl.endswith(ts_exts) or fl in ts_configs or (fl.startswith("tsconfig") and fl.endswith(".json")):
                return "FAIL", f"Prohibited TypeScript/Node file found: {os.path.join(r, f)}"

    return "PASS", "Syntax, constraints, and TypeScript prohibition verified"
