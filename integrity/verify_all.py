"""Consolidated multi-stage verification runner for Tapewright research engine.

CLI: python -m integrity.verify_all --lock-sha <HEX> [--quick]

Dynamically discovers and executes all stage modules defined in integrity/stages/,
prints a consolidated verification table, and exits 1 if any stage reports FAIL.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from typing import Any

import integrity.stages


def discover_stages() -> list[tuple[str, Any]]:
    """Discover all available stage modules in integrity.stages via pkgutil."""
    stages = []
    package = integrity.stages
    for mod_info in pkgutil.iter_modules(package.__path__):
        if mod_info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"integrity.stages.{mod_info.name}")
        name = getattr(mod, "NAME", mod_info.name)
        stages.append((name, mod))

    # Preferred logical ordering for output clarity
    preferred_order = [
        "lock",
        "ledger_chain",
        "constraints",
        "test_integrity",
        "claims_audit",
        "docs_lint",
        "pytest",
    ]
    def sort_key(item: tuple[str, Any]) -> int:
        name = item[0]
        return preferred_order.index(name) if name in preferred_order else 99

    return sorted(stages, key=sort_key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidated integrity verification gate runner.")
    parser.add_argument("--lock-sha", type=str, default="", help="Expected tests/TEST_LOCK.json SHA-256 hex digest.")
    parser.add_argument("--quick", action="store_true", help="Execute quick test suite instead of full regression suite.")

    args = parser.parse_args()
    stages = discover_stages()

    results: list[tuple[str, str, str]] = []
    any_failed = False

    print("=" * 80)
    print("TAPEWRIGHT VERIFICATION GATE RUNNER")
    print("=" * 80)

    for stage_name, stage_mod in stages:
        if not hasattr(stage_mod, "run"):
            results.append((stage_name, "NOT_RUN", "Stage module missing run() function"))
            continue

        try:
            res = stage_mod.run(args)
            if isinstance(res, tuple) and len(res) >= 2:
                status, msg = res[0], res[1]
                if isinstance(status, bool):
                    status_str = "PASS" if status else "FAIL"
                else:
                    status_str = str(status).upper()
            else:
                status_str = "PASS" if bool(res) else "FAIL"
                msg = ""
        except Exception as e:
            status_str = "FAIL"
            msg = f"Unhandled exception in stage: {e}"

        if status_str == "FAIL":
            any_failed = True

        results.append((stage_name, status_str, str(msg)))

    # Print summary table
    print(f"{'STAGE':<20} | {'STATUS':<9} | {'DETAILS'}")
    print("-" * 20 + "-+-" + "-" * 9 + "-+-" + "-" * 47)
    for name, st, details in results:
        print(f"{name:<20} | {st:<9} | {details}")
    print("=" * 80)

    if any_failed:
        print("VERIFICATION RESULT: FAILED (at least one stage failed)")
        return 1

    print("VERIFICATION RESULT: PASSED (all stages completed successfully)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
