"""Executable runner for empirical historical stress scenario calibration."""

from __future__ import annotations

import logging
import sys

from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.protocol import active_protocol
from research.real_stress import run_real_stress_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("research.run_stress")


def main() -> None:
    ledger = Ledger("research/ledger/ledger.jsonl")
    data = ResearchData()

    pid = active_protocol(ledger)
    if not pid:
        print("ERROR: No active protocol found in ledger.", file=sys.stderr)
        sys.exit(1)

    print("=== FINPULSE-APEX EMPIRICAL STRESS SCENARIO ANALYSIS ===")
    print(f"Active Protocol ID: {pid}")

    reports = run_real_stress_study(protocol_id=pid, ledger=ledger, data=data)

    print("\n" + "=" * 115)
    print(f"{'Scenario Name':<28} | {'Window Dates':<25} | {'Equity Shock':<14} | {'Vol Multiplier':<15} | {'Mean Corr':<10}")
    print("-" * 115)

    for r in reports:
        sc = r["calibrated_parameters"]
        name = sc.get("name", "")
        dates = f"{sc.get('start_date')} to {sc.get('end_date')}"
        eq_s = f"{sc.get('equity_shock'):.4f}"
        vol_m = f"{sc.get('vol_multiplier'):.4f}"
        corr = f"{sc.get('correlation_override'):.4f}"
        print(f"{name:<28} | {dates:<25} | {eq_s:<14} | {vol_m:<15} | {corr:<10}")

    print("=" * 115)


if __name__ == "__main__":
    main()
