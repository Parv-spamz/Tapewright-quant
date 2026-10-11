"""Executable runner for real sensitivity surfaces and cost break-even analysis."""

from __future__ import annotations

import logging
import sys

from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.protocol import active_protocol
from research.sensitivity import run_sensitivity_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("research.run_sensitivity")


def main() -> None:
    ledger = Ledger("research/ledger/ledger.jsonl")
    data = ResearchData()

    pid = active_protocol(ledger)
    if not pid:
        print("ERROR: No active protocol found in ledger.", file=sys.stderr)
        sys.exit(1)

    print("=== FINPULSE-APEX SENSITIVITY ANALYSIS EXECUTION ===")
    print(f"Active Protocol ID: {pid}")

    results = run_sensitivity_study(protocol_id=pid, ledger=ledger, data=data)

    print("\n" + "=" * 110)
    print(f"{'Family':<16} | {'Plateau Score':<14} | {'0x Sharpe':<10} | {'1x Sharpe':<10} | {'2x Sharpe':<10} | {'4x Sharpe':<10} | {'Break-Even Mult':<16}")
    print("-" * 110)

    for r in results:
        fam = r.get("family", "")
        plat = float(r.get("plateau_score", 0.0))
        c_sharpes = r.get("cost_sharpes", [0.0, 0.0, 0.0, 0.0])
        s0 = f"{c_sharpes[0]:.4f}" if len(c_sharpes) > 0 else "N/A"
        s1 = f"{c_sharpes[1]:.4f}" if len(c_sharpes) > 1 else "N/A"
        s2 = f"{c_sharpes[2]:.4f}" if len(c_sharpes) > 2 else "N/A"
        s4 = f"{c_sharpes[3]:.4f}" if len(c_sharpes) > 3 else "N/A"
        be = r.get("break_even_multiplier", "NOT_FOUND")
        be_str = f"{be:.4f}" if isinstance(be, (int, float)) else str(be)

        print(f"{fam:<16} | {plat:<14.4f} | {s0:<10} | {s1:<10} | {s2:<10} | {s4:<10} | {be_str:<16}")

    print("=" * 110)


if __name__ == "__main__":
    main()
