"""Executable runner for real ablation decomposition study."""

from __future__ import annotations

import logging
import sys

from integrity.ledger import Ledger
from research.ablation import run_ablation_study
from research.data_access import ResearchData
from research.protocol import active_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("research.run_ablation")


def main() -> None:
    ledger = Ledger("research/ledger/ledger.jsonl")
    data = ResearchData()

    pid = active_protocol(ledger)
    if not pid:
        print("ERROR: No active protocol found in ledger.", file=sys.stderr)
        sys.exit(1)

    print("=== FINPULSE-APEX ABLATION STUDY EXECUTION ===")
    print(f"Active Protocol ID: {pid}")

    results = run_ablation_study(protocol_id=pid, ledger=ledger, data=data)

    print("\n" + "=" * 105)
    print(f"{'Variant':<16} | {'Split':<6} | {'Status':<10} | {'Sharpe':<10} | {'Paired 95% CI':<24} | {'Outcome':<24}")
    print("-" * 105)

    for r in results:
        v = r.get("variant", "")
        spl = r.get("split", "")
        status = r.get("status", "")
        sr = float(r.get("sharpe", 0.0))
        ci = r.get("paired_ci", [0.0, 0.0])
        label = r.get("label", r.get("reason", ""))
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if status == "PASS" else "N/A"
        sr_str = f"{sr:.4f}" if status == "PASS" else "N/A"
        print(f"{v:<16} | {spl:<6} | {status:<10} | {sr_str:<10} | {ci_str:<24} | {label:<24}")

    print("=" * 105)


if __name__ == "__main__":
    main()
