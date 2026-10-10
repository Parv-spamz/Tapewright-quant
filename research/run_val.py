"""Executable pipeline runner for real validation split evaluation."""

from __future__ import annotations

import logging
import sys
from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.protocol import active_protocol
from research.selection import evaluate_on_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("research.run_val")


def main() -> None:
    ledger = Ledger("research/ledger/ledger.jsonl")
    data = ResearchData()

    pid = active_protocol(ledger)
    if not pid:
        print("ERROR: No active protocol found in ledger.", file=sys.stderr)
        sys.exit(1)

    print(f"=== FINPULSE-APEX VALIDATION PIPELINE EXECUTION ===")
    print(f"Active Protocol ID: {pid}")

    print("\n--- Evaluating Selected Candidates on Validation Split ---")
    val_results = evaluate_on_validation(protocol_id=pid, ledger=ledger, data=data)

    print("\n" + "=" * 120)
    print(f"{'Family':<16} | {'Spec Hash':<14} | {'Val Sharpe':<10} | {'PSR (0)':<10} | {'PSR (Base)':<10} | {'Dev DSR':<10} | {'Survives':<10}")
    print("-" * 120)

    any_survives = False
    for res in val_results:
        fam = res["family"]
        sh = res["spec_hash"][:12]
        sr = float(res.get("val_net_sharpe", 0.0))
        psr0 = float(res.get("val_psr_zero", 0.0))
        psrb = float(res.get("val_psr_best_baseline", 0.0))
        dsr = float(res.get("dev_dsr", 0.0))
        surv = bool(res.get("survives", False))
        if surv:
            any_survives = True
        print(f"{fam:<16} | {sh:<14} | {sr:<10.4f} | {psr0:<10.4f} | {psrb:<10.4f} | {dsr:<10.4f} | {str(surv):<10}")

    print("=" * 120)

    if not any_survives:
        print("\nno candidate survives the protocol")
    else:
        print(f"\nCandidates surviving pre-registered protocol: {sum(1 for r in val_results if r.get('survives'))}")


if __name__ == "__main__":
    main()
