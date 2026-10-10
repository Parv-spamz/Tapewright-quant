"""Executable pipeline runner for real dev split backtesting and selection."""

from __future__ import annotations

import logging
import sys
from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.experiment import trial_count
from research.grid import run_all_dev
from research.protocol import active_protocol
from research.selection import dev_dsr, dev_pbo, select_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("research.run_dev")


def main() -> None:
    ledger = Ledger("research/ledger/ledger.jsonl")
    data = ResearchData()

    pid = active_protocol(ledger)
    if not pid:
        print("ERROR: No active protocol found in ledger.", file=sys.stderr)
        sys.exit(1)

    print(f"=== FINPULSE-APEX DEV PIPELINE EXECUTION ===")
    print(f"Active Protocol ID: {pid}")

    # 1. Run all dev grids and baselines
    print("\n--- Running all strategy grids and baselines on DEV ---")
    dev_results = run_all_dev(protocol_id=pid, ledger=ledger, data=data)
    total_trials = trial_count(pid, ledger)
    print(f"Total dev trials recorded: {total_trials}")

    # 2. Compute PBO
    print("\n--- Computing Probability of Backtest Overfitting (PBO) ---")
    pbo = dev_pbo(pid, ledger)
    print(f"Dev PBO: {pbo:.4f}")

    # 3. Select candidates
    print("\n--- Selecting Strategy Candidates (>=30 trades, MDD >= -40%, Max Sharpe) ---")
    selected = select_candidates(pid, ledger)
    print(f"Candidates selected: {len(selected)}")

    # 4. Print Summary Table
    print("\n" + "=" * 105)
    print(f"{'Family':<16} | {'Spec Hash':<14} | {'Net Sharpe':<10} | {'Dev DSR':<10} | {'Trades':<8} | {'Max DD':<10} | {'Ann Ret':<10}")
    print("-" * 105)

    for cand in selected:
        fam = cand["family"]
        sh = cand["spec_hash"][:12]
        just = cand["justification"]
        sr = float(just.get("dev_net_sharpe", 0.0))
        tc = int(just.get("trade_count", 0))
        mdd = float(just.get("max_drawdown", 0.0))
        ann_r = float(just.get("annualized_return", 0.0))

        try:
            dsr_val = dev_dsr(cand["run_id"], ledger)
        except Exception as e:
            dsr_val = 0.0

        print(f"{fam:<16} | {sh:<14} | {sr:<10.4f} | {dsr_val:<10.4f} | {tc:<8} | {mdd:<10.4f} | {ann_r:<10.4f}")

    print("=" * 105)


if __name__ == "__main__":
    main()
