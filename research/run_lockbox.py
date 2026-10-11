"""Thin CLI harness for final one-shot lockbox holdout evaluation.

DO NOT RUN THIS MODULE DIRECTLY DURING DEV OR VALIDATION.
Execution is restricted to the final stage with explicit human confirmation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.experiment import ExperimentSpec, run_experiment
from research.lockbox import ONE_SHOT_CONFIRMATION, evaluate_lockbox_once
from research.protocol import active_protocol, protocol_chain

logger = logging.getLogger(__name__)


def run_lockbox_cli(
    confirm: str,
    protocol_id: str | None = None,
    vault_path: Path | str = "research/vault",
    ledger_path: Path | str = "research/ledger/ledger.jsonl",
) -> list[dict[str, Any]]:
    """Execute one-shot lockbox evaluation for all surviving candidates."""
    if confirm != ONE_SHOT_CONFIRMATION:
        raise ValueError(
            f"Lockbox execution refused. Confirmation string must be exactly '{ONE_SHOT_CONFIRMATION}'."
        )

    ledger = Ledger(ledger_path)
    pid = protocol_id or active_protocol(ledger)
    if not pid:
        raise ValueError("No active protocol found in ledger.")

    chain = protocol_chain(pid, ledger)
    chain_set = set(chain)

    records = ledger._read_all_records()
    surviving_val_records = [
        r for r in records
        if r.get("kind") == "experiment"
        and str(r.get("tag", "")).startswith("val.")
        and r.get("payload", {}).get("survives") is True
        and r.get("payload", {}).get("protocol_id") in chain_set
    ]

    if not surviving_val_records:
        logger.info("No candidates survived validation to evaluate in lockbox.")
        return []

    results: list[dict[str, Any]] = []

    for val_rec in surviving_val_records:
        vp = val_rec["payload"]
        spec_hash = vp["spec_hash"]

        # Find original selected spec
        sel_rec = next(
            (
                r for r in records
                if r.get("kind") == "experiment"
                and r.get("payload", {}).get("role") == "selected"
                and r.get("payload", {}).get("spec_hash") == spec_hash
            ),
            None,
        )
        if not sel_rec:
            logger.warning("Selected record not found for surviving spec %s; skipping.", spec_hash[:12])
            continue

        sel_spec_dict = sel_rec["payload"]["spec"]
        lockbox_spec = ExperimentSpec(
            family=sel_spec_dict["family"],
            params=sel_spec_dict.get("params", {}),
            universe_name=sel_spec_dict.get("universe_name", "etfs"),
            split="lockbox",  # Evaluated strictly inside lockbox
            role="lockbox_eval",
            execution=sel_spec_dict.get("execution", "realistic"),
            sizing=sel_spec_dict.get("sizing", "half_kelly"),
            flags=sel_spec_dict.get("flags", {}),
            seeds=sel_spec_dict.get("seeds", [42]),
        )

        def runner(store: Any) -> dict[str, Any]:
            # Accesses vault data through isolated lockbox runner
            vault_data = ResearchData(store_dir=vault_path)
            return run_experiment(lockbox_spec, pid, ledger, vault_data)

        eval_res = evaluate_lockbox_once(
            protocol_id=pid,
            spec_hash=spec_hash,
            runner=runner,
            confirm=confirm,
            ledger=ledger,
            vault_path=vault_path,
            ledger_path=ledger_path,
        )
        results.append(eval_res)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot lockbox evaluation CLI.")
    parser.add_argument(
        "--confirm",
        required=True,
        help="Mandatory confirmation passphrase: 'I-UNDERSTAND-THIS-IS-ONE-SHOT'",
    )
    parser.add_argument("--protocol-id", default=None, help="Target protocol ID.")
    parser.add_argument("--vault-path", default="research/vault", help="Path to holdout vault.")
    parser.add_argument("--ledger-path", default="research/ledger/ledger.jsonl", help="Path to ledger.")
    args = parser.parse_args()

    try:
        res = run_lockbox_cli(
            confirm=args.confirm,
            protocol_id=args.protocol_id,
            vault_path=args.vault_path,
            ledger_path=args.ledger_path,
        )
        print(f"Lockbox evaluation completed for {len(res)} candidates.")
    except Exception as e:
        print(f"Lockbox evaluation failed or refused: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
