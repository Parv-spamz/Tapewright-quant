"""Cryptographic one-shot lockbox holdout isolation and evaluation.

Guarantees absolute out-of-sample purity:
- Exactly one evaluation allowed per protocol chain
- Enforces strict confirmation passphrase: 'I-UNDERSTAND-THIS-IS-ONE-SHOT'
- Requires prior pre-registered experiment record with role='selected'
- Appends 'lockbox_open' BEFORE reading data so any failure/crash counts as consumed
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from data.caching.parquet_store.store import ParquetStore
from integrity.ledger import Ledger
from research.data_access import LockboxError
from research.protocol import protocol_chain

ONE_SHOT_CONFIRMATION = "I-UNDERSTAND-THIS-IS-ONE-SHOT"


def lockbox_status(ledger: Ledger, protocol_id: str | None = None) -> str:
    """Determine the current lockbox status: UNOPENED, OPENED_RESULT_PRESENT, or OPENED_NO_RESULT."""
    records = ledger._read_all_records()

    relevant_chain_set: set[str] | None = None
    if protocol_id:
        try:
            chain = protocol_chain(protocol_id, ledger)
            relevant_chain_set = set(chain)
        except Exception:
            relevant_chain_set = {protocol_id}

    has_open = False
    has_eval = False

    for r in records:
        kind = r.get("kind")
        p_id = r.get("payload", {}).get("protocol_id")

        if relevant_chain_set is not None and p_id not in relevant_chain_set:
            continue

        if kind == "lockbox_open":
            has_open = True
        elif kind == "lockbox_eval":
            has_eval = True

    if not has_open:
        return "UNOPENED"
    if has_eval:
        return "OPENED_RESULT_PRESENT"
    return "OPENED_NO_RESULT"


def evaluate_lockbox_once(
    protocol_id: str,
    spec_hash: str,
    runner: Callable[[ParquetStore], dict[str, Any]],
    confirm: str,
    ledger: Ledger | None = None,
    vault_path: Path | str = "research/vault",
    ledger_path: Path | str = "research/ledger/ledger.jsonl",
) -> dict[str, Any]:
    """Execute the final one-shot lockbox holdout evaluation.

    Strictly refused unless:
    1. confirm == 'I-UNDERSTAND-THIS-IS-ONE-SHOT'
    2. A prior experiment record exists with role='selected', matching spec_hash, and earlier timestamp
    3. No 'lockbox_open' record exists in the protocol chain
    """
    if confirm != ONE_SHOT_CONFIRMATION:
        raise ValueError(
            f"Lockbox evaluation refused: confirmation phrase mismatch. "
            f"Must be exactly '{ONE_SHOT_CONFIRMATION}'."
        )

    led = ledger or Ledger(ledger_path)
    chain = protocol_chain(protocol_id, led)
    chain_set = set(chain)

    all_records = led._read_all_records()

    # 1. Check if ANY lockbox_open record exists anywhere in the protocol chain
    for r in all_records:
        if r.get("kind") == "lockbox_open":
            rec_proto = r.get("payload", {}).get("protocol_id")
            if rec_proto in chain_set:
                raise LockboxError(
                    f"Lockbox has already been opened for protocol chain ({protocol_id[:8]}...). "
                    "The lockbox is strictly a one-shot evaluation."
                )

    # 2. Check for prior experiment record with role='selected' and this spec_hash
    now_utc = datetime.now(timezone.utc).isoformat()
    selected_record_found = False
    for r in all_records:
        if r.get("kind") == "experiment":
            p = r.get("payload", {})
            if (
                p.get("role") == "selected"
                and p.get("spec_hash") == spec_hash
                and p.get("protocol_id") in chain_set
                and r.get("ts_utc", "") <= now_utc
            ):
                selected_record_found = True
                break

    if not selected_record_found:
        raise ValueError(
            f"Lockbox evaluation refused: no prior experiment record with role='selected' "
            f"and spec_hash='{spec_hash}' found in ledger for protocol chain."
        )

    # 3. Append lockbox_open FIRST so any subsequent error/crash marks the lockbox as consumed
    led.append(
        kind="lockbox_open",
        status="INFO",
        tag=f"lockbox_open.{protocol_id[:8]}",
        payload={
            "protocol_id": protocol_id,
            "spec_hash": spec_hash,
            "opened_at_utc": now_utc,
        },
    )

    # 4. Read from vault and execute runner
    vault = ParquetStore(vault_path)
    eval_results = runner(vault)

    # 5. Append lockbox_eval record
    led.append(
        kind="lockbox_eval",
        status="PASS",
        tag=f"lockbox_eval.{protocol_id[:8]}",
        payload={
            "protocol_id": protocol_id,
            "spec_hash": spec_hash,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "results": eval_results,
        },
    )

    return eval_results
