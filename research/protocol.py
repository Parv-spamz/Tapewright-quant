"""Pre-registration protocol definition and ledger registration.

Enforces experimental discipline before any backtesting or model search:
- Fixed cost models, sizing constraints, survival rules, and known biases
- Cryptographic binding to dataset manifests and split boundaries
- Multi-testing trial count preservation across superseding revisions
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

from integrity.ledger import Ledger, canonical_json
from research.splits import CRYPTO_SPLITS, EQUITY_SPLITS

COST_MODEL = {
    "equities": {
        "half_spread_bps": 1.0,
        "commission": 0.0,
        "sqrt_impact_adv_coef": 0.1,
    },
    "crypto": {
        "taker_fee_bps": 10.0,
        "half_spread_bps": 1.0,
    },
}

SIZING = {
    "method": "Half-Kelly",
    "fraction": 0.5,
    "volatility_target": 0.15,
    "cvar_99_budget": 0.03,
    "direction": "long-only except pairs and cross-sectional long-short",
}

SURVIVAL_RULE = {
    "dev_dsr_min": 0.90,
    "dev_dsr_trial_count": "full ledger trial count",
    "val_net_sharpe_min": 0.0,
    "val_psr_against_zero_min": 0.90,
}

KNOWN_BIASES = [
    "universe chosen with hindsight (ETFs that exist today)",
    "adjusted prices use future dividend information in levels",
    "daily bars only, no intraday fills for equities",
    "crypto fees and spreads are assumed, not measured, until Prompt 22",
]

STRATEGY_FAMILIES = [
    "crossover",
    "breakout",
    "mean_reversion",
    "pairs",
    "cross_sectional",
]


def compute_splits_hash() -> str:
    """Compute deterministic SHA-256 digest of defined split constants."""
    data = {
        "CRYPTO_SPLITS": CRYPTO_SPLITS,
        "EQUITY_SPLITS": EQUITY_SPLITS,
    }
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Protocol:
    """Frozen research protocol specifying universe, datasets, models, and constraints."""

    name: str
    universe: list[str]
    dataset_ids: list[str]
    splits_hash: str
    strategy_families: list[str]
    cost_model: dict[str, Any]
    sizing: dict[str, Any]
    survival_rule: dict[str, Any]
    max_trials: int
    known_biases: list[str]
    supersedes: str | None = None

    def protocol_id(self) -> str:
        """Calculate canonical SHA-256 identifier of this protocol specification."""
        d = asdict(self)
        return hashlib.sha256(canonical_json(d).encode("utf-8")).hexdigest()


def protocol_chain(protocol_id: str, ledger: Ledger) -> list[str]:
    """Retrieve full ancestor chain of protocol IDs ordered from root to current."""
    records = [
        r for r in ledger._read_all_records()
        if r.get("kind") == "protocol" and r.get("status") == "PASS"
    ]
    by_id = {r["payload"]["protocol_id"]: r["payload"]["protocol"] for r in records}

    if protocol_id not in by_id:
        raise ValueError(f"Protocol '{protocol_id}' not found in ledger")

    chain = [protocol_id]
    curr = by_id[protocol_id].get("supersedes")
    while curr:
        chain.append(curr)
        if curr in by_id:
            curr = by_id[curr].get("supersedes")
        else:
            break

    return list(reversed(chain))


def get_accumulated_trials(protocol_id: str | None, ledger: Ledger) -> int:
    """Accumulate total historical trials conducted across supersedes chain."""
    if not protocol_id:
        return 0

    chain = protocol_chain(protocol_id, ledger)
    chain_set = set(chain)

    trials = 0
    for r in ledger._read_all_records():
        if r.get("kind") in ("experiment", "cv"):
            p_id = r.get("payload", {}).get("protocol_id")
            if p_id in chain_set:
                trials += 1
    return trials


def register_protocol(
    protocol: Protocol,
    ledger: Ledger,
    manifest_dir: Path | str = "research/manifests",
) -> str:
    """Register protocol in tamper-evident ledger with integrity validations."""
    p_id = protocol.protocol_id()

    # 1. Validate max_trials
    if protocol.max_trials <= 0:
        raise ValueError(f"max_trials must be positive, got {protocol.max_trials}")

    # 2. Validate splits_hash
    expected_splits_hash = compute_splits_hash()
    if protocol.splits_hash != expected_splits_hash:
        raise ValueError(
            f"splits_hash mismatch: expected {expected_splits_hash}, got {protocol.splits_hash}"
        )

    # 3. Validate dataset manifests and require real sources
    m_dir = Path(manifest_dir).resolve()
    for ds_id in protocol.dataset_ids:
        m_file = m_dir / f"{ds_id}.json"
        if not m_file.is_file():
            raise ValueError(f"Manifest not found for dataset_id: {ds_id}")
        m_data = json.loads(m_file.read_text(encoding="utf-8"))
        if not m_data.get("source_kind", "").startswith("real"):
            raise ValueError(
                f"Synthetic dataset '{ds_id}' rejected for research protocol (source_kind='{m_data.get('source_kind')}')"
            )

    # 4. Check ledger records
    existing_records = [
        r for r in ledger._read_all_records()
        if r.get("kind") == "protocol" and r.get("status") == "PASS"
    ]

    # Idempotent check
    for r in existing_records:
        if r["payload"]["protocol_id"] == p_id:
            return p_id

    # Check for modification without supersedes
    for r in existing_records:
        prev_p = r["payload"]["protocol"]
        if prev_p["name"] == protocol.name and protocol.supersedes is None:
            raise ValueError(
                f"Protocol '{protocol.name}' already exists with different parameters. "
                "Modified protocol must set 'supersedes' to preserve multi-testing trial history."
            )

    if protocol.supersedes:
        chain_ids = [r["payload"]["protocol_id"] for r in existing_records]
        if protocol.supersedes not in chain_ids:
            raise ValueError(f"Superseded protocol '{protocol.supersedes}' does not exist in ledger")

    prior_trials = get_accumulated_trials(protocol.supersedes, ledger)

    payload = {
        "protocol_id": p_id,
        "protocol": asdict(protocol),
        "prior_trials": prior_trials,
    }
    ledger.append(
        kind="protocol",
        status="PASS",
        tag=f"protocol.{protocol.name}",
        payload=payload,
    )
    return p_id


def build_default_protocol(manifest_dir: Path | str = "research/manifests") -> tuple[Protocol, list[str]]:
    """Build default research protocol from available real manifests."""
    m_dir = Path(manifest_dir).resolve()
    manifest_files = sorted(m_dir.glob("*.json"))

    passed_ids: list[str] = []
    available_symbols: set[str] = set()

    for mf in manifest_files:
        try:
            m_data = json.loads(mf.read_text(encoding="utf-8"))
            if m_data.get("source_kind", "").startswith("real"):
                passed_ids.append(m_data["dataset_id"])
                src_name = m_data.get("source_name", "")
                sym = src_name.split(".")[-1]
                available_symbols.add(sym)
        except Exception as e:
            logging.warning("Failed to parse manifest %s: %s", mf.name, e)
            continue

    all_expected = [
        "SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "GLD", "XLE", "XLF", "XLK", "XLV", "XLY",
        "BTCUSDT", "ETHUSDT"
    ]
    missing_symbols = [s for s in all_expected if s not in available_symbols]
    active_universe = [s for s in all_expected if s in available_symbols]

    proto = Protocol(
        name="tapewright_core_v1",
        universe=active_universe,
        dataset_ids=passed_ids,
        splits_hash=compute_splits_hash(),
        strategy_families=STRATEGY_FAMILIES,
        cost_model=COST_MODEL,
        sizing=SIZING,
        survival_rule=SURVIVAL_RULE,
        max_trials=100,
        known_biases=KNOWN_BIASES,
        supersedes=None,
    )
    return proto, missing_symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Research protocol registration.")
    parser.add_argument("action", choices=["register", "show"], help="Action to perform")
    parser.add_argument("--manifest-dir", default="research/manifests", help="Path to manifests directory")
    parser.add_argument("--ledger", default="research/ledger/ledger.jsonl", help="Path to ledger file")

    args = parser.parse_args()
    ledger = Ledger(args.ledger)

    if args.action == "register":
        proto, missing = build_default_protocol(args.manifest_dir)
        p_id = register_protocol(proto, ledger, manifest_dir=args.manifest_dir)
        print(f"PROTOCOL_ID={p_id}")
        print(f"Active Universe ({len(proto.universe)} assets): {proto.universe}")
        print(f"Registered Manifests: {len(proto.dataset_ids)}")
        if missing:
            print(f"Missing/NOT_RUN Universes: {missing}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
