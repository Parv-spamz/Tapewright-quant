"""Unit tests for research protocol pre-registration, data access layer, and lockbox isolation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
import pandas as pd

from integrity.ledger import Ledger
from research.data_access import LockboxError, ResearchData
from research.lockbox import evaluate_lockbox_once, lockbox_status
from research.protocol import (
    COST_MODEL,
    KNOWN_BIASES,
    SIZING,
    STRATEGY_FAMILIES,
    SURVIVAL_RULE,
    Protocol,
    compute_splits_hash,
    protocol_chain,
    register_protocol,
)


@pytest.fixture
def mock_env(tmp_path):
    """Setup isolated test environment with manifests, ledger, and parquet stores."""
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = tmp_path / "ledger" / "ledger.jsonl"
    ledger = Ledger(ledger_path, root=tmp_path)

    store_devval = tmp_path / "store_devval"
    vault = tmp_path / "vault"
    raw_archive = tmp_path / "raw_archive"
    raw_archive.mkdir(parents=True, exist_ok=True)

    # Create real dataset manifest
    real_manifest_id = "real_ds_001"
    m_real = {
        "dataset_id": real_manifest_id,
        "source_kind": "real_api",
        "source_name": "yahoo.SPY",
        "raw_pages": [],
        "raw_sha256": "0" * 64,
        "parsed_sha256": "0" * 64,
        "n_rows": 100,
        "first_ts": "2020-01-01",
        "last_ts": "2023-12-31",
    }
    (manifest_dir / f"{real_manifest_id}.json").write_text(json.dumps(m_real), encoding="utf-8")

    # Create synthetic dataset manifest
    synthetic_manifest_id = "synth_ds_001"
    m_synth = {
        "dataset_id": synthetic_manifest_id,
        "source_kind": "synthetic",
        "source_name": "gbm.SPY",
        "raw_pages": [],
        "raw_sha256": "1" * 64,
        "parsed_sha256": "1" * 64,
        "n_rows": 100,
    }
    (manifest_dir / f"{synthetic_manifest_id}.json").write_text(json.dumps(m_synth), encoding="utf-8")

    return {
        "manifest_dir": manifest_dir,
        "ledger": ledger,
        "store_devval": store_devval,
        "vault": vault,
        "real_ds_id": real_manifest_id,
        "synth_ds_id": synthetic_manifest_id,
    }


def _make_proto(
    real_ds_id: str,
    name: str = "proto_v1",
    max_trials: int = 50,
    supersedes: str | None = None,
    splits_hash: str | None = None,
) -> Protocol:
    return Protocol(
        name=name,
        universe=["SPY"],
        dataset_ids=[real_ds_id],
        splits_hash=splits_hash or compute_splits_hash(),
        strategy_families=STRATEGY_FAMILIES,
        cost_model=COST_MODEL,
        sizing=SIZING,
        survival_rule=SURVIVAL_RULE,
        max_trials=max_trials,
        known_biases=KNOWN_BIASES,
        supersedes=supersedes,
    )


def test_protocol_registration_idempotent(mock_env):
    """Verify that registering the exact same protocol twice returns the same ID."""
    proto = _make_proto(mock_env["real_ds_id"])
    p_id_1 = register_protocol(proto, mock_env["ledger"], manifest_dir=mock_env["manifest_dir"])
    p_id_2 = register_protocol(proto, mock_env["ledger"], manifest_dir=mock_env["manifest_dir"])

    assert p_id_1 == p_id_2
    records = [r for r in mock_env["ledger"]._read_all_records() if r.get("kind") == "protocol"]
    assert len(records) == 1


def test_modified_protocol_requires_supersedes_and_accumulates_trials(mock_env):
    """Verify modifying protocol requires supersedes and accumulates trial count."""
    ledger = mock_env["ledger"]
    proto1 = _make_proto(mock_env["real_ds_id"], max_trials=50)
    p_id_1 = register_protocol(proto1, ledger, manifest_dir=mock_env["manifest_dir"])

    # Conduct 3 experiments under proto1
    for i in range(3):
        ledger.append(
            kind="experiment",
            status="PASS",
            tag=f"exp.{i}",
            payload={"protocol_id": p_id_1, "trial_idx": i},
        )

    # Modification without supersedes is refused
    proto2_invalid = _make_proto(mock_env["real_ds_id"], max_trials=100, supersedes=None)
    with pytest.raises(ValueError) as exc:
        register_protocol(proto2_invalid, ledger, manifest_dir=mock_env["manifest_dir"])
    assert "Modified protocol must set 'supersedes'" in str(exc.value)

    # Modification with supersedes succeeds and accumulates prior trials
    proto2_valid = _make_proto(mock_env["real_ds_id"], max_trials=100, supersedes=p_id_1)
    p_id_2 = register_protocol(proto2_valid, ledger, manifest_dir=mock_env["manifest_dir"])

    records = [
        r for r in ledger._read_all_records()
        if r.get("kind") == "protocol" and r["payload"]["protocol_id"] == p_id_2
    ]
    assert len(records) == 1
    assert records[0]["payload"]["prior_trials"] == 3

    chain = protocol_chain(p_id_2, ledger)
    assert chain == [p_id_1, p_id_2]


def test_splits_hash_mismatch_refused(mock_env):
    """Verify protocol with mismatched splits_hash is rejected."""
    proto = _make_proto(mock_env["real_ds_id"], splits_hash="0" * 64)
    with pytest.raises(ValueError) as exc:
        register_protocol(proto, mock_env["ledger"], manifest_dir=mock_env["manifest_dir"])
    assert "splits_hash mismatch" in str(exc.value)


def test_synthetic_dataset_rejected(mock_env):
    """Verify protocol referencing synthetic dataset is rejected."""
    proto = Protocol(
        name="proto_synth",
        universe=["SPY"],
        dataset_ids=[mock_env["synth_ds_id"]],
        splits_hash=compute_splits_hash(),
        strategy_families=STRATEGY_FAMILIES,
        cost_model=COST_MODEL,
        sizing=SIZING,
        survival_rule=SURVIVAL_RULE,
        max_trials=50,
        known_biases=KNOWN_BIASES,
    )
    with pytest.raises(ValueError) as exc:
        register_protocol(proto, mock_env["ledger"], manifest_dir=mock_env["manifest_dir"])
    assert "Synthetic dataset" in str(exc.value)


def test_research_data_access_and_logging(mock_env):
    """Verify ResearchData gates access and logs val split access."""
    proto = _make_proto(mock_env["real_ds_id"])
    p_id = register_protocol(proto, mock_env["ledger"], manifest_dir=mock_env["manifest_dir"])

    rdata = ResearchData(
        p_id,
        ledger=mock_env["ledger"],
        store_path=mock_env["store_devval"],
    )

    # Accessing lockbox raises LockboxError
    with pytest.raises(LockboxError) as exc:
        rdata.get("SPY", "lockbox")
    assert "Unauthorized access to lockbox split" in str(exc.value)

    # Accessing dev returns frame and does not log data_access
    df_dev = rdata.get("SPY", "dev")
    dev_records = [r for r in mock_env["ledger"]._read_all_records() if r.get("kind") == "data_access"]
    assert len(dev_records) == 0

    # Accessing val increments count and logs data_access INFO record
    df_val = rdata.get("SPY", "val")
    val_records = [r for r in mock_env["ledger"]._read_all_records() if r.get("kind") == "data_access"]
    assert len(val_records) == 1
    assert rdata.access_counts() == 1


def test_lockbox_evaluation_invariants(mock_env):
    """Verify lockbox one-shot confirmation, pre-requisites, and crash handling."""
    ledger = mock_env["ledger"]
    proto = _make_proto(mock_env["real_ds_id"])
    p_id = register_protocol(proto, ledger, manifest_dir=mock_env["manifest_dir"])
    spec_hash = "cand_spec_abc123"

    def dummy_runner(store):
        return {"test_metric": 1.25}

    # 1. Refuse with wrong confirm text
    with pytest.raises(ValueError) as exc:
        evaluate_lockbox_once(p_id, spec_hash, dummy_runner, confirm="WRONG-TEXT", ledger=ledger)
    assert "confirmation phrase mismatch" in str(exc.value)

    # 2. Refuse without prior selected experiment record
    with pytest.raises(ValueError) as exc:
        evaluate_lockbox_once(
            p_id,
            spec_hash,
            dummy_runner,
            confirm="I-UNDERSTAND-THIS-IS-ONE-SHOT",
            ledger=ledger,
        )
    assert "no prior experiment record with role='selected'" in str(exc.value)

    # Add selected experiment record
    ledger.append(
        kind="experiment",
        status="PASS",
        tag="candidate.selected",
        payload={"protocol_id": p_id, "spec_hash": spec_hash, "role": "selected"},
    )

    # Status before open
    assert lockbox_status(ledger, p_id) == "UNOPENED"

    # 3. Successful one-shot run
    res = evaluate_lockbox_once(
        p_id,
        spec_hash,
        dummy_runner,
        confirm="I-UNDERSTAND-THIS-IS-ONE-SHOT",
        ledger=ledger,
    )
    assert res["test_metric"] == 1.25
    assert lockbox_status(ledger, p_id) == "OPENED_RESULT_PRESENT"

    # 4. Refuse second run
    with pytest.raises(LockboxError) as exc:
        evaluate_lockbox_once(
            p_id,
            spec_hash,
            dummy_runner,
            confirm="I-UNDERSTAND-THIS-IS-ONE-SHOT",
            ledger=ledger,
        )
    assert "Lockbox has already been opened" in str(exc.value)


def test_lockbox_crash_blocks_further_attempt(mock_env):
    """Verify simulated crash during runner marks lockbox consumed and blocks retry."""
    ledger = mock_env["ledger"]
    proto = _make_proto(mock_env["real_ds_id"], name="proto_crash_test")
    p_id = register_protocol(proto, ledger, manifest_dir=mock_env["manifest_dir"])
    spec_hash = "crash_spec_999"

    ledger.append(
        kind="experiment",
        status="PASS",
        tag="candidate.selected_crash",
        payload={"protocol_id": p_id, "spec_hash": spec_hash, "role": "selected"},
    )

    def crashing_runner(store):
        raise RuntimeError("Simulated crash during lockbox evaluation")

    # Run should fail with RuntimeError
    with pytest.raises(RuntimeError):
        evaluate_lockbox_once(
            p_id,
            spec_hash,
            crashing_runner,
            confirm="I-UNDERSTAND-THIS-IS-ONE-SHOT",
            ledger=ledger,
        )

    # Status must now be OPENED_NO_RESULT
    assert lockbox_status(ledger, p_id) == "OPENED_NO_RESULT"

    # Second attempt must be refused by LockboxError before runner is ever invoked
    with pytest.raises(LockboxError):
        evaluate_lockbox_once(
            p_id,
            spec_hash,
            crashing_runner,
            confirm="I-UNDERSTAND-THIS-IS-ONE-SHOT",
            ledger=ledger,
        )
