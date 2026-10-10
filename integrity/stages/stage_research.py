"""Verification stage: Research manifests, protocol pre-registration, and lockbox isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from integrity.ledger import Ledger
from integrity.provenance import load_raw
from research.lockbox import lockbox_status
from research.protocol import compute_splits_hash

NAME = "research"


def run(args: argparse.Namespace) -> tuple[str, str]:
    root = Path(".").resolve()
    manifest_dir = root / "research" / "manifests"
    archive_dir = root / "research" / "raw_archive"
    ledger_path = root / "research" / "ledger" / "ledger.jsonl"

    if not manifest_dir.exists():
        return "NOT_RUN", "Manifests directory does not exist yet"

    manifest_files = sorted(manifest_dir.glob("*.json"))
    if not manifest_files:
        return "NOT_RUN", "No manifests exist yet in research/manifests"

    # (a) Verify every manifest's raw pages exist and re-hash correctly
    verified_manifests = 0
    for mf in manifest_files:
        try:
            m_data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as e:
            return "FAIL", f"Failed to parse manifest {mf.name}: {e}"

        raw_pages = m_data.get("raw_pages", [])
        for page_sha in raw_pages:
            try:
                load_raw(page_sha, archive_dir)
            except Exception as e:
                return "FAIL", f"Raw page {page_sha[:16]} validation failed in manifest {mf.name}: {e}"
        verified_manifests += 1

    # (b) Lockbox verification: status is not OPENED_NO_RESULT and at most one lockbox_open exists
    ledger = Ledger(ledger_path)
    records = ledger._read_all_records()

    st = lockbox_status(ledger)
    if st == "OPENED_NO_RESULT":
        return "FAIL", "Lockbox status is OPENED_NO_RESULT (evaluation crashed or incomplete)"

    open_records = [r for r in records if r.get("kind") == "lockbox_open"]
    if len(open_records) > 1:
        return "FAIL", f"Multiple lockbox_open records found ({len(open_records)}), violating one-shot isolation"

    # (c) Protocol verification: splits_hash matches and no synthetic datasets are used
    protocols = [r for r in records if r.get("kind") == "protocol" and r.get("status") == "PASS"]
    expected_splits_hash = compute_splits_hash()

    for p_rec in protocols:
        proto_data = p_rec.get("payload", {}).get("protocol", {})
        splits_h = proto_data.get("splits_hash")
        if splits_h != expected_splits_hash:
            return "FAIL", f"Protocol splits_hash mismatch: expected {expected_splits_hash}, got {splits_h}"

        ds_ids = proto_data.get("dataset_ids", [])
        for ds_id in ds_ids:
            mf = manifest_dir / f"{ds_id}.json"
            if not mf.is_file():
                return "FAIL", f"Protocol references missing dataset manifest: {ds_id}"
            m_info = json.loads(mf.read_text(encoding="utf-8"))
            if not m_info.get("source_kind", "").startswith("real"):
                return "FAIL", f"Protocol references synthetic dataset: {ds_id}"

    return "PASS", f"Verified {verified_manifests} manifests, lockbox status ({st}), and {len(protocols)} protocol(s)"
