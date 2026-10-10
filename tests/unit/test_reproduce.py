"""Unit tests for offline data rebuild, experiment reproduction, and documentation integrity."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from data.caching.parquet_store.store import ParquetStore
from integrity.ledger import Ledger
from integrity.provenance import DatasetManifest, archive_raw, parsed_hash, realism_fingerprint
from integrity.render import RenderError, lint_docs, render_template_string
from research.rebuild import RebuildError, rebuild_all_stores, verify_rebuild_from_archive
from research.reproduce import reproduce_all


def test_rebuild_fails_on_archive_blob_corruption(tmp_path: Path) -> None:
    """Modifying 1 byte in an archived blob file causes rebuild to fail with hash mismatch."""
    archive_dir = tmp_path / "raw_archive"
    manifest_dir = tmp_path / "manifests"
    archive_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    # 1. Create valid synthetic Yahoo JSON payload
    chart_payload = {
        "chart": {
            "result": [
                {
                    "meta": {"gmtoffset": 0},
                    "timestamp": [1609459200, 1609545600, 1609632000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0, 102.0],
                                "high": [102.0, 103.0, 104.0],
                                "low": [99.0, 100.0, 101.0],
                                "close": [101.0, 102.0, 103.0],
                                "volume": [1000, 1100, 1200],
                            }
                        ],
                        "adjclose": [{"adjclose": [101.0, 102.0, 103.0]}],
                    },
                }
            ],
            "error": None,
        }
    }
    raw_bytes = json.dumps(chart_payload).encode("utf-8")
    raw_sha = archive_raw(raw_bytes, archive_dir)

    from research.real_data import parse_yahoo_json
    df = parse_yahoo_json(raw_bytes)
    p_sha = parsed_hash(df)
    fp = {"n": len(df)}

    manifest = DatasetManifest.create(
        source_kind="real_api",
        source_name="yahoo.SPY",
        params={"symbol": "SPY"},
        fetched_at_utc="2024-01-01T00:00:00Z",
        n_rows=len(df),
        raw_pages=[raw_sha],
        raw_sha256=raw_sha,
        parsed_sha256=p_sha,
        fingerprint=fp,
    )
    (manifest_dir / f"{manifest.dataset_id}.json").write_text(
        json.dumps(manifest.__dict__, indent=2), encoding="utf-8"
    )

    # Verification before corruption succeeds
    status, msg = verify_rebuild_from_archive(manifest_dir, archive_dir)
    assert status == "PASS"

    # 2. Corrupt exactly 1 byte in the archived blob file
    blob_file = archive_dir / raw_sha
    if not blob_file.exists():
        blob_file = archive_dir / f"{raw_sha}.bin"
    blob_bytes = bytearray(blob_file.read_bytes())
    blob_bytes[10] ^= 0xFF  # Flip 1 byte
    blob_file.write_bytes(bytes(blob_bytes))

    # Rebuild must raise error or fail verification
    with pytest.raises((RebuildError, Exception)):
        rebuild_all_stores(manifest_dir, archive_dir, write_stores=False)

    status_corrupt, msg_corrupt = verify_rebuild_from_archive(manifest_dir, archive_dir)
    assert status_corrupt == "FAIL"
    assert "failed" in msg_corrupt.lower() or "mismatch" in msg_corrupt.lower()


def test_render_fails_on_missing_tag(tmp_path: Path) -> None:
    """Document rendering raises RenderError when template contains unresolvable tag."""
    ledger_file = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_file, root=tmp_path)
    ledger.append(
        kind="experiment",
        status="PASS",
        tag="existing.tag",
        payload={"sharpe": 1.50},
    )

    bad_template = "# Strategy Report\nSharpe: {{tag:nonexistent.tag:sharpe|.2f}}"
    with pytest.raises(RenderError, match="Tag 'nonexistent.tag' not found"):
        render_template_string(bad_template, ledger)


def test_lint_docs_passes_rendered_and_fails_hand_edited(tmp_path: Path) -> None:
    """Docs linter passes properly generated blocks and catches hand-typed metric numbers outside generated blocks."""
    # 1. Properly rendered document wrapped in GENERATED block passes
    rendered_file = tmp_path / "FINDINGS.md"
    rendered_file.write_text(
        "<!-- GENERATED:BEGIN -->\n"
        "## Performance\n"
        "The model attained a Sharpe of 1.45 and Sortino of 2.10 across 55 trials.\n"
        "<!-- GENERATED:END -->\n",
        encoding="utf-8",
    )
    violations = lint_docs([rendered_file])
    assert len(violations) == 0

    # 2. Hand-edited document with number near metric word outside GENERATED block fails
    hand_edited_file = tmp_path / "README.md"
    hand_edited_file.write_text(
        "# Overview\n"
        "Our momentum strategy delivered an annualized return of 18.5% with max drawdown of 12.0%.\n",
        encoding="utf-8",
    )
    violations_hand = lint_docs([hand_edited_file])
    assert len(violations_hand) >= 1
    assert any("Hardcoded metric" in v for v in violations_hand)


def test_reproduce_synthetic_project_matches(tmp_path: Path) -> None:
    """Offline reproduction harness re-runs experiments and reproduces recorded metrics within 1e-9."""
    ledger_path = Path("research/ledger/ledger.jsonl")
    assert ledger_path.exists(), "Active project ledger does not exist"

    success, reports = reproduce_all(
        ledger_path=ledger_path,
        manifest_dir="research/manifests",
        archive_dir="research/raw_archive",
        store_devval_path="research/store_devval",
        vault_path="research/vault",
    )
    assert success is True, "Offline reproduction failed: metrics diverged"
    assert len(reports) == 3, f"Expected 3 reports, got {len(reports)}"
    for r in reports:
        assert r["ok"] is True, f"Experiment {r['role']} ({r['family']}) failed reproduction"
        for m, diff in r["diffs"].items():
            assert diff < 1e-9, f"Metric {m} diverged with diff {diff}"
