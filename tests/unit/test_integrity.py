"""Unit tests for the trust and integrity package."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import tempfile
import pytest

from integrity.ledger import Ledger, code_hash


class TestLedger:
    """Tests for the tamper-evident cryptographic ledger."""

    def test_ledger_append_200_and_verify_chain(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "test_ledger.jsonl"
        ledger = Ledger(ledger_file, root=tmp_path)

        for i in range(200):
            ledger.append(
                kind="experiment",
                status="PASS",
                tag=f"exp_{i}",
                payload={"iteration": i, "score": 0.5 + i * 0.001},
            )

        ok, bad_seq, msg = ledger.verify_chain()
        assert ok is True
        assert bad_seq is None
        assert msg == "Chain valid"
        assert len(ledger._read_all_records()) == 200
        assert ledger.head_hash() != "0" * 64

    def test_ledger_flip_one_byte_fails_verification(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "corrupt_ledger.jsonl"
        ledger = Ledger(ledger_file, root=tmp_path)

        for i in range(10):
            ledger.append(
                kind="cv",
                status="PASS",
                tag=f"fold_{i}",
                payload={"fold": i},
            )

        lines = ledger_file.read_text(encoding="utf-8").splitlines(keepends=True)
        # Corrupt line 5 by changing one byte
        corrupt_line = lines[5]
        if "cv" in corrupt_line:
            corrupted = corrupt_line.replace("cv", "xp", 1)
        else:
            corrupted = corrupt_line[:10] + "X" + corrupt_line[11:]
        lines[5] = corrupted
        ledger_file.write_text("".join(lines), encoding="utf-8")

        ok, bad_seq, msg = ledger.verify_chain()
        assert ok is False
        assert bad_seq == 5
        assert "Tampered" in msg or "prev_hash" in msg

    def test_ledger_delete_one_line_fails_verification(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "deleted_ledger.jsonl"
        ledger = Ledger(ledger_file, root=tmp_path)

        for i in range(10):
            ledger.append(
                kind="protocol",
                status="PASS",
                tag=f"step_{i}",
                payload={"step": i},
            )

        lines = ledger_file.read_text(encoding="utf-8").splitlines(keepends=True)
        # Delete line 4
        del lines[4]
        ledger_file.write_text("".join(lines), encoding="utf-8")

        ok, bad_seq, msg = ledger.verify_chain()
        assert ok is False
        assert bad_seq == 4

    def test_ledger_swap_two_lines_fails_verification(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "swapped_ledger.jsonl"
        ledger = Ledger(ledger_file, root=tmp_path)

        for i in range(10):
            ledger.append(
                kind="audit_note",
                status="INFO",
                tag=f"note_{i}",
                payload={"index": i},
            )

        lines = ledger_file.read_text(encoding="utf-8").splitlines(keepends=True)
        # Swap line 3 and line 4
        lines[3], lines[4] = lines[4], lines[3]
        ledger_file.write_text("".join(lines), encoding="utf-8")

        ok, bad_seq, msg = ledger.verify_chain()
        assert ok is False
        assert bad_seq == 3

    def test_ledger_nan_or_inf_payload_raises(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "nan_ledger.jsonl"
        ledger = Ledger(ledger_file, root=tmp_path)

        with pytest.raises(ValueError, match="non-compliant"):
            ledger.append(
                kind="experiment",
                status="FAIL",
                tag="nan_test",
                payload={"bad": float("nan")},
            )

        with pytest.raises(ValueError, match="non-compliant"):
            ledger.append(
                kind="experiment",
                status="FAIL",
                tag="inf_test",
                payload={"nested": {"bad": float("inf")}},
            )

    def test_ledger_multithreaded_concurrency(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "threaded_ledger.jsonl"
        ledger = Ledger(ledger_file, root=tmp_path)

        def worker(thread_id: int) -> None:
            for j in range(50):
                ledger.append(
                    kind="proof",
                    status="PASS",
                    tag=f"th_{thread_id}",
                    payload={"thread": thread_id, "idx": j},
                )

        threads = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(worker, t) for t in range(threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        records = ledger._read_all_records()
        assert len(records) == 200

        ok, bad_seq, msg = ledger.verify_chain()
        assert ok is True, f"Chain corrupted under concurrency: seq={bad_seq}, msg={msg}"
        assert bad_seq is None

    def test_code_hash_sensitivity_and_tests_isolation(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        tests_dir = tmp_path / "tests"
        src_dir.mkdir(parents=True)
        tests_dir.mkdir(parents=True)

        mod_a = src_dir / "module_a.py"
        mod_a.write_text("x = 1\n", encoding="utf-8")

        test_a = tests_dir / "test_a.py"
        test_a.write_text("def test_x(): pass\n", encoding="utf-8")

        h1 = code_hash(tmp_path)

        # Modifying a test file must NOT change code_hash
        test_a.write_text("def test_x(): assert 1 == 1\n", encoding="utf-8")
        h2 = code_hash(tmp_path)
        assert h1 == h2, "code_hash changed when only a test file was modified"

        # Modifying a source file MUST change code_hash
        mod_a.write_text("x = 2\n", encoding="utf-8")
        h3 = code_hash(tmp_path)
        assert h1 != h3, "code_hash failed to detect modification in source file"


class TestProvenance:
    """Tests for dataset provenance, raw page archiving, and realism fingerprinting."""

    def test_parsed_hash_deterministic(self) -> None:
        import numpy as np
        import pandas as pd
        from integrity.provenance import parsed_hash

        df1 = pd.DataFrame(
            {"price": [100.0, 101.5, 99.8], "volume": [1000, 1500, 1200]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        df2 = pd.DataFrame(
            {"price": [100.0, 101.5, 99.8], "volume": [1000, 1500, 1200]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        assert parsed_hash(df1) == parsed_hash(df2)

        # Content difference must yield different hash
        df3 = df1.copy()
        df3.iloc[0, 0] = 100.01
        assert parsed_hash(df1) != parsed_hash(df3)

    def test_archive_raw_and_load_raw_tamper_detection(self, tmp_path: Path) -> None:
        from integrity.provenance import archive_raw, load_raw, ProvenanceError

        archive_dir = tmp_path / "raw_archive"
        raw_data = b"raw_market_tick_data_page_123"

        sha = archive_raw(raw_data, archive_dir)
        loaded = load_raw(sha, archive_dir)
        assert loaded == raw_data

        # Tampering with raw file makes load_raw raise ProvenanceError
        raw_file = archive_dir / f"{sha}.bin"
        raw_file.write_bytes(b"corrupted_market_tick_data_page")

        with pytest.raises(ProvenanceError, match="corrupted or tampered"):
            load_raw(sha, archive_dir)

    def test_require_real_raises_for_synthetic_manifest(self) -> None:
        from integrity.provenance import DatasetManifest, ProvenanceError, require_real

        synthetic_manifest = DatasetManifest.create(
            source_kind="synthetic",
            source_name="monte_carlo_gbm",
            params={"n": 1000},
            fetched_at_utc="2024-01-01T00:00:00Z",
            n_rows=1000,
            raw_pages=[],
            raw_sha256="0" * 64,
            parsed_sha256="0" * 64,
            fingerprint={"n": 1000},
        )
        with pytest.raises(ProvenanceError, match="synthetic"):
            require_real(synthetic_manifest)

        real_manifest = DatasetManifest.create(
            source_kind="real_api",
            source_name="polygon_equities",
            params={"ticker": "AAPL"},
            fetched_at_utc="2024-01-01T00:00:00Z",
            n_rows=1000,
            raw_pages=[],
            raw_sha256="0" * 64,
            parsed_sha256="0" * 64,
            fingerprint={"n": 1000},
        )
        # Does not raise
        require_real(real_manifest)

    def test_looks_synthetic_gbm_true_for_gbm_and_false_for_garch(self) -> None:
        import numpy as np
        from integrity.provenance import looks_synthetic_gbm, realism_fingerprint

        # 1. Simulate GBM daily returns: n=2000, i.i.d. normal
        np.random.seed(101)
        n = 2000
        gbm_returns = np.random.normal(loc=0.0002, scale=0.01, size=n)
        fp_gbm = realism_fingerprint(gbm_returns)

        print("\nGBM Fingerprint:", fp_gbm)
        assert looks_synthetic_gbm(fp_gbm) is True, (
            f"GBM series failed looks_synthetic_gbm: {fp_gbm}"
        )

        # 2. Simulate GARCH(1,1) series with t(5) innovations (volatility clustering & heavy tails)
        np.random.seed(202)
        omega = 1e-5
        alpha = 0.15
        beta = 0.80
        z = np.random.standard_t(df=5, size=n)
        z = z / np.std(z)

        r_garch = np.zeros(n)
        sig2 = np.zeros(n)
        sig2[0] = omega / max(1e-6, 1.0 - alpha - beta)
        r_garch[0] = np.sqrt(sig2[0]) * z[0]

        for t in range(1, n):
            sig2[t] = omega + alpha * (r_garch[t - 1] ** 2) + beta * sig2[t - 1]
            r_garch[t] = np.sqrt(sig2[t]) * z[t]

        fp_garch = realism_fingerprint(r_garch)
        print("GARCH(1,1) Fingerprint:", fp_garch)
        assert looks_synthetic_gbm(fp_garch) is False, (
            f"GARCH(1,1) was falsely flagged as synthetic GBM: {fp_garch}"
        )

    def test_verify_manifest_full_cycle(self, tmp_path: Path) -> None:
        import pandas as pd
        from integrity.provenance import (
            DatasetManifest,
            archive_raw,
            parsed_hash,
            realism_fingerprint,
            verify_manifest,
            ProvenanceError,
        )

        archive_dir = tmp_path / "archive"
        raw_bytes = b"sample_raw_csv_page_bytes"
        raw_sha = archive_raw(raw_bytes, archive_dir)

        df = pd.DataFrame(
            {"returns": [0.01, -0.02, 0.015, -0.005] * 10},
            index=pd.date_range("2024-01-01", periods=40),
        )
        p_sha = parsed_hash(df)
        fp = realism_fingerprint(df["returns"])

        manifest = DatasetManifest.create(
            source_kind="real_user_file",
            source_name="user_returns_csv",
            params={"file": "test.csv"},
            fetched_at_utc="2024-01-01T12:00:00Z",
            n_rows=len(df),
            raw_pages=[raw_sha],
            raw_sha256=raw_sha,
            parsed_sha256=p_sha,
            fingerprint=fp,
        )

        # Valid manifest passes
        assert verify_manifest(manifest, df, archive_dir) is True

        # Modified DataFrame raises ProvenanceError
        df_modified = df.copy()
        df_modified.iloc[0, 0] = 0.99
        with pytest.raises(ProvenanceError, match="DataFrame hash mismatch"):
            verify_manifest(manifest, df_modified, archive_dir)
