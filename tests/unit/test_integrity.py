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
