"""Tamper-evident append-only ledger with cryptographic hash chaining and file locking."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

ALLOWED_KINDS = {
    "protocol",
    "data_fetch",
    "experiment",
    "cv",
    "proof",
    "lockbox_open",
    "lockbox_eval",
    "forward_signal",
    "forward_outcome",
    "calibration",
    "audit_note",
    "data_access",
}

ALLOWED_STATUSES = {
    "PASS",
    "FAIL",
    "NOT_RUN",
    "INFO",
}

EXCLUDED_DIR_NAMES = {
    "tests",
    "logs",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
}


def _check_payload_numbers(val: Any) -> None:
    """Recursively verify that payload contains no NaN or infinite float values."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Payload contains non-compliant float value: {val}")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_payload_numbers(v)
    elif isinstance(val, (list, tuple, set)):
        for item in val:
            _check_payload_numbers(item)


def canonical_json(obj: Any) -> str:
    """Serialize dictionary/object to canonical JSON string without NaN or non-ASCII characters."""
    _check_payload_numbers(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )


def compute_record_hash(record_without_hash: dict[str, Any]) -> str:
    """Compute SHA-256 digest of canonical JSON representation of a record without 'hash'."""
    data = canonical_json(record_without_hash).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def code_hash(root: Path | str = ".") -> str:
    """Calculate SHA-256 digest over sorted relative paths and content of eligible source files.

    Excludes files inside tests/, logs/, .git/, __pycache__, and test caches.
    """
    root_path = Path(root).resolve()
    hasher = hashlib.sha256()

    eligible_files: list[tuple[str, Path]] = []
    for p in root_path.rglob("*.py"):
        if not p.is_file():
            continue
        rel = p.relative_to(root_path)
        parts = rel.parts
        if any(part in EXCLUDED_DIR_NAMES for part in parts):
            continue
        eligible_files.append((rel.as_posix(), p))

    eligible_files.sort(key=lambda x: x[0])

    for rel_posix, full_path in eligible_files:
        hasher.update(rel_posix.encode("utf-8"))
        hasher.update(full_path.read_bytes())

    return hasher.hexdigest()


class FileLock:
    """Cross-platform advisory file lock using atomic os.open with O_CREAT | O_EXCL."""

    def __init__(self, lock_path: Path, timeout: float = 10.0, retry_delay: float = 0.05) -> None:
        self.lock_path = lock_path
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.fd: int | None = None

    def __enter__(self) -> FileLock:
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                )
                return self
            except (FileExistsError, PermissionError, OSError):
                if time.time() - start_time >= self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.lock_path} within {self.timeout}s")
                time.sleep(self.retry_delay)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError as e:
                logger.warning("Error closing lock file fd: %s", e)
            self.fd = None
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except OSError as e:
            logger.warning("Error unlinking lock path: %s", e)


class Ledger:
    """Tamper-evident append-only ledger with cryptographic hash chaining."""

    def __init__(self, path: Path | str = "research/ledger/ledger.jsonl", root: Path | str = ".") -> None:
        self.path = Path(path).resolve()
        self.root = Path(root).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        kind: str,
        status: str,
        tag: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Append a new record to the ledger under an exclusive file lock."""
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"Invalid record kind '{kind}'. Must be one of {sorted(ALLOWED_KINDS)}")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid record status '{status}'. Must be one of {sorted(ALLOWED_STATUSES)}")
        if not isinstance(payload, dict):
            raise ValueError(f"Payload must be a dictionary, got {type(payload).__name__}")

        _check_payload_numbers(payload)

        with FileLock(self.lock_path):
            records = self._read_all_records()
            seq = len(records)
            prev_hash = "0" * 64 if seq == 0 else records[-1]["hash"]

            current_code_hash = code_hash(self.root)
            ts_utc = datetime.now(timezone.utc).isoformat()

            record_no_hash = {
                "code_hash": current_code_hash,
                "kind": kind,
                "payload": payload,
                "prev_hash": prev_hash,
                "seq": seq,
                "status": status,
                "tag": str(tag),
                "ts_utc": ts_utc,
            }

            rec_hash = compute_record_hash(record_no_hash)
            full_record = dict(record_no_hash)
            full_record["hash"] = rec_hash

            canonical_line = canonical_json(full_record) + "\n"

            with open(self.path, "a", encoding="utf-8") as f:
                f.write(canonical_line)
                f.flush()
                os.fsync(f.fileno())

            return full_record

    def _read_all_records(self) -> list[dict[str, Any]]:
        """Read all records currently written to the ledger file."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue
                records.append(json.loads(line_str))
        return records

    def verify_chain(self) -> tuple[bool, int | None, str]:
        """Verify contiguous sequence numbers, previous-hash links, recomputed hashes, and monotonic timestamps."""
        records = self._read_all_records()
        if not records:
            return True, None, "Ledger is empty"

        prev_hash = "0" * 64
        prev_dt: datetime | None = None

        for idx, rec in enumerate(records):
            seq = rec.get("seq")
            if seq != idx:
                return False, idx, f"Non-contiguous sequence at index {idx}: expected {idx}, got {seq}"

            rec_prev_hash = rec.get("prev_hash")
            if rec_prev_hash != prev_hash:
                return (
                    False,
                    idx,
                    f"Broken prev_hash link at seq {seq}: expected {prev_hash}, got {rec_prev_hash}",
                )

            recorded_hash = rec.get("hash")
            no_hash = {k: v for k, v in rec.items() if k != "hash"}
            expected_hash = compute_record_hash(no_hash)
            if recorded_hash != expected_hash:
                return (
                    False,
                    idx,
                    f"Tampered record hash at seq {seq}: expected {expected_hash}, got {recorded_hash}",
                )

            ts_str = rec.get("ts_utc")
            if not ts_str:
                return False, idx, f"Missing timestamp at seq {seq}"
            try:
                curr_dt = datetime.fromisoformat(ts_str)
            except Exception as e:
                return False, idx, f"Malformed timestamp at seq {seq}: {e}"

            if prev_dt is not None and curr_dt < prev_dt:
                return (
                    False,
                    idx,
                    f"Decreasing timestamp at seq {seq}: {curr_dt.isoformat()} < {prev_dt.isoformat()}",
                )

            prev_dt = curr_dt
            prev_hash = recorded_hash

        return True, None, "Chain valid"

    def head_hash(self) -> str:
        """Return hash of the latest ledger record, or 64 zeros if empty."""
        records = self._read_all_records()
        if not records:
            return "0" * 64
        return str(records[-1]["hash"])

    def query(
        self,
        kind: str | None = None,
        status: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter ledger records by kind, status, or tag."""
        records = self._read_all_records()
        matches: list[dict[str, Any]] = []
        for r in records:
            if kind is not None and r.get("kind") != kind:
                continue
            if status is not None and r.get("status") != status:
                continue
            if tag is not None and r.get("tag") != tag:
                continue
            matches.append(r)
        return matches

    def latest(self, tag: str) -> dict[str, Any] | None:
        """Return the most recent record with the given tag, or None if not found."""
        records = self._read_all_records()
        for r in reversed(records):
            if r.get("tag") == tag:
                return r
        return None
