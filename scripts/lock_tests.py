#!/usr/bin/env python3
"""Cryptographic test suite and integrity script locking mechanism.

Generates and verifies tests/TEST_LOCK.json containing SHA-256 digests
of test files, check scripts, lock_tests itself, and integrity modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_target_files(root: Path) -> list[Path]:
    """Gather all eligible files for locking."""
    target_files: list[Path] = []

    # 1. Every file under tests/ (excluding TEST_LOCK.json and __pycache__)
    tests_dir = root / "tests"
    if tests_dir.exists():
        for p in tests_dir.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.name != "TEST_LOCK.json":
                target_files.append(p)

    # 2. scripts/check_*.py
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        for p in scripts_dir.glob("check_*.py"):
            if p.is_file():
                target_files.append(p)

    # 3. scripts/lock_tests.py
    lock_script = scripts_dir / "lock_tests.py"
    if lock_script.exists():
        target_files.append(lock_script)

    # 4. Every file in integrity/ except integrity/stages/
    integrity_dir = root / "integrity"
    if integrity_dir.exists():
        for p in integrity_dir.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                rel_parts = p.relative_to(integrity_dir).parts
                if "stages" not in rel_parts:
                    target_files.append(p)

    return sorted(set(target_files), key=lambda p: p.relative_to(root).as_posix())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def compute_lock_data(root: Path) -> dict[str, str]:
    files = get_target_files(root)
    lock_data: dict[str, str] = {}
    for f in files:
        rel = f.relative_to(root).as_posix()
        lock_data[rel] = sha256_file(f)
    return lock_data


def write_lock(root: Path) -> str:
    lock_path = root / "tests" / "TEST_LOCK.json"
    lock_data = compute_lock_data(root)
    content = json.dumps(lock_data, indent=2, sort_keys=True) + "\n"
    lock_path.write_text(content, encoding="utf-8")
    lock_hash = sha256_file(lock_path)
    return lock_hash


def verify_lock(root: Path, expected_hash: str) -> bool:
    lock_path = root / "tests" / "TEST_LOCK.json"
    if not lock_path.exists():
        print("ERROR: tests/TEST_LOCK.json does not exist.")
        return False

    actual_lock_hash = sha256_file(lock_path)
    if actual_lock_hash.lower() != expected_hash.lower():
        print(f"ERROR: Lock file hash mismatch.\n  Expected: {expected_hash}\n  Actual:   {actual_lock_hash}")
        return False

    try:
        lock_data: dict[str, str] = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to parse tests/TEST_LOCK.json: {e}")
        return False

    all_matched = True
    for rel_path, expected_sha in sorted(lock_data.items()):
        fpath = root / rel_path
        if not fpath.exists():
            print(f"ERROR: Locked file missing: {rel_path}")
            all_matched = False
            continue
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            print(f"ERROR: File digest mismatch for {rel_path}:\n  Expected: {expected_sha}\n  Actual:   {actual_sha}")
            all_matched = False

    return all_matched


def extend_lock(root: Path) -> str:
    lock_path = root / "tests" / "TEST_LOCK.json"
    if not lock_path.exists():
        return write_lock(root)

    try:
        existing_data: dict[str, str] = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to read existing tests/TEST_LOCK.json: {e}")
        sys.exit(1)

    # Verify existing entries haven't changed
    for rel_path, expected_sha in existing_data.items():
        fpath = root / rel_path
        if not fpath.exists():
            print(f"ERROR: Cannot extend; locked file missing: {rel_path}")
            sys.exit(1)
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            print(f"ERROR: Cannot extend; existing file {rel_path} changed hash:\n  Expected: {expected_sha}\n  Actual:   {actual_sha}")
            sys.exit(1)

    # Add new files
    current_data = compute_lock_data(root)
    updated_data = dict(existing_data)
    for rel_path, sha in current_data.items():
        if rel_path not in updated_data:
            updated_data[rel_path] = sha

    content = json.dumps(updated_data, indent=2, sort_keys=True) + "\n"
    lock_path.write_text(content, encoding="utf-8")
    new_hash = sha256_file(lock_path)
    return new_hash


def main() -> int:
    parser = argparse.ArgumentParser(description="Cryptographic test suite lock management.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="Write tests/TEST_LOCK.json and print its SHA-256.")
    group.add_argument("--print-hash", action="store_true", help="Print the SHA-256 of tests/TEST_LOCK.json.")
    group.add_argument("--verify", metavar="HEX", type=str, help="Verify tests/TEST_LOCK.json against HEX.")
    group.add_argument("--extend", action="store_true", help="Extend lock with newly added files.")

    args = parser.parse_args()
    root = get_repo_root()
    lock_path = root / "tests" / "TEST_LOCK.json"

    if args.write:
        lock_hash = write_lock(root)
        print(f"LOCK_FILE_SHA256={lock_hash}")
        return 0

    if args.print_hash:
        if not lock_path.exists():
            print("ERROR: tests/TEST_LOCK.json does not exist.")
            return 1
        lock_hash = sha256_file(lock_path)
        print(f"LOCK_FILE_SHA256={lock_hash}")
        return 0

    if args.verify:
        ok = verify_lock(root, args.verify)
        if ok:
            print("Lock verification PASSED.")
            return 0
        print("Lock verification FAILED.")
        return 1

    if args.extend:
        new_hash = extend_lock(root)
        print(f"LOCK_FILE_SHA256={new_hash}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
