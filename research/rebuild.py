"""Offline store rebuild and raw archive hash verification without network."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from integrity.provenance import load_raw, parsed_hash
from research.data_access import write_1d_dataset
from research.real_data import parse_binance_klines, parse_fred_csv, parse_yahoo_json, store_dataset_splits

logger = logging.getLogger("research.rebuild")


class RebuildError(Exception):
    """Raised when offline data rebuild fails verification or integrity checks."""


def rebuild_dataset(
    manifest_data: dict[str, Any],
    archive_dir: Path | str = "research/raw_archive",
    store_devval_path: Path | str = "research/store_devval",
    vault_path: Path | str = "research/vault",
    write_stores: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Rebuild a single dataset DataFrame offline from raw archive pages and verify hash."""
    arch_path = Path(archive_dir).resolve()
    sname = manifest_data.get("source_name", "")
    raw_pages = manifest_data.get("raw_pages", [])
    expected_parsed_sha = manifest_data.get("parsed_sha256")

    if not raw_pages:
        raise RebuildError(f"Dataset manifest '{sname}' contains no raw_pages")

    # 1. Parse raw pages according to source kind
    if sname.startswith("yahoo.") or "yahoo" in sname:
        page_bytes = load_raw(raw_pages[0], arch_path)
        df = parse_yahoo_json(page_bytes)
        is_crypto = False
        is_macro = False
        sym = sname.split(".")[-1]
    elif sname.startswith("binance.") or "binance" in sname:
        all_frames = []
        for p_sha in raw_pages:
            p_bytes = load_raw(p_sha, arch_path)
            frame = parse_binance_klines(p_bytes)
            if not frame.empty:
                all_frames.append(frame)
        if not all_frames:
            raise RebuildError(f"Binance archive returned no frames for {sname}")
        df = pd.concat(all_frames, axis=0)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()
        is_crypto = True
        is_macro = False
        sym = sname.split(".")[-1]
    elif sname.startswith("fred.") or "fred" in sname:
        page_bytes = load_raw(raw_pages[0], arch_path)
        df = parse_fred_csv(page_bytes)
        is_crypto = False
        is_macro = True
        sym = sname.split(".")[-1]
    else:
        raise RebuildError(f"Unknown source_name format for rebuild: '{sname}'")

    # 2. Verify parsed SHA-256 matches manifest parsed_sha256
    computed_sha = parsed_hash(df)
    if expected_parsed_sha and computed_sha != expected_parsed_sha:
        raise RebuildError(
            f"Parsed hash mismatch for '{sname}': expected {expected_parsed_sha}, computed {computed_sha}"
        )

    # 3. Optionally write to partition stores
    if write_stores:
        if is_macro:
            write_1d_dataset(sym, df, store_devval_path)
        else:
            store_dataset_splits(
                symbol=sym,
                df=df,
                is_crypto=is_crypto,
                store_devval_path=store_devval_path,
                vault_path=vault_path,
            )

    return df, computed_sha


def rebuild_all_stores(
    manifest_dir: Path | str = "research/manifests",
    archive_dir: Path | str = "research/raw_archive",
    store_devval_path: Path | str = "research/store_devval",
    vault_path: Path | str = "research/vault",
    write_stores: bool = True,
) -> dict[str, Any]:
    """Rebuild all datasets registered with parsed_sha256 from raw archive without network."""
    m_dir = Path(manifest_dir).resolve()
    arch_dir = Path(archive_dir).resolve()

    if not m_dir.exists():
        raise RebuildError(f"Manifests directory not found: {m_dir}")
    if not arch_dir.exists():
        raise RebuildError(f"Raw archive directory not found: {arch_dir}")

    manifest_files = sorted(m_dir.glob("*.json"))
    rebuilt_count = 0
    results: list[dict[str, Any]] = []

    for mf in manifest_files:
        try:
            m_data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as e:
            raise RebuildError(f"Failed to read manifest {mf.name}: {e}") from e

        # Skip manifests without parsed_sha256 (e.g. microstructure recording manifest)
        if "parsed_sha256" not in m_data:
            continue

        sname = m_data.get("source_name", mf.stem)
        df, p_sha = rebuild_dataset(
            manifest_data=m_data,
            archive_dir=arch_dir,
            store_devval_path=store_devval_path,
            vault_path=vault_path,
            write_stores=write_stores,
        )
        rebuilt_count += 1
        results.append({
            "source_name": sname,
            "rows": len(df),
            "parsed_sha256": p_sha,
            "verified": True,
        })
        logger.info("Rebuilt %s: %d rows (SHA: %s...)", sname, len(df), p_sha[:16])

    return {
        "status": "PASS",
        "total_rebuilt": rebuilt_count,
        "datasets": results,
    }


def verify_rebuild_from_archive(
    manifest_dir: Path | str = "research/manifests",
    archive_dir: Path | str = "research/raw_archive",
) -> tuple[str, str]:
    """Verification stage runner: rebuilds in-memory without overwriting stores and verifies all hashes."""
    try:
        res = rebuild_all_stores(
            manifest_dir=manifest_dir,
            archive_dir=archive_dir,
            write_stores=False,
        )
        n = res["total_rebuilt"]
        return "PASS", f"Successfully rebuilt and verified {n} datasets from raw archive"
    except Exception as e:
        return "FAIL", f"Archive rebuild verification failed: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild offline stores from raw archive and manifests")
    parser.add_argument("--verify-only", action="store_true", help="Verify without writing to disk stores")
    parser.add_argument("--manifest-dir", default="research/manifests", help="Path to manifests directory")
    parser.add_argument("--archive-dir", default="research/raw_archive", help="Path to raw archive directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        res = rebuild_all_stores(
            manifest_dir=args.manifest_dir,
            archive_dir=args.archive_dir,
            write_stores=not args.verify_only,
        )
        print(f"Rebuild completed successfully: {res['total_rebuilt']} datasets rebuilt and verified.")
        return 0
    except Exception as e:
        print(f"Rebuild failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
