"""Dataset provenance, content-addressed archiving, and realism fingerprinting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox


class ProvenanceError(Exception):
    """Raised when dataset provenance, content-addressing, or integrity checks fail."""


ALLOWED_SOURCE_KINDS = {
    "real_api",
    "real_user_file",
    "synthetic",
}


def _canonical_manifest_dict(d: dict[str, Any]) -> str:
    """Produce deterministic canonical JSON string for manifest hashing."""
    return json.dumps(
        d,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class DatasetManifest:
    """Cryptographic manifest documenting origin, hashes, and realism metrics of a dataset."""

    dataset_id: str
    source_kind: str
    source_name: str
    url: str | None
    params: dict[str, Any]
    fetched_at_utc: str
    n_rows: int
    first_ts: str | None
    last_ts: str | None
    raw_pages: list[str]
    raw_sha256: str
    parsed_sha256: str
    fingerprint: dict[str, Any]
    generator: str | None

    @classmethod
    def create(
        cls,
        source_kind: str,
        source_name: str,
        params: dict[str, Any],
        fetched_at_utc: str,
        n_rows: int,
        raw_pages: list[str],
        raw_sha256: str,
        parsed_sha256: str,
        fingerprint: dict[str, Any],
        url: str | None = None,
        first_ts: str | None = None,
        last_ts: str | None = None,
        generator: str | None = None,
    ) -> DatasetManifest:
        """Construct DatasetManifest and deterministically compute its dataset_id."""
        if source_kind not in ALLOWED_SOURCE_KINDS:
            raise ValueError(f"Invalid source_kind '{source_kind}'. Must be in {sorted(ALLOWED_SOURCE_KINDS)}")

        content_dict = {
            "fingerprint": fingerprint,
            "first_ts": first_ts,
            "generator": generator,
            "last_ts": last_ts,
            "n_rows": n_rows,
            "params": params,
            "parsed_sha256": parsed_sha256,
            "raw_pages": sorted(raw_pages),
            "raw_sha256": raw_sha256,
            "source_kind": source_kind,
            "source_name": source_name,
            "url": url,
            "fetched_at_utc": fetched_at_utc,
        }

        canonical_repr = _canonical_manifest_dict(content_dict)
        dataset_id = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()

        return cls(
            dataset_id=dataset_id,
            source_kind=source_kind,
            source_name=source_name,
            url=url,
            params=params,
            fetched_at_utc=fetched_at_utc,
            n_rows=n_rows,
            first_ts=first_ts,
            last_ts=last_ts,
            raw_pages=sorted(raw_pages),
            raw_sha256=raw_sha256,
            parsed_sha256=parsed_sha256,
            fingerprint=fingerprint,
            generator=generator,
        )


def archive_raw(data: bytes, archive_dir: Path | str) -> str:
    """Store raw data blob under content-addressed name <sha256>.bin."""
    sha = hashlib.sha256(data).hexdigest()
    target_dir = Path(archive_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{sha}.bin"

    if not target_file.exists():
        target_file.write_bytes(data)

    return sha


def load_raw(sha: str, archive_dir: Path | str) -> bytes:
    """Load content-addressed raw blob and verify its cryptographic digest on every read."""
    target_file = Path(archive_dir).resolve() / f"{sha}.bin"
    if not target_file.is_file():
        raise ProvenanceError(f"Archived raw blob not found: {sha}")

    data = target_file.read_bytes()
    recomputed = hashlib.sha256(data).hexdigest()
    if recomputed != sha:
        raise ProvenanceError(f"Archived raw blob corrupted or tampered: expected {sha}, got {recomputed}")

    return data


def parsed_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 hash across DataFrame schema, index, and data values."""
    hasher = hashlib.sha256()

    # Hash column names
    col_names = [str(c) for c in df.columns]
    hasher.update(",".join(col_names).encode("utf-8"))

    # Hash dtypes
    dtype_strs = [str(dt) for dt in df.dtypes]
    hasher.update(",".join(dtype_strs).encode("utf-8"))

    # Hash index and row contents
    hashed_series = pd.util.hash_pandas_object(df, index=True)
    hasher.update(hashed_series.values.tobytes())

    return hasher.hexdigest()


def verify_manifest(manifest: DatasetManifest, df: pd.DataFrame, archive_dir: Path | str) -> bool:
    """Verify that every raw page exists and is untampered, and parsed DataFrame hash matches."""
    # 1. Verify all raw pages
    for page_sha in manifest.raw_pages:
        try:
            load_raw(page_sha, archive_dir)
        except ProvenanceError as e:
            raise ProvenanceError(f"Manifest raw page validation failed: {e}") from e

    # 2. Verify parsed DataFrame hash
    current_parsed_sha = parsed_hash(df)
    if current_parsed_sha != manifest.parsed_sha256:
        raise ProvenanceError(
            f"Parsed DataFrame hash mismatch: expected {manifest.parsed_sha256}, got {current_parsed_sha}"
        )

    return True


def require_real(manifest: DatasetManifest) -> None:
    """Enforce that dataset originates from real market data or real user files."""
    if not manifest.source_kind.startswith("real"):
        raise ProvenanceError(
            f"Dataset '{manifest.dataset_id}' is synthetic (source_kind='{manifest.source_kind}'). Real data required."
        )


def realism_fingerprint(returns: pd.Series | np.ndarray) -> dict[str, Any]:
    """Calculate econometric realism metrics on return series."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 20:
        raise ValueError(f"At least 20 observations required for realism fingerprint, got {n}")

    # Excess kurtosis (Fisher definition: normal = 0.0)
    k_excess = float(kurtosis(arr, fisher=True))

    # Ljung-Box test on squared returns at lag 10
    lb_df = acorr_ljungbox(arr ** 2, lags=[10], return_df=True)
    lb_p = float(lb_df["lb_pvalue"].iloc[0])

    # Autocorrelation of absolute returns at lag 1
    abs_ret = np.abs(arr)
    autocorr_s = pd.Series(abs_ret).autocorr(lag=1)
    autocorr_lag1 = float(autocorr_s) if not np.isnan(autocorr_s) else 0.0

    return {
        "abs_return_autocorr_lag1": autocorr_lag1,
        "excess_kurtosis": k_excess,
        "ljung_box_p_squared_lag10": lb_p,
        "n": n,
    }


def looks_synthetic_gbm(fp: dict[str, Any]) -> bool:
    """Heuristic test checking if returns resemble pure geometric Brownian motion.

    NOTE: This is an empirical heuristic flag, not mathematical proof. Real market data
    exhibits volatility clustering (small Ljung-Box p-value) and fat tails (excess kurtosis >= 1.0).
    """
    n = fp.get("n", 0)
    k = fp.get("excess_kurtosis", 0.0)
    lb_p = fp.get("ljung_box_p_squared_lag10", 0.0)

    return bool(n >= 500 and k < 1.0 and lb_p > 0.05)
