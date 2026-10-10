"""Chronological dataset splits and split boundary classification.

Defines non-overlapping split windows for equities and crypto:
- dev: in-sample model exploration and training
- val: out-of-sample model selection and hyperparameter validation
- lockbox: untouched evaluation holdout
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping
import pandas as pd

EQUITY_SPLITS = {
    "dev": ("2005-01-03", "2019-12-31"),
    "val": ("2020-01-01", "2023-12-31"),
    "lockbox": ("2024-01-01", "2099-12-31"),
}

CRYPTO_SPLITS = {
    "dev": ("2017-08-17", "2021-12-31"),
    "val": ("2022-01-01", "2023-12-31"),
    "lockbox": ("2024-01-01", "2099-12-31"),
}


def _validate_splits(splits: Mapping[str, tuple[str, str]], name: str) -> None:
    """Validate that split boundaries are chronological and non-overlapping."""
    required = ("dev", "val", "lockbox")
    for r in required:
        if r not in splits:
            raise ValueError(f"Splits mapping {name} missing required key '{r}'")

    dev_start, dev_end = pd.Timestamp(splits["dev"][0]), pd.Timestamp(splits["dev"][1])
    val_start, val_end = pd.Timestamp(splits["val"][0]), pd.Timestamp(splits["val"][1])
    lock_start, lock_end = pd.Timestamp(splits["lockbox"][0]), pd.Timestamp(splits["lockbox"][1])

    if not (dev_start <= dev_end < val_start <= val_end < lock_start <= lock_end):
        raise ValueError(
            f"Splits mapping {name} is not strictly chronological or contains overlapping windows: "
            f"dev=[{dev_start}, {dev_end}], val=[{val_start}, {val_end}], lockbox=[{lock_start}, {lock_end}]"
        )


# Validate splits at import time
_validate_splits(EQUITY_SPLITS, "EQUITY_SPLITS")
_validate_splits(CRYPTO_SPLITS, "CRYPTO_SPLITS")


def split_of(timestamp: str | date | datetime | pd.Timestamp, splits: Mapping[str, tuple[str, str]]) -> str:
    """Classify a timestamp into 'dev', 'val', or 'lockbox' based on provided split windows."""
    ts = pd.Timestamp(timestamp)

    for split_name in ("dev", "val", "lockbox"):
        start_str, end_str = splits[split_name]
        start_ts = pd.Timestamp(start_str)
        end_ts = pd.Timestamp(end_str)
        if start_ts <= ts <= end_ts:
            return split_name

    raise ValueError(f"Timestamp {ts} does not fall into any defined split window ({list(splits.keys())})")
