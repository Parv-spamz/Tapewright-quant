"""Audited data access layer for research pipelines.

Strictly gates access to model exploration (dev) and model selection (val) splits,
prohibiting unmediated access to lockbox evaluation data and recording audit trails
in the tamper-evident ledger for every validation access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from data.caching.parquet_store.store import ParquetStore
from integrity.ledger import Ledger
from research.splits import CRYPTO_SPLITS, EQUITY_SPLITS, split_of


class LockboxError(Exception):
    """Raised when an unauthorized attempt is made to access lockbox holdout data."""


class ResearchData:
    """Access layer gating datasets by pre-registered protocol and split."""

    def __init__(
        self,
        protocol_id: str,
        ledger: Ledger | None = None,
        store_path: Path | str = "research/store_devval",
        ledger_path: Path | str = "research/ledger/ledger.jsonl",
    ) -> None:
        self.protocol_id = protocol_id
        self.ledger = ledger or Ledger(ledger_path)
        self.store = ParquetStore(store_path)
        self._val_access_count = 0

    def get(self, symbol: str, split: str) -> pd.DataFrame:
        """Retrieve dataset for a given symbol and split ('dev' or 'val').

        Access to 'lockbox' raises LockboxError.
        Every access to 'val' records an audit log entry in the ledger.
        """
        clean_split = split.strip().lower()

        if clean_split == "lockbox":
            raise LockboxError(
                f"Unauthorized access to lockbox split requested for '{symbol}'. "
                "Lockbox data may only be evaluated once via evaluate_lockbox_once()."
            )

        if clean_split not in ("dev", "val"):
            raise ValueError(f"Invalid split '{split}'. Must be 'dev' or 'val'.")

        if clean_split == "val":
            self._val_access_count += 1
            if self.ledger:
                self.ledger.append(
                    kind="data_access",
                    status="INFO",
                    tag=f"val_access.{symbol}",
                    payload={
                        "protocol_id": self.protocol_id,
                        "symbol": symbol,
                        "access_number": self._val_access_count,
                    },
                )

        full_df = self.store.read(symbol, "1d")
        if full_df.empty:
            return full_df

        # Filter to requested split
        is_crypto = "USDT" in symbol.upper()
        splits_map = CRYPTO_SPLITS if is_crypto else EQUITY_SPLITS

        mask = [split_of(dt, splits_map) == clean_split for dt in full_df.index]
        return full_df.loc[mask]

    def access_counts(self) -> int:
        """Return the number of validation split data accesses performed by this instance."""
        return self._val_access_count
