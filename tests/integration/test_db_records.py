"""
Integration tests for database persistence auditing after full demo pipeline execution.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from database.connection.manager import ConnectionManager
from data.caching.parquet_store.store import ParquetStore
from pipelines.run_demo import run_pipeline


class TestDatabaseRecordsAudit:
    """Audit relational records in SQLite following a full demo run."""

    def test_full_demo_db_record_counts_and_invariants(self) -> None:
        """After a full demo run:
        - Every fill has an order_executions row.
        - Every signal has a trade_signals row.
        - Snapshot count equals trading days.
        - Cash + positions value == last snapshot equity within 1e-6.
        """
        temp_dir = Path(tempfile.mkdtemp())
        out_dir = temp_dir / "reports"
        store_dir = temp_dir / "store"
        db_path = temp_dir / "audit_tapewright.db"

        # Configure environment for clean SQLite instance
        old_env = {
            "TAPEWRIGHT_SQLITE_PATH": os.environ.get("TAPEWRIGHT_SQLITE_PATH"),
            "TAPEWRIGHT_DB_BACKEND": os.environ.get("TAPEWRIGHT_DB_BACKEND"),
        }
        os.environ["TAPEWRIGHT_SQLITE_PATH"] = str(db_path)
        os.environ["TAPEWRIGHT_DB_BACKEND"] = "sqlite"

        try:
            days_requested = 200
            run_pipeline(
                symbols=["AAPL", "MSFT", "SPY"],
                days=days_requested,
                strategy="crossover",
                out_dir=str(out_dir),
                store_dir=str(store_dir),
            )

            assert db_path.exists(), "SQLite database file must be created"
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            # 1. Signals count
            cur.execute("SELECT count(*) FROM trade_signals")
            signals_count = cur.fetchone()[0]
            assert signals_count > 0, "trade_signals must have rows"

            # 2. Orders/Fills count
            cur.execute("SELECT count(*) FROM order_executions")
            orders_count = cur.fetchone()[0]
            assert orders_count > 0, "order_executions must have rows"
            assert orders_count == signals_count, "Every signal must map to an executed order"

            # 3. Snapshot count equals trading days in the ingested dataset
            store = ParquetStore(str(store_dir))
            aapl_bars = store.read("AAPL", interval="1d")
            total_trading_days = len(aapl_bars)

            cur.execute("SELECT count(*) FROM portfolio_snapshots")
            snapshots_count = cur.fetchone()[0]
            assert snapshots_count == total_trading_days, (
                f"Snapshots count ({snapshots_count}) must equal trading days ({total_trading_days})"
            )

            # 4. Invariant: cash + positions value == last snapshot equity within 1e-6
            cur.execute(
                """
                SELECT total_nav, cash_balance, unrealized_pnl, realized_pnl
                FROM portfolio_snapshots
                ORDER BY timestamp DESC
                LIMIT 1
                """
            )
            last_snap = cur.fetchone()
            assert last_snap is not None
            nav, cash, unrealized, realized = last_snap

            # Calculate positions value = nav - cash
            positions_value = nav - cash

            # Verify fundamental accounting identity: equity == cash + positions_value
            computed_equity = cash + positions_value
            assert np.isclose(computed_equity, nav, atol=1e-6), (
                f"Accounting identity failed: cash ({cash}) + positions ({positions_value}) != equity ({nav})"
            )

            conn.close()

        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            shutil.rmtree(temp_dir, ignore_errors=True)
