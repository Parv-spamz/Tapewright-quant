import concurrent.futures
import sqlite3
import numpy as np
import pandas as pd
import pytest

from database.connection.manager import ConnectionManager, get_db_connection
from database.migrations.runner import apply_migrations
from database.repositories.market_data_repo.model_registry_repo import ModelRegistryRepo
from database.repositories.market_data_repo.repo import AssetRepo
from database.repositories.order_repo.repo import OrderRepo
from database.repositories.portfolio_repo.repo import PortfolioRepo
from database.repositories.signal_repo.repo import SignalRepo


@pytest.fixture(autouse=True)
def setup_sqlite_database() -> None:
    """Setup a clean SQLite in-memory database and run migrations before each test."""
    ConnectionManager.reset_instance()
    manager = ConnectionManager.get_instance(backend="sqlite", sqlite_path=":memory:")
    with manager.connection() as conn:
        apply_migrations(conn)
    yield
    ConnectionManager.reset_instance()


class TestMigrations:
    def test_all_schema_tables_exist_and_migrations_are_idempotent(self) -> None:
        manager = ConnectionManager.get_instance()
        with manager.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                "schema_migrations",
                "assets",
                "daily_market_summary",
                "model_registry",
                "trade_signals",
                "order_executions",
                "portfolio_snapshots",
            }
            assert expected_tables.issubset(tables)

            # Re-running migrations applies nothing the second time
            second_run = apply_migrations(conn)
            assert second_run == []


class TestRepositoriesRoundTrip:
    def test_crud_and_bulk_insert(self) -> None:
        asset_repo = AssetRepo()
        model_repo = ModelRegistryRepo()
        signal_repo = SignalRepo()
        order_repo = OrderRepo()
        portfolio_repo = PortfolioRepo()

        # 1. AssetRepo round trip
        asset_repo.upsert_asset(
            asset_id="asset_aapl",
            symbol="AAPL",
            name="Apple Inc.",
            asset_class="EQUITY",
            exchange="NASDAQ",
        )
        asset = asset_repo.get_asset("AAPL")
        assert asset is not None
        assert asset["symbol"] == "AAPL"
        assert asset["is_active"] in (1, True)

        asset_repo.set_trading_status("AAPL", False)
        updated_asset = asset_repo.get_asset("AAPL")
        assert updated_asset is not None
        assert updated_asset["is_active"] in (0, False)

        assets_list = asset_repo.list_assets(asset_class="EQUITY")
        assert len(assets_list) == 1

        # 2. ModelRegistryRepo round trip
        model_repo.register(
            model_id="mod_garch_01",
            model_name="garch_vol",
            model_family="TIME_SERIES",
            version="1.0.0",
            hyperparameters={"p": 1, "q": 1},
            metrics={"aic": 245.5},
        )
        model = model_repo.get_latest("garch_vol")
        assert model is not None
        assert model["model_id"] == "mod_garch_01"
        assert model["hyperparameters"] == {"p": 1, "q": 1}
        assert model["metrics"] == {"aic": 245.5}

        models_list = model_repo.list(family="TIME_SERIES")
        assert len(models_list) == 1

        # 3. SignalRepo round trip & bulk insert of 100 signals
        sig_id = signal_repo.insert_signal(
            asset_id="asset_aapl",
            model_id="mod_garch_01",
            signal_type="BUY",
            confidence_score=0.92,
            target_weight=0.15,
            metadata={"indicator": "rsi_oversold"},
        )
        assert sig_id > 0

        # Bulk insert of 100 signals
        bulk_signals = [
            {
                "asset_id": "asset_aapl",
                "model_id": "mod_garch_01",
                "signal_type": "BUY" if i % 2 == 0 else "HOLD",
                "confidence_score": 0.80 + (i % 20) * 0.01,
                "target_weight": 0.10,
                "metadata": {"batch_index": i},
            }
            for i in range(100)
        ]
        count = signal_repo.insert_signals_bulk(bulk_signals)
        assert count == 100

        retrieved_signals = signal_repo.get_signals("AAPL")
        assert len(retrieved_signals) == 101

        latest = signal_repo.latest_signal("AAPL")
        assert latest is not None
        assert latest["symbol"] == "AAPL"

        # 4. OrderRepo round trip
        order_repo.insert_order(
            order_id="ord_001",
            asset_id="asset_aapl",
            side="BUY",
            order_type="MARKET",
            quantity=100.0,
            executed_price=175.50,
            signal_id=sig_id,
            slippage_cost=0.05,
            fee=1.0,
        )
        order = order_repo.get_order("ord_001")
        assert order is not None
        assert order["executed_price"] == 175.50
        assert order["symbol"] == "AAPL"

        # Update fill status
        order_repo.update_fill_status("ord_001", executed_price=175.60, fee=1.20)
        updated_order = order_repo.get_order("ord_001")
        assert updated_order is not None
        assert updated_order["executed_price"] == 175.60
        assert updated_order["fee"] == 1.20

        orders = order_repo.list_orders(symbol="AAPL")
        assert len(orders) == 1

        # 5. PortfolioRepo round trip
        snap_id = portfolio_repo.insert_snapshot(
            portfolio_name="test_apex",
            total_nav=1_000_000.0,
            cash_balance=400_000.0,
            unrealized_pnl=15_000.0,
            realized_pnl=5_000.0,
            sharpe_ratio=2.15,
            max_drawdown=0.08,
            timestamp="2024-01-15 16:00:00",
        )
        assert snap_id > 0

        eq_curve = portfolio_repo.get_equity_curve(portfolio_name="test_apex")
        assert len(eq_curve) == 1
        assert eq_curve.iloc[0] == 1_000_000.0


class TestTransactionRollback:
    def test_rollback_on_exception(self) -> None:
        manager = ConnectionManager.get_instance()

        with pytest.raises(RuntimeError, match="Simulated crash"):
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO assets (asset_id, symbol, name, asset_class, exchange)
                    VALUES ('asset_crash', 'CRASH', 'Crash Test', 'EQUITY', 'NYSE');
                    """
                )
                raise RuntimeError("Simulated crash inside transaction")

        # After rollback, asset must not exist
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assets WHERE asset_id = 'asset_crash';")
            count = cursor.fetchone()[0]
            assert count == 0


class TestSQLInjectionSafety:
    def test_sql_injection_verbatim_storage(self) -> None:
        asset_repo = AssetRepo()
        malicious_symbol = "x'); DROP TABLE assets;--"

        asset_repo.upsert_asset(
            asset_id="asset_inj_01",
            symbol=malicious_symbol,
            name="Injection Vector Test",
            asset_class="EQUITY",
            exchange="NYSE",
        )

        # 1. Assets table must still exist
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='assets';")
            assert cursor.fetchone() is not None

        # 2. Malicious symbol must be stored verbatim
        retrieved = asset_repo.get_asset(malicious_symbol)
        assert retrieved is not None
        assert retrieved["symbol"] == malicious_symbol
        assert retrieved["name"] == "Injection Vector Test"


class TestMultithreadedConcurrency:
    def test_8_threads_x_25_inserts(self) -> None:
        portfolio_repo = PortfolioRepo()
        num_threads = 8
        inserts_per_thread = 25
        total_expected = num_threads * inserts_per_thread

        def worker_task(thread_id: int) -> list[int]:
            inserted_ids: list[int] = []
            for i in range(inserts_per_thread):
                sid = portfolio_repo.insert_snapshot(
                    portfolio_name="multi_thread_test",
                    total_nav=100_000.0 + thread_id * 1000.0 + i,
                    cash_balance=50_000.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                )
                inserted_ids.append(sid)
            return inserted_ids

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, tid) for tid in range(num_threads)]
            results = [f.result() for f in futures]

        all_ids = [item for sublist in results for item in sublist]
        assert len(all_ids) == total_expected

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM portfolio_snapshots WHERE portfolio_name = 'multi_thread_test';"
            )
            count = cursor.fetchone()[0]
            assert count == total_expected


class TestDrawdownMath:
    def test_compute_drawdown_series_exact_values(self) -> None:
        # Prompt requirement: equity [100,110,99,120,90] -> [0,0,-0.1,0,-0.25] to 1e-12
        equity_series = [100.0, 110.0, 99.0, 120.0, 90.0]
        expected_drawdown = np.array([0.0, 0.0, -0.1, 0.0, -0.25])

        computed_drawdown = PortfolioRepo.compute_drawdown_series(equity_series)

        np.testing.assert_allclose(computed_drawdown.values, expected_drawdown, atol=1e-12)
