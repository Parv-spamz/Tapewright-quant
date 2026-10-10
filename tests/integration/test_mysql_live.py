import os
import pytest

from database.connection.manager import ConnectionManager
from database.migrations.runner import apply_migrations
from database.repositories.market_data_repo.repo import AssetRepo

TAPEWRIGHT_DB_HOST = os.environ.get("TAPEWRIGHT_DB_HOST")


@pytest.mark.skipif(
    not TAPEWRIGHT_DB_HOST,
    reason="Live MySQL tests skipped: TAPEWRIGHT_DB_HOST environment variable is not set.",
)
class TestLiveMySQL:
    """Integration test suite executing against a live MySQL server when configured."""

    def test_live_mysql_connection_and_migrations(self) -> None:
        ConnectionManager.reset_instance()
        manager = ConnectionManager.get_instance(backend="mysql")

        with manager.connection() as conn:
            apply_migrations(conn)

            cursor = conn.cursor()
            cursor.execute("SHOW TABLES;")
            tables = {row[0] for row in cursor.fetchall()}
            assert "assets" in tables
            assert "trade_signals" in tables

        asset_repo = AssetRepo(connection_manager=manager)
        asset_repo.upsert_asset(
            asset_id="live_test_01",
            symbol="TEST-LIVE",
            name="Live Integration Test Asset",
            asset_class="EQUITY",
            exchange="NYSE",
        )
        asset = asset_repo.get_asset("TEST-LIVE")
        assert asset is not None
        assert asset["symbol"] == "TEST-LIVE"
