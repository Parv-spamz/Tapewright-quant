from typing import Any

from database.connection.manager import ConnectionManager, adapt_sql, get_db_connection


def _fetch_all_dicts(cursor: Any) -> list[dict[str, Any]]:
    if not cursor.description:
        return []
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _fetch_one_dict(cursor: Any) -> dict[str, Any] | None:
    if not cursor.description:
        return None
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))


class AssetRepo:
    """Repository for querying and updating asset registry metadata."""

    def __init__(self, connection_manager: ConnectionManager | None = None) -> None:
        self.cm = connection_manager or ConnectionManager.get_instance()

    def upsert_asset(
        self,
        asset_id: str,
        symbol: str,
        name: str,
        asset_class: str,
        exchange: str,
        currency: str = "USD",
        is_active: bool = True,
    ) -> None:
        """Insert or update an asset record."""
        backend = self.cm.backend

        if backend == "sqlite":
            sql = """
                INSERT INTO assets (asset_id, symbol, name, asset_class, exchange, currency, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(asset_id) DO UPDATE SET
                    symbol = excluded.symbol,
                    name = excluded.name,
                    asset_class = excluded.asset_class,
                    exchange = excluded.exchange,
                    currency = excluded.currency,
                    is_active = excluded.is_active;
            """
        else:
            sql = """
                INSERT INTO assets (asset_id, symbol, name, asset_class, exchange, currency, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    symbol = VALUES(symbol),
                    name = VALUES(name),
                    asset_class = VALUES(asset_class),
                    exchange = VALUES(exchange),
                    currency = VALUES(currency),
                    is_active = VALUES(is_active);
            """

        adapted = adapt_sql(sql, backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                adapted,
                (asset_id, symbol, name, asset_class, exchange, currency, int(is_active)),
            )

    def get_asset(self, symbol_or_id: str) -> dict[str, Any] | None:
        """Fetch asset by symbol or asset_id."""
        sql = """
            SELECT asset_id, symbol, name, asset_class, exchange, currency, is_active, created_at
            FROM assets
            WHERE symbol = %s OR asset_id = %s
            LIMIT 1;
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, (symbol_or_id, symbol_or_id))
            return _fetch_one_dict(cursor)

    def list_assets(self, asset_class: str | None = None) -> list[dict[str, Any]]:
        """List assets, optionally filtered by asset_class."""
        if asset_class is not None:
            sql = """
                SELECT asset_id, symbol, name, asset_class, exchange, currency, is_active, created_at
                FROM assets
                WHERE asset_class = %s
                ORDER BY symbol;
            """
            params: tuple[Any, ...] = (asset_class,)
        else:
            sql = """
                SELECT asset_id, symbol, name, asset_class, exchange, currency, is_active, created_at
                FROM assets
                ORDER BY symbol;
            """
            params = ()

        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, params)
            return _fetch_all_dicts(cursor)

    def set_trading_status(self, symbol_or_id: str, is_active: bool) -> bool:
        """Update active trading status for an asset."""
        sql = """
            UPDATE assets
            SET is_active = %s
            WHERE symbol = %s OR asset_id = %s;
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, (int(is_active), symbol_or_id, symbol_or_id))
            return cursor.rowcount > 0
