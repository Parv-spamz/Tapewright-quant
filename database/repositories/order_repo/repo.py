from typing import Any

from database.connection.manager import ConnectionManager, adapt_sql
from database.repositories.market_data_repo.repo import _fetch_all_dicts, _fetch_one_dict


class OrderRepo:
    """Repository for managing simulated and executed orders."""

    def __init__(self, connection_manager: ConnectionManager | None = None) -> None:
        self.cm = connection_manager or ConnectionManager.get_instance()

    def insert_order(
        self,
        order_id: str,
        asset_id: str,
        side: str,
        order_type: str,
        quantity: float,
        executed_price: float,
        signal_id: int | None = None,
        slippage_cost: float = 0.0,
        fee: float = 0.0,
        executed_at: Any = None,
    ) -> str:
        """Insert an order execution record."""
        if executed_at is not None:
            sql = """
                INSERT INTO order_executions (
                    order_id, asset_id, signal_id, side, order_type,
                    quantity, executed_price, slippage_cost, fee, executed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            params: tuple[Any, ...] = (
                order_id, asset_id, signal_id, side, order_type,
                quantity, executed_price, slippage_cost, fee, executed_at,
            )
        else:
            sql = """
                INSERT INTO order_executions (
                    order_id, asset_id, signal_id, side, order_type,
                    quantity, executed_price, slippage_cost, fee
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            params = (
                order_id, asset_id, signal_id, side, order_type,
                quantity, executed_price, slippage_cost, fee,
            )

        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, params)
        return order_id

    def update_fill_status(
        self,
        order_id: str,
        executed_price: float | None = None,
        quantity: float | None = None,
        slippage_cost: float | None = None,
        fee: float | None = None,
        status: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Update fill execution details for an existing order.

        Accepts updated executed_price, quantity, slippage_cost, and fee.
        """
        updates: list[str] = []
        params: list[Any] = []

        if executed_price is not None:
            updates.append("executed_price = %s")
            params.append(executed_price)
        if quantity is not None:
            updates.append("quantity = %s")
            params.append(quantity)
        if slippage_cost is not None:
            updates.append("slippage_cost = %s")
            params.append(slippage_cost)
        if fee is not None:
            updates.append("fee = %s")
            params.append(fee)

        if not updates:
            # Check existence if no numeric update was requested
            sql_check = "SELECT 1 FROM order_executions WHERE order_id = %s LIMIT 1;"
            adapted_check = adapt_sql(sql_check, self.cm.backend)
            with self.cm.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(adapted_check, (order_id,))
                return cursor.fetchone() is not None

        set_clause = ", ".join(updates)
        params.append(order_id)
        sql = f"UPDATE order_executions SET {set_clause} WHERE order_id = %s;"
        adapted = adapt_sql(sql, self.cm.backend)

        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, tuple(params))
            return cursor.rowcount > 0

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Fetch order details by order_id."""
        sql = """
            SELECT o.order_id, o.asset_id, a.symbol, o.signal_id, o.side,
                   o.order_type, o.quantity, o.executed_price, o.slippage_cost,
                   o.fee, o.executed_at
            FROM order_executions o
            JOIN assets a ON o.asset_id = a.asset_id
            WHERE o.order_id = %s
            LIMIT 1;
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, (order_id,))
            return _fetch_one_dict(cursor)

    def list_orders(
        self,
        symbol: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List orders, optionally filtered by symbol or execution status."""
        # Since order_executions records filled transactions:
        if status is not None and status.strip().upper() not in ("FILLED", "EXECUTED"):
            return []

        conditions: list[str] = []
        params: list[Any] = []

        if symbol is not None:
            conditions.append("a.symbol = %s")
            params.append(symbol)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT o.order_id, o.asset_id, a.symbol, o.signal_id, o.side,
                   o.order_type, o.quantity, o.executed_price, o.slippage_cost,
                   o.fee, o.executed_at
            FROM order_executions o
            JOIN assets a ON o.asset_id = a.asset_id
            {where_clause}
            ORDER BY o.executed_at DESC, o.order_id DESC;
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, tuple(params))
            return _fetch_all_dicts(cursor)
