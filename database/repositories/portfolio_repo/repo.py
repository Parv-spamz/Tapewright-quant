from typing import Any
import numpy as np
import pandas as pd

from database.connection.manager import ConnectionManager, adapt_sql


class PortfolioRepo:
    """Repository for logging and analyzing portfolio equity and performance snapshots."""

    def __init__(self, connection_manager: ConnectionManager | None = None) -> None:
        self.cm = connection_manager or ConnectionManager.get_instance()

    def insert_snapshot(
        self,
        portfolio_name: str,
        total_nav: float,
        cash_balance: float,
        unrealized_pnl: float,
        realized_pnl: float,
        sharpe_ratio: float | None = None,
        max_drawdown: float | None = None,
        timestamp: Any = None,
    ) -> int:
        """Insert a portfolio performance snapshot and return its generated snapshot_id."""
        if timestamp is not None:
            sql = """
                INSERT INTO portfolio_snapshots (
                    portfolio_name, total_nav, cash_balance, unrealized_pnl,
                    realized_pnl, sharpe_ratio, max_drawdown, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            params: tuple[Any, ...] = (
                portfolio_name, total_nav, cash_balance, unrealized_pnl,
                realized_pnl, sharpe_ratio, max_drawdown, timestamp,
            )
        else:
            sql = """
                INSERT INTO portfolio_snapshots (
                    portfolio_name, total_nav, cash_balance, unrealized_pnl,
                    realized_pnl, sharpe_ratio, max_drawdown
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            params = (
                portfolio_name, total_nav, cash_balance, unrealized_pnl,
                realized_pnl, sharpe_ratio, max_drawdown,
            )

        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, params)
            return int(cursor.lastrowid)

    def get_equity_curve(
        self,
        start: Any = None,
        end: Any = None,
        portfolio_name: str = "default",
    ) -> pd.Series:
        """Retrieve historical total NAV equity curve as a pandas Series indexed by timestamp."""
        conditions = ["portfolio_name = %s"]
        params: list[Any] = [portfolio_name]

        if start is not None:
            conditions.append("timestamp >= %s")
            params.append(start)
        if end is not None:
            conditions.append("timestamp <= %s")
            params.append(end)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT timestamp, total_nav
            FROM portfolio_snapshots
            WHERE {where_clause}
            ORDER BY timestamp ASC, snapshot_id ASC;
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, tuple(params))
            rows = cursor.fetchall()

        if not rows:
            return pd.Series(dtype=float, name="equity")

        timestamps = [pd.to_datetime(r[0]) for r in rows]
        navs = [float(r[1]) for r in rows]
        return pd.Series(data=navs, index=pd.DatetimeIndex(timestamps), name="equity")

    @staticmethod
    def compute_drawdown_series(equity: pd.Series | list[float] | np.ndarray) -> pd.Series:
        """Compute the relative drawdown series from an equity series.

        Drawdown formula: (Equity_t - CumulativeMax_t) / CumulativeMax_t
        """
        if not isinstance(equity, pd.Series):
            s = pd.Series(equity, dtype=float)
        else:
            s = equity.astype(float)

        if s.empty:
            return pd.Series(dtype=float)

        running_max = s.cummax()
        drawdown = (s - running_max) / running_max
        return drawdown
