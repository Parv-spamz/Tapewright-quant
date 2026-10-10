import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from database.connection.manager import ConnectionManager, adapt_sql
from database.repositories.market_data_repo.repo import _fetch_all_dicts, _fetch_one_dict


def _parse_signal_metadata(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    val = row.get("metadata")
    if isinstance(val, str):
        try:
            row["metadata"] = json.loads(val)
        except Exception as e:
            logger.warning("Failed to deserialize signal metadata JSON: %s", e)
    return row


class SignalRepo:
    """Repository for managing generated trading signals."""

    def __init__(self, connection_manager: ConnectionManager | None = None) -> None:
        self.cm = connection_manager or ConnectionManager.get_instance()

    def insert_signal(
        self,
        asset_id: str,
        model_id: str,
        signal_type: str,
        confidence_score: float,
        target_weight: float | None = None,
        metadata: dict[str, Any] | str | None = None,
        timestamp: Any = None,
    ) -> int:
        """Insert a single trade signal and return its generated signal_id."""
        meta_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata

        if timestamp is not None:
            sql = """
                INSERT INTO trade_signals (asset_id, model_id, signal_type, confidence_score, target_weight, metadata, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            params = (asset_id, model_id, signal_type, confidence_score, target_weight, meta_json, timestamp)
        else:
            sql = """
                INSERT INTO trade_signals (asset_id, model_id, signal_type, confidence_score, target_weight, metadata)
                VALUES (%s, %s, %s, %s, %s, %s);
            """
            params = (asset_id, model_id, signal_type, confidence_score, target_weight, meta_json)

        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, params)
            return int(cursor.lastrowid)

    def insert_signals_bulk(self, signals: list[dict[str, Any]]) -> int:
        """Bulk insert a list of signal dictionaries. Returns the number of inserted signals."""
        if not signals:
            return 0

        has_timestamp = any("timestamp" in s and s["timestamp"] is not None for s in signals)

        if has_timestamp:
            sql = """
                INSERT INTO trade_signals (asset_id, model_id, signal_type, confidence_score, target_weight, metadata, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            rows = [
                (
                    s["asset_id"],
                    s["model_id"],
                    s["signal_type"],
                    s["confidence_score"],
                    s.get("target_weight"),
                    json.dumps(s["metadata"]) if isinstance(s.get("metadata"), dict) else s.get("metadata"),
                    s.get("timestamp"),
                )
                for s in signals
            ]
        else:
            sql = """
                INSERT INTO trade_signals (asset_id, model_id, signal_type, confidence_score, target_weight, metadata)
                VALUES (%s, %s, %s, %s, %s, %s);
            """
            rows = [
                (
                    s["asset_id"],
                    s["model_id"],
                    s["signal_type"],
                    s["confidence_score"],
                    s.get("target_weight"),
                    json.dumps(s["metadata"]) if isinstance(s.get("metadata"), dict) else s.get("metadata"),
                )
                for s in signals
            ]

        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(adapted, rows)
            return len(signals)

    def get_signals(
        self,
        symbol: str,
        start: Any = None,
        end: Any = None,
    ) -> list[dict[str, Any]]:
        """Retrieve trade signals for a given symbol within optional start/end timestamps."""
        conditions = ["a.symbol = %s"]
        params: list[Any] = [symbol]

        if start is not None:
            conditions.append("s.timestamp >= %s")
            params.append(start)
        if end is not None:
            conditions.append("s.timestamp <= %s")
            params.append(end)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT s.signal_id, s.asset_id, a.symbol, s.model_id, s.timestamp,
                   s.signal_type, s.confidence_score, s.target_weight, s.metadata
            FROM trade_signals s
            JOIN assets a ON s.asset_id = a.asset_id
            WHERE {where_clause}
            ORDER BY s.timestamp ASC, s.signal_id ASC;
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, tuple(params))
            rows = _fetch_all_dicts(cursor)
            return [_parse_signal_metadata(r) for r in rows if r is not None]  # type: ignore[misc]

    def latest_signal(self, symbol: str) -> dict[str, Any] | None:
        """Fetch the most recent trade signal for a given symbol."""
        sql = """
            SELECT s.signal_id, s.asset_id, a.symbol, s.model_id, s.timestamp,
                   s.signal_type, s.confidence_score, s.target_weight, s.metadata
            FROM trade_signals s
            JOIN assets a ON s.asset_id = a.asset_id
            WHERE a.symbol = %s
            ORDER BY s.timestamp DESC, s.signal_id DESC
            LIMIT 1;
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, (symbol,))
            row = _fetch_one_dict(cursor)
            return _parse_signal_metadata(row)
