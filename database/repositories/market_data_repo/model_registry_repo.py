import json
from typing import Any

from database.connection.manager import ConnectionManager, adapt_sql
from database.repositories.market_data_repo.repo import _fetch_all_dicts, _fetch_one_dict


def _parse_json_fields(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    for field in ("hyperparameters", "metrics"):
        val = row.get(field)
        if isinstance(val, str):
            try:
                row[field] = json.loads(val)
            except Exception:
                pass
    return row


class ModelRegistryRepo:
    """Repository for managing registered quantitative and ML models."""

    def __init__(self, connection_manager: ConnectionManager | None = None) -> None:
        self.cm = connection_manager or ConnectionManager.get_instance()

    def register(
        self,
        model_id: str,
        model_name: str,
        model_family: str,
        version: str,
        hyperparameters: dict[str, Any] | str | None = None,
        metrics: dict[str, Any] | str | None = None,
    ) -> str:
        """Register a new model with metadata, hyperparameters, and evaluation metrics."""
        hp_json = json.dumps(hyperparameters) if isinstance(hyperparameters, dict) else hyperparameters
        metrics_json = json.dumps(metrics) if isinstance(metrics, dict) else metrics

        sql = """
            INSERT INTO model_registry (model_id, model_name, model_family, version, hyperparameters, metrics)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                adapted,
                (model_id, model_name, model_family, version, hp_json, metrics_json),
            )
        return model_id

    def get_latest(self, name: str) -> dict[str, Any] | None:
        """Retrieve the most recently registered model by model_name."""
        sql = """
            SELECT model_id, model_name, model_family, version, hyperparameters, metrics, trained_at
            FROM model_registry
            WHERE model_name = %s
            ORDER BY trained_at DESC, model_id DESC
            LIMIT 1;
        """
        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, (name,))
            row = _fetch_one_dict(cursor)
            return _parse_json_fields(row)

    def list(self, family: str | None = None) -> list[dict[str, Any]]:
        """List registered models, optionally filtered by model_family."""
        if family is not None:
            sql = """
                SELECT model_id, model_name, model_family, version, hyperparameters, metrics, trained_at
                FROM model_registry
                WHERE model_family = %s
                ORDER BY trained_at DESC;
            """
            params: tuple[Any, ...] = (family,)
        else:
            sql = """
                SELECT model_id, model_name, model_family, version, hyperparameters, metrics, trained_at
                FROM model_registry
                ORDER BY trained_at DESC;
            """
            params = ()

        adapted = adapt_sql(sql, self.cm.backend)
        with self.cm.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(adapted, params)
            rows = _fetch_all_dicts(cursor)
            return [_parse_json_fields(r) for r in rows if r is not None]  # type: ignore[misc]
