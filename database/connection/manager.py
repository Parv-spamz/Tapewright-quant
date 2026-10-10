import logging
import os
import queue
import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

from config.database.db_config import DBConfig, retry_with_backoff


def adapt_sql(sql: str, backend: str) -> str:
    """Adapt SQL syntax and placeholders for the specified backend.

    Translates parameter placeholders: '%s' -> '?' for SQLite in ONE place.
    """
    if backend.lower() == "sqlite":
        # Replace %s placeholder with ? for SQLite parameter binding
        # Avoid replacing % if it's not a %s placeholder
        return re.sub(r"%s", "?", sql)
    return sql


class ConnectionManager:
    """Thread-safe database connection manager supporting MySQL and SQLite backends."""

    _instance: "ConnectionManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        backend: str = "sqlite",
        db_config: DBConfig | None = None,
        sqlite_path: str = ":memory:",
    ) -> None:
        self.backend = backend.lower()
        self.db_config = db_config or DBConfig.from_env()
        self.sqlite_path = sqlite_path

        # Synchronization primitives
        self._sqlite_lock = threading.RLock()
        self._sqlite_conn: sqlite3.Connection | None = None

        # MySQL pool
        self._mysql_pool: queue.Queue[Any] | None = None
        self._mysql_lock = threading.Lock()
        self._created_mysql_connections = 0

        if self.backend == "sqlite":
            self._init_sqlite()
        elif self.backend == "mysql":
            self._init_mysql_pool()
        else:
            raise ValueError(f"Unsupported database backend '{backend}'. Must be 'mysql' or 'sqlite'.")

    def _init_sqlite(self) -> None:
        self._sqlite_conn = sqlite3.connect(
            self.sqlite_path,
            check_same_thread=False,
            timeout=30.0,
        )
        # Enable foreign keys for SQLite
        self._sqlite_conn.execute("PRAGMA foreign_keys = ON;")

    def _init_mysql_pool(self) -> None:
        self._mysql_pool = queue.Queue(maxsize=self.db_config.pool_size)

    def _create_mysql_connection(self) -> Any:
        def _connect() -> Any:
            try:
                import pymysql

                return pymysql.connect(
                    host=self.db_config.host,
                    port=self.db_config.port,
                    user=self.db_config.user,
                    password=self.db_config.password,
                    database=self.db_config.database,
                    connect_timeout=self.db_config.connect_timeout,
                    autocommit=False,
                )
            except ImportError:
                import mysql.connector

                return mysql.connector.connect(
                    host=self.db_config.host,
                    port=self.db_config.port,
                    user=self.db_config.user,
                    password=self.db_config.password,
                    database=self.db_config.database,
                    connection_timeout=self.db_config.connect_timeout,
                    autocommit=False,
                )

        return retry_with_backoff(
            _connect,
            max_retries=self.db_config.max_retries,
            backoff_s=self.db_config.retry_backoff_s,
        )

    @classmethod
    def get_instance(
        cls,
        backend: str | None = None,
        db_config: DBConfig | None = None,
        sqlite_path: str = ":memory:",
    ) -> "ConnectionManager":
        with cls._instance_lock:
            if cls._instance is None:
                chosen_backend = backend or os.environ.get("TAPEWRIGHT_DB_BACKEND", "sqlite")
                cls._instance = cls(
                    backend=chosen_backend,
                    db_config=db_config,
                    sqlite_path=sqlite_path,
                )
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close_all()
                cls._instance = None

    def close_all(self) -> None:
        if self._sqlite_conn is not None:
            with self._sqlite_lock:
                try:
                    self._sqlite_conn.close()
                except Exception as e:
                    logger.warning("Error closing SQLite connection: %s", e)
                self._sqlite_conn = None

        if self._mysql_pool is not None:
            while not self._mysql_pool.empty():
                try:
                    conn = self._mysql_pool.get_nowait()
                    conn.close()
                except Exception as e:
                    logger.warning("Error closing MySQL pooled connection: %s", e)

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Context manager yielding an active database connection.

        Commits on success, rolls back on exception, and always releases/synchronizes.
        """
        if self.backend == "sqlite":
            with self._sqlite_lock:
                if self._sqlite_conn is None:
                    self._init_sqlite()
                conn = self._sqlite_conn
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
        else:
            # MySQL connection from pool
            conn = None
            try:
                try:
                    conn = self._mysql_pool.get_nowait()
                except queue.Empty:
                    with self._mysql_lock:
                        if self._created_mysql_connections < self.db_config.pool_size:
                            conn = self._create_mysql_connection()
                            self._created_mysql_connections += 1
                        else:
                            conn = self._mysql_pool.get(timeout=self.db_config.connect_timeout)

                yield conn
                conn.commit()
            except Exception:
                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception as e:
                        logger.warning("Error rolling back MySQL connection: %s", e)
                raise
            finally:
                if conn is not None and self._mysql_pool is not None:
                    self._mysql_pool.put(conn)


@contextmanager
def get_db_connection() -> Generator[Any, None, None]:
    """Convenience module-level context manager for database connection."""
    manager = ConnectionManager.get_instance()
    with manager.connection() as conn:
        yield conn
