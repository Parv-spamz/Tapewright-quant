import os
import re
import sqlite3
from pathlib import Path
from typing import Any


def translate_ddl_for_sqlite(sql: str) -> str:
    """Translate MySQL DDL script into SQLite-compatible DDL.

    Handles:
    - Stripping CREATE DATABASE and USE statements
    - Stripping ENGINE=InnoDB, CHARACTER SET, COLLATE
    - AUTO_INCREMENT -> INTEGER PRIMARY KEY AUTOINCREMENT
    - ENUM(...) -> TEXT
    - JSON -> TEXT
    - UNSIGNED -> removed
    - Backticks -> stripped
    - UNIQUE KEY uq_name (cols) -> UNIQUE (cols)
    - Non-unique inline KEY / INDEX definitions -> stripped or separated
    """
    s = sql

    # Remove comments that might interfere with regexes
    # Remove CREATE DATABASE and USE statements
    s = re.sub(r"CREATE\s+DATABASE\s+[^;]+;", "", s, flags=re.IGNORECASE)
    s = re.sub(r"USE\s+[^;]+;", "", s, flags=re.IGNORECASE)

    # Remove table-level options: ENGINE=..., DEFAULT CHARSET=..., COLLATE=...
    s = re.sub(r"ENGINE\s*=\s*\w+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"DEFAULT\s+CHARSET\s*=\s*\w+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"DEFAULT\s+CHARACTER\s+SET\s*=\s*\w+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"COLLATE\s*=\s*\w+", "", s, flags=re.IGNORECASE)

    # Remove backticks
    s = s.replace("`", "")

    # Convert AUTO_INCREMENT to SQLite INTEGER PRIMARY KEY AUTOINCREMENT
    s = re.sub(
        r"(?:BIGINT|INT|INTEGER)\s+AUTO_INCREMENT\s+PRIMARY\s+KEY",
        "INTEGER PRIMARY KEY AUTOINCREMENT",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"AUTO_INCREMENT\s+PRIMARY\s+KEY",
        "INTEGER PRIMARY KEY AUTOINCREMENT",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"AUTO_INCREMENT", "", s, flags=re.IGNORECASE)

    # Convert ENUM(...) to TEXT
    s = re.sub(r"ENUM\s*\([^)]+\)", "TEXT", s, flags=re.IGNORECASE)

    # Convert JSON to TEXT
    s = re.sub(r"\bJSON\b", "TEXT", s, flags=re.IGNORECASE)

    # Remove UNSIGNED
    s = re.sub(r"\bUNSIGNED\b", "", s, flags=re.IGNORECASE)

    # Convert UNIQUE KEY name (cols) -> UNIQUE (cols)
    s = re.sub(
        r"UNIQUE\s+KEY\s+(?:\w+\s*)?\(([^)]+)\)",
        r"UNIQUE (\1)",
        s,
        flags=re.IGNORECASE,
    )

    # Remove inline non-unique KEY name (cols)
    s = re.sub(
        r",\s*(?:KEY|INDEX)\s+\w+\s*\([^)]+\)",
        "",
        s,
        flags=re.IGNORECASE,
    )

    return s


def apply_migrations(conn: Any, migrations_dir: str | Path | None = None) -> list[str]:
    """Apply unapplied database migrations from migrations_dir in order.

    Ensures table schema_migrations exists, applies each migration in a transaction,
    and records the applied version. Returns a list of newly applied migration versions.
    """
    if migrations_dir is None:
        target_dir = Path(__file__).resolve().parent
    else:
        target_dir = Path(migrations_dir).resolve()

    is_sqlite = isinstance(conn, sqlite3.Connection) or hasattr(conn, "total_changes")

    cursor = conn.cursor()

    # 1. Ensure schema_migrations table exists
    if is_sqlite:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
    else:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(64) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
            """
        )
    conn.commit()

    # 2. Query already applied migrations
    cursor.execute("SELECT version FROM schema_migrations")
    rows = cursor.fetchall()
    applied_versions = {row[0] for row in rows}

    # 3. Discover migration files
    migration_files = sorted(
        [f for f in target_dir.glob("*.sql") if re.match(r"^\d{3}_.*\.sql$", f.name)],
        key=lambda p: p.name,
    )

    newly_applied: list[str] = []

    for mfile in migration_files:
        version_name = mfile.stem
        if version_name in applied_versions:
            continue

        with open(mfile, "r", encoding="utf-8") as f:
            raw_sql = f.read()

        sql_to_run = translate_ddl_for_sqlite(raw_sql) if is_sqlite else raw_sql

        # Split into individual statements
        statements = [stmt.strip() for stmt in sql_to_run.split(";") if stmt.strip()]

        try:
            for stmt in statements:
                cursor.execute(stmt)

            # Record migration
            placeholder = "?" if is_sqlite else "%s"
            cursor.execute(
                f"INSERT INTO schema_migrations (version) VALUES ({placeholder})",
                (version_name,),
            )
            conn.commit()
            newly_applied.append(version_name)
        except Exception:
            conn.rollback()
            raise

    return newly_applied
