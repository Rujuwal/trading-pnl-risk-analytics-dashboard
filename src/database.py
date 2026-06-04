from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DB_PATH, SCHEMA_PATH, ensure_directories


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_database(db_path: Path = DB_PATH, reset: bool = False) -> None:
    ensure_directories()
    if reset and db_path.exists():
        db_path.unlink()

    with connect(db_path) as conn:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema)
        conn.commit()


def table_row_count(table_name: str, db_path: Path = DB_PATH) -> int:
    if not db_path.exists():
        return 0

    with connect(db_path) as conn:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(result[0])


def write_dataframe(table_name: str, frame: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        frame.to_sql(table_name, conn, if_exists="append", index=False)
        conn.commit()


def read_table(table_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def read_trade_pnl_view(db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM trade_pnl", conn)

