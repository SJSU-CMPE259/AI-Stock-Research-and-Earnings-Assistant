"""Open the SQLite database and create the tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from finsight import config

SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the database (creating the folder and tables if needed).

    Rows come back as sqlite3.Row, so you can use row["ticker"] or dict(row).
    """
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA.read_text())
    return conn
