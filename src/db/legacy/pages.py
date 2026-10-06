from __future__ import annotations

import sqlite3
from pathlib import Path

from src.shared.env import ROOT, load_local_env, load_yaml, pages_db

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def raw_pages_db_path() -> Path:
    load_local_env()
    return pages_db()


def connect_pages() -> sqlite3.Connection:
    conn = sqlite3.connect(raw_pages_db_path())
    conn.execute("pragma foreign_keys = on")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_page_artifacts_db() -> None:
    schema = (ROOT / "sql" / "eval.sql").read_text(encoding="utf-8")
    with connect_pages() as conn:
        conn.executescript(schema)
        sources = [
            row["source"]
            for row in conn.execute("select distinct source from page_metadata")
        ]
        conn.executemany(
            "insert or ignore into page_sources (source, language) values (?, ?)",
            [(source, CONFIG["default_source_language"]) for source in sources],
        )
