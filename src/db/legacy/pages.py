from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from src.shared.env import ROOT, load_local_env, load_yaml, pages_db

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
DEFAULT_CHUNK_VARIANT = "base"


def raw_pages_db_path() -> Path:
    load_local_env()
    return pages_db()


def chunk_variant() -> str:
    """The chunk cut every command works on: `base` unless CHUNK_VARIANT says otherwise."""
    return os.getenv("CHUNK_VARIANT", DEFAULT_CHUNK_VARIANT)


def variant_tag(variant: str, separator: str) -> str:
    """The variant name behind `separator`, or nothing for `base` so it keeps its names."""
    return "" if variant == DEFAULT_CHUNK_VARIANT else f"{separator}{variant}"


def connect_pages() -> sqlite3.Connection:
    conn = sqlite3.connect(raw_pages_db_path())
    conn.execute("pragma foreign_keys = on")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_page_artifacts_db() -> None:
    schema = (ROOT / "sql" / "eval.sql").read_text(encoding="utf-8")
    with connect_pages() as conn:
        _require_chunk_variant_column(conn)
        conn.executescript(schema)
        sources = [
            row["source"]
            for row in conn.execute("select distinct source from page_metadata")
        ]
        conn.executemany(
            "insert or ignore into page_sources (source, language) values (?, ?)",
            [(source, CONFIG["default_source_language"]) for source in sources],
        )


def _require_chunk_variant_column(conn: sqlite3.Connection) -> None:
    """Refuse a `page_chunks` from before chunk variants; its rebuild is a manual step."""
    columns = {row["name"] for row in conn.execute("pragma table_info(page_chunks)")}
    if columns and "variant" not in columns:
        raise SystemExit(
            "page_chunks predates chunk variants. Apply the one-off migration first:\n"
            f'  sqlite3 "{raw_pages_db_path()}" < sql/migrate_chunk_variants.sql'
        )
