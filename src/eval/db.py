from __future__ import annotations

import sqlite3
from pathlib import Path

from src.shared.env import ROOT, load_local_env, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def raw_pages_db_path() -> Path:
    load_local_env()
    path = ROOT / CONFIG["raw_pages_db"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(raw_pages_db_path())
    conn.execute("pragma foreign_keys = on")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_eval_db() -> None:
    schema = (ROOT / "sql" / "eval.sql").read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema)
        question_columns = {
            row["name"] for row in conn.execute("pragma table_info(eval_questions)")
        }
        if "approved" not in question_columns:
            conn.execute(
                "alter table eval_questions add column approved integer not null default 1"
            )
        chunk_columns = {
            row["name"] for row in conn.execute("pragma table_info(page_chunks)")
        }
        if "summary" not in chunk_columns:
            conn.execute("alter table page_chunks add column summary text")
        sources = [
            row["source"]
            for row in conn.execute("select distinct source from page_metadata")
        ]
        conn.executemany(
            "insert or ignore into page_sources (source, language) values (?, ?)",
            [(source, CONFIG["default_source_language"]) for source in sources],
        )
