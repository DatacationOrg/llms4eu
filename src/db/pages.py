from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from src.shared.env import ROOT, load_local_env, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def raw_pages_db_path() -> Path:
    """Path to the page database.

    `PAGES_DB_PATH` overrides the configured location. Experiments that change
    the schema or write new chunk variants can then run against a copy under
    `.local/` instead of the durable database in `data/db/`.
    """
    load_local_env()
    path = ROOT / os.getenv("PAGES_DB_PATH", CONFIG["raw_pages_db"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect_pages() -> sqlite3.Connection:
    conn = sqlite3.connect(raw_pages_db_path())
    conn.execute("pragma foreign_keys = on")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_page_artifacts_db() -> None:
    schema = (ROOT / "sql" / "eval.sql").read_text(encoding="utf-8")
    with connect_pages() as conn:
        conn.executescript(schema)
        question_columns = {
            row["name"] for row in conn.execute("pragma table_info(eval_questions)")
        }
        if "approved" not in question_columns:
            conn.execute(
                "alter table eval_questions add column approved integer not null default 1"
            )
        if "variant" not in question_columns:
            conn.execute(
                "alter table eval_questions "
                "add column variant text not null default 'base'"
            )
        chunk_columns = {
            row["name"] for row in conn.execute("pragma table_info(page_chunks)")
        }
        if "summary" in chunk_columns:
            _drop_page_chunk_summary(conn)
            chunk_columns = {
                row["name"] for row in conn.execute("pragma table_info(page_chunks)")
            }
        if "variant" not in chunk_columns:
            _add_page_chunk_variant(conn)
        conn.executescript(
            """
            create index if not exists idx_page_chunks_variant
              on page_chunks(variant);
            create index if not exists idx_eval_questions_variant
              on eval_questions(variant, approved);
            """
        )
        sources = [
            row["source"]
            for row in conn.execute("select distinct source from page_metadata")
        ]
        conn.executemany(
            "insert or ignore into page_sources (source, language) values (?, ?)",
            [(source, CONFIG["default_source_language"]) for source in sources],
        )


def _add_page_chunk_variant(conn: sqlite3.Connection) -> None:
    """Give page_chunks a variant and character span, keeping existing ids.

    The unique constraint moves from (page_id, chunk_index) to
    (page_id, variant, chunk_index), which SQLite cannot express as an ALTER, so
    the table is rebuilt. Existing rows become the 'base' variant and keep their
    `{page_id}:{index}` ids, so the approved eval labels and every checkpointed
    ranking stay valid.
    """
    conn.commit()
    conn.execute("pragma foreign_keys = off")
    try:
        conn.executescript(
            """
            drop table if exists page_chunks_with_variant;

            create table page_chunks_with_variant (
              id text primary key,
              page_id text not null references page_metadata(id) on delete cascade,
              chunk_index integer not null,
              variant text not null default 'base',
              heading_path text,
              text text not null,
              char_count integer not null,
              start_char integer,
              end_char integer,
              unique(page_id, variant, chunk_index)
            );

            insert into page_chunks_with_variant (
              id, page_id, chunk_index, variant, heading_path, text, char_count
            )
            select id, page_id, chunk_index, 'base', heading_path, text, char_count
            from page_chunks;

            drop table page_chunks;
            alter table page_chunks_with_variant rename to page_chunks;

            create index if not exists idx_page_chunks_page_id
              on page_chunks(page_id);
            create index if not exists idx_page_chunks_variant
              on page_chunks(variant);
            """
        )
    finally:
        conn.execute("pragma foreign_keys = on")


def _drop_page_chunk_summary(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("alter table page_chunks drop column summary")
    except sqlite3.OperationalError:
        conn.commit()
        conn.execute("pragma foreign_keys = off")
        try:
            conn.executescript(
                """
                drop table if exists page_chunks_without_summary;

                create table page_chunks_without_summary (
                  id text primary key,
                  page_id text not null references page_metadata(id) on delete cascade,
                  chunk_index integer not null,
                  heading_path text,
                  text text not null,
                  char_count integer not null,
                  unique(page_id, chunk_index)
                );

                insert into page_chunks_without_summary (
                  id, page_id, chunk_index, heading_path, text, char_count
                )
                select id, page_id, chunk_index, heading_path, text, char_count
                from page_chunks;

                drop table page_chunks;
                alter table page_chunks_without_summary rename to page_chunks;

                create index if not exists idx_page_chunks_page_id
                  on page_chunks(page_id);
                """
            )
        finally:
            conn.execute("pragma foreign_keys = on")
