from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.shared.env import ROOT, load_local_env, load_yaml
from src.shared.geocode import Coordinates

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


@dataclass(frozen=True)
class PageForGeocoding:
    id: str
    title: str | None
    markdown: str


def raw_pages_db_path() -> Path:
    load_local_env()
    path = ROOT / CONFIG["raw_pages_db"]
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
        chunk_columns = {
            row["name"] for row in conn.execute("pragma table_info(page_chunks)")
        }
        if "summary" in chunk_columns:
            _drop_page_chunk_summary(conn)
        sources = [
            row["source"]
            for row in conn.execute("select distinct source from page_metadata")
        ]
        conn.executemany(
            "insert or ignore into page_sources (source, language) values (?, ?)",
            [(source, CONFIG["default_source_language"]) for source in sources],
        )


def load_pages_for_geocoding(force: bool = False) -> list[PageForGeocoding]:
    sql = """
        select m.id, m.title, c.markdown
        from page_metadata m
        join page_markdown_content c on c.page_id = m.id
        where m.error is null
          and m.page_kind != 'empty'
          and length(trim(c.markdown)) > 0
    """
    if not force:
        sql += " and m.id not in (select page_id from page_locations)"
    with connect_pages() as conn:
        rows = conn.execute(sql).fetchall()
    return [PageForGeocoding(row["id"], row["title"], row["markdown"]) for row in rows]


def upsert_page_location(page_id: str, latitude: float, longitude: float) -> None:
    with connect_pages() as conn:
        conn.execute(
            "insert into page_locations (page_id, latitude, longitude) "
            "values (?, ?, ?) "
            "on conflict(page_id) do update set "
            "latitude = excluded.latitude, longitude = excluded.longitude",
            (page_id, latitude, longitude),
        )


def load_chunk_coordinates(chunk_ids: list[str]) -> dict[str, Coordinates]:
    if not chunk_ids:
        return {}
    placeholders = ", ".join("?" for _ in chunk_ids)
    with connect_pages() as conn:
        rows = conn.execute(
            f"""
            select pc.id as chunk_id, pl.latitude, pl.longitude
            from page_chunks pc
            join page_locations pl on pl.page_id = pc.page_id
            where pc.id in ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
    return {
        row["chunk_id"]: Coordinates(row["latitude"], row["longitude"]) for row in rows
    }


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
