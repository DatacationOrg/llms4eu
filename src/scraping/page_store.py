import sqlite3
from pathlib import Path

from src.scraping.page_extract import FetchResult
from src.shared.env import ROOT
from src.shared.schema import PageMetadata


def initialize_raw_pages_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = (ROOT / "sql" / "raw_pages.sql").read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.execute("pragma foreign_keys = on")
        conn.executescript(schema)


def record_source_languages(db_path: Path, languages: dict[str, str]) -> None:
    """Store each source's language from the seed file.

    `page_sources` is otherwise filled with the configured default when the
    eval schema is initialised, which is right for the Slovenian corpus and
    wrong for every other country's seed.
    """
    if not languages:
        return
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "create table if not exists page_sources "
            "(source text primary key, language text not null)"
        )
        conn.executemany(
            "insert into page_sources (source, language) values (?, ?) "
            "on conflict(source) do update set language = excluded.language",
            sorted(languages.items()),
        )


def upsert_fetch_result(db_path: Path, result: FetchResult) -> None:
    metadata = result.metadata
    columns = PageMetadata.db_columns()
    placeholders = ", ".join(["?"] * len(columns))
    updates = ", ".join(
        f"{column} = excluded.{column}" for column in columns if column != "id"
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("pragma foreign_keys = on")
        conn.execute(
            f"insert into page_metadata ({', '.join(columns)}) values ({placeholders}) "
            f"on conflict(id) do update set {updates}",
            metadata.db_values(),
        )
        conn.execute(
            "delete from page_markdown_content where page_id = ?", (metadata.id,)
        )
        if result.markdown_content:
            conn.execute(
                "insert into page_markdown_content (page_id, markdown) values (?, ?)",
                (result.markdown_content.page_id, result.markdown_content.markdown),
            )
