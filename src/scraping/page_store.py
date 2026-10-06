from src.db.pages import connect_pages
from src.scraping.page_extract import FetchResult
from src.scraping.schema import PageMetadata
from src.shared.env import ROOT


def initialize_raw_pages_db() -> None:
    schema = (ROOT / "sql" / "raw_pages.sql").read_text(encoding="utf-8")
    with connect_pages() as conn:
        conn.executescript(schema)


def _upsert_metadata_sql() -> str:
    columns = PageMetadata.db_columns()
    # Column names come from the model's own fields; refuse anything that is not a
    # plain identifier before it goes into SQL text.
    if not all(column.isidentifier() for column in columns):
        raise ValueError(f"Unsafe page_metadata column in {columns}")
    names = ", ".join(columns)
    marks = ", ".join(["?"] * len(columns))
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "id")
    return " ".join(
        [
            "insert into page_metadata (",
            names,
            ") values (",
            marks,
            ") on conflict(id) do update set",
            updates,
        ]
    )


def upsert_fetch_result(result: FetchResult) -> None:
    metadata = result.metadata
    with connect_pages() as conn:
        conn.execute(_upsert_metadata_sql(), metadata.db_values())
        conn.execute(
            "delete from page_markdown_content where page_id = ?", (metadata.id,)
        )
        if result.markdown_content:
            conn.execute(
                "insert into page_markdown_content (page_id, markdown) values (?, ?)",
                (result.markdown_content.page_id, result.markdown_content.markdown),
            )
