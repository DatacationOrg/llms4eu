from src.db.pages import connect_pages
from src.scraping.page_extract import FetchResult
from src.scraping.schema import PageMetadata
from src.shared.env import ROOT


def initialize_raw_pages_db() -> None:
    schema = (ROOT / "sql" / "raw_pages.sql").read_text(encoding="utf-8")
    with connect_pages() as conn:
        conn.executescript(schema)


def upsert_fetch_result(result: FetchResult) -> None:
    metadata = result.metadata
    columns = PageMetadata.db_columns()
    placeholders = ", ".join(["?"] * len(columns))
    updates = ", ".join(
        f"{column} = excluded.{column}" for column in columns if column != "id"
    )

    with connect_pages() as conn:
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
