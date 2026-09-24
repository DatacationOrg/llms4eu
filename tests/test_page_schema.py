import sqlite3

from src.scraping.schema import PageMetadata


def test_page_metadata_fields_match_the_table_columns(page_db):
    """The row shape is declared twice, in pydantic and in sql/raw_pages.sql."""
    with sqlite3.connect(page_db) as conn:
        columns = [row[1] for row in conn.execute("pragma table_info(page_metadata)")]

    assert PageMetadata.db_columns() == columns
