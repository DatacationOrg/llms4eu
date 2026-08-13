import sqlite3

import pytest

from src.db.pages import (
    load_chunk_coordinates,
    load_pages_for_geocoding,
    upsert_page_location,
)
from src.shared.geocode import Coordinates


@pytest.fixture
def pages_db(monkeypatch, tmp_path):
    path = tmp_path / "raw_pages.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key, title text,
              page_kind text not null default 'prose', error text
            );
            create table page_markdown_content (
              page_id text primary key references page_metadata(id),
              markdown text not null
            );
            create table page_chunks (
              id text primary key,
              page_id text not null references page_metadata(id),
              chunk_index integer not null, text text not null
            );
            create table page_locations (
              page_id text primary key references page_metadata(id),
              latitude real not null, longitude real not null
            );

            insert into page_metadata (id, title, page_kind, error) values
              ('page-castle', 'Castle', 'prose', null),
              ('page-located', 'Located', 'prose', null),
              ('page-empty', 'Empty', 'empty', null),
              ('page-errored', 'Errored', 'prose', 'fetch failed');
            insert into page_markdown_content (page_id, markdown) values
              ('page-castle', 'A page about a real castle.'),
              ('page-located', 'Already geocoded.'),
              ('page-empty', ''),
              ('page-errored', 'unreachable');
            insert into page_chunks (id, page_id, chunk_index, text) values
              ('chunk-castle', 'page-castle', 0, 'castle chunk'),
              ('chunk-located', 'page-located', 0, 'located chunk');
            insert into page_locations (page_id, latitude, longitude) values
              ('page-located', 45.9, 15.5);
            """
        )
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: path)


def test_load_pages_for_geocoding_skips_located_empty_and_errored(pages_db):
    assert [page.id for page in load_pages_for_geocoding()] == ["page-castle"]
    assert {page.id for page in load_pages_for_geocoding(force=True)} == {
        "page-castle",
        "page-located",
    }


def test_chunk_coordinates_come_from_the_source_page(pages_db):
    assert load_chunk_coordinates(["chunk-castle", "chunk-located"]) == {
        "chunk-located": Coordinates(45.9, 15.5)
    }

    upsert_page_location("page-castle", 45.98, 15.46)
    assert load_chunk_coordinates(["chunk-castle"])["chunk-castle"] == Coordinates(
        45.98, 15.46
    )

    upsert_page_location("page-castle", 46.0, 15.0)
    assert load_chunk_coordinates(["chunk-castle"])["chunk-castle"] == Coordinates(
        46.0, 15.0
    )
