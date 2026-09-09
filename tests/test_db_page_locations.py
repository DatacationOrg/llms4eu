"""page_locations storage and the scope -> page set queries behind geo filters."""

import sqlite3

import pytest

from src.db import pages as pages_module
from src.db.pages import (
    PageLocation,
    load_chunk_coordinates,
    load_chunk_locations,
    located_page_share,
    page_ids_in_scope,
    page_locations,
    pages_in_region,
    pages_near,
    primary_locations,
    replace_page_locations,
)
from src.shared.geo_scope import GeoScope
from src.shared.geocode import Coordinates


@pytest.fixture
def db(monkeypatch, tmp_path):
    path = tmp_path / "pages.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table page_metadata (id text primary key, source text not null,
              url text not null, fetched_at text not null, title text);
            create table page_markdown_content (page_id text primary key, markdown text);
            insert into page_metadata (id, source, url, fetched_at, title) values
              ('castle', 's', 'u1', 't', 'Grad Rajhenburg'),
              ('sevnica', 's', 'u2', 't', 'Sevnica'),
              ('person', 's', 'u3', 't', 'A biography');
            """
        )
    monkeypatch.setattr(pages_module, "raw_pages_db_path", lambda: path)
    pages_module.initialize_page_artifacts_db()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            insert into page_chunks (id, page_id, chunk_index, text, char_count) values
              ('castle:0', 'castle', 0, 'c', 1), ('sevnica:0', 'sevnica', 0, 's', 1),
              ('person:0', 'person', 0, 'p', 1);
            """
        )
    replace_page_locations(
        "castle",
        [
            PageLocation(
                "castle",
                "primary",
                "Q12790253",
                "Grad Rajhenburg",
                "Q12790253",
                45.989,
                15.466,
                "point",
                "SI",
                "SI03",
                "SI036",
                "Posavska",
                None,
                1.0,
                "wikidata",
            ),
            PageLocation(
                "castle",
                "mentioned",
                "name:brestanica",
                "Brestanica",
                None,
                45.996,
                15.477,
                "point",
                "SI",
                "SI03",
                "SI036",
                "Posavska",
                None,
                0.8,
                "llm_nominatim",
            ),
        ],
    )
    replace_page_locations(
        "sevnica",
        [
            PageLocation(
                "sevnica",
                "primary",
                "Q15915",
                "Sevnica",
                "Q15915",
                46.009,
                15.304,
                "municipality",
                "SI",
                "SI03",
                "SI036",
                "Posavska",
                "SI-110",
                1.0,
                "wikidata",
            )
        ],
    )
    return path


def test_replace_is_the_whole_set_and_one_primary_per_page(db):
    stored = page_locations(["castle"])["castle"]
    assert [row.role for row in stored] == ["primary", "mentioned"]
    replace_page_locations("castle", stored[:1])
    assert len(page_locations(["castle"])["castle"]) == 1
    with pytest.raises(sqlite3.IntegrityError):
        replace_page_locations(
            "castle",
            [stored[0], PageLocation("castle", "primary", "other", method="manual")],
        )


def test_primary_locations_and_chunk_coordinates(db):
    assert set(primary_locations()) == {"castle", "sevnica"}
    coords = load_chunk_coordinates(["castle:0", "person:0", "sevnica:0"])
    assert set(coords) == {"castle:0", "sevnica:0"}
    assert coords["castle:0"] == Coordinates(45.989, 15.466)


def test_page_ids_in_scope_by_code_radius_and_null_policy(db):
    assert page_ids_in_scope(GeoScope(nuts3="SI036")) == {"castle", "sevnica"}
    assert page_ids_in_scope(GeoScope(country_code="AT")) == frozenset()
    assert page_ids_in_scope(GeoScope()) is None
    near_castle = GeoScope(latitude=45.99, longitude=15.47, radius_km=5)
    assert page_ids_in_scope(near_castle) == {"castle"}
    with_null = GeoScope(nuts3="SI036", include_null=True)
    assert page_ids_in_scope(with_null) == {"castle", "sevnica", "person"}


def test_pages_near_and_in_region(db):
    near = pages_near(Coordinates(45.99, 15.47), radius_km=20)
    assert [(loc.page_id, round(km)) for loc, km in near] == [
        ("castle", 0),
        ("sevnica", 13),
    ]
    assert [loc.page_id for loc in pages_in_region("SI036")] == ["castle", "sevnica"]
    assert [loc.page_id for loc in pages_in_region("si")] == ["castle", "sevnica"]
    assert pages_in_region("SI03X") == []


def test_chunk_locations_and_nuts1_codes(db):
    located = load_chunk_locations(["castle:0", "person:0", "sevnica:0"])
    assert set(located) == {"castle:0", "sevnica:0"}
    assert located["castle:0"].nuts3 == "SI036"
    assert located["castle:0"].coordinates == Coordinates(45.989, 15.466)
    assert [row.page_id for row in pages_in_region("SI0")] == ["castle", "sevnica"]
    assert pages_in_region("Posavje") == []


def test_located_page_share_is_the_selectivity_of_a_level(db):
    assert located_page_share(GeoScope(nuts3="SI036")) == 1.0
    assert located_page_share(GeoScope(country_code="AT")) == 0.0
    assert located_page_share(GeoScope()) is None
    near_castle = GeoScope(
        latitude=45.99, longitude=15.47, radius_km=5, include_null=True
    )
    assert located_page_share(near_castle) == 0.5
