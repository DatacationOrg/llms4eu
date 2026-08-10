"""Contract tests for chunking variants and the library-backed chunkers.

The ablation rests on four invariants: the frozen `legacy` strategy still
produces exactly the chunks the approved labels point at, variants never
collide, spans stay ordered so they can be projected between variants, and a
chunk never exceeds the size it was cut for.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.eval.spans import project_relevance
from src.preprocess.chunkers import build_chunker
from src.preprocess.chunks import (
    backfill_chunk_spans,
    base_variant_chunker,
    chunk_id,
    rebuild_page_chunks,
)

MARKDOWN = """# Castle

The castle stands above the river and guards the valley below it, looking out
over the terraced vineyards that climb the opposite bank. Travellers approaching
from the south see the towers long before they reach the bridge, which is why the
site was chosen for a fortification in the first place.

It was first mentioned in 895 and rebuilt several times after that, most
thoroughly after the earthquake that brought down the eastern wing. The current
outline dates from the sixteenth century, when the courtyard was enclosed and the
chapel was added along the northern range.

## History

Monks lived here for centuries and kept detailed records of the estate, listing
every harvest, every repair and every dispute with the neighbouring parish. Those
ledgers are the reason so much is known about daily life on the estate compared
with other holdings in the region.

The last owners left in 1941 when the estate was seized, and the buildings served
several institutional purposes before restoration began. Much of the interior
detail had to be reconstructed from the photographs taken during the survey of
1936, which remain the best record of the original rooms.

## Visiting

Open Tuesday to Sunday, closed on Mondays throughout the winter season, with
guided tours starting on the hour from the gatehouse. Tickets cover the chapel,
the cellars and the exhibition in the eastern wing, and the last tour leaves
ninety minutes before closing time.
"""

PAGE = {
    "id": "page-castle",
    "title": "Castle",
    "source": "fixture",
    "page_kind": "prose",
    "language": "sl",
}


def test_chunk_ids_keep_the_historical_form_for_the_base_variant():
    assert chunk_id("page-1", "base", 3) == "page-1:3"
    assert chunk_id("page-1", "tok512", 3) == "page-1:tok512:3"


def test_unknown_strategy_and_unit_are_rejected():
    with pytest.raises(ValueError):
        build_chunker("nonsense", size=500, overlap=0)
    with pytest.raises(ValueError):
        build_chunker("recursive", size=500, overlap=0, unit="furlongs")


def test_token_sizing_needs_a_provider():
    with pytest.raises(ValueError):
        build_chunker("recursive", size=500, overlap=0, unit="tokens")


def test_overlap_must_stay_below_the_chunk_size():
    with pytest.raises(ValueError):
        build_chunker("recursive", size=100, overlap=100)


@pytest.mark.parametrize("strategy", ["recursive", "markdown", "legacy"])
def test_every_strategy_respects_its_chunk_size(strategy):
    chunker = build_chunker(strategy, size=300, overlap=0)

    chunks = chunker.split(MARKDOWN, PAGE)

    assert chunks
    # `legacy` allows one long paragraph to run to its derived ceiling; the
    # library strategies stay at or below the requested size.
    ceiling = 300 * 13 // 9 if strategy == "legacy" else 300
    assert all(len(chunk.text) <= ceiling for chunk in chunks)


@pytest.mark.parametrize("strategy", ["recursive", "markdown", "legacy"])
def test_every_strategy_returns_ordered_spans_inside_the_page(strategy):
    chunker = build_chunker(strategy, size=300, overlap=0)

    chunks = chunker.split(MARKDOWN, PAGE)

    for chunk in chunks:
        assert 0 <= chunk.start_char <= chunk.end_char <= len(MARKDOWN)
    starts = [chunk.start_char for chunk in chunks]
    assert starts == sorted(starts)


def test_smaller_size_produces_more_chunks():
    big = build_chunker("recursive", size=600, overlap=0).split(MARKDOWN, PAGE)
    small = build_chunker("recursive", size=200, overlap=0).split(MARKDOWN, PAGE)

    assert len(small) > len(big)


def test_overlap_repeats_text_between_consecutive_chunks():
    without = build_chunker("recursive", size=300, overlap=0).split(MARKDOWN, PAGE)
    with_overlap = build_chunker("recursive", size=300, overlap=100).split(
        MARKDOWN, PAGE
    )

    assert len(with_overlap) >= len(without)
    assert sum(len(c.text) for c in with_overlap) > sum(len(c.text) for c in without)


def test_markdown_strategy_keeps_the_heading_path():
    chunks = build_chunker("markdown", size=300, overlap=0).split(MARKDOWN, PAGE)

    paths = {chunk.heading_path for chunk in chunks}
    assert any(path and "History" in path for path in paths)


def test_variants_coexist_without_colliding(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks()
    rebuild_page_chunks("small", build_chunker("recursive", size=200, overlap=0))

    with sqlite3.connect(db_path) as conn:
        counts = dict(
            conn.execute("select variant, count(*) from page_chunks group by 1")
        )
        base_ids = [
            row[0]
            for row in conn.execute("select id from page_chunks where variant = 'base'")
        ]

    assert counts["base"] < counts["small"], "smaller size must cut more chunks"
    assert all(":small:" not in identifier for identifier in base_ids)


def test_rebuild_is_append_only_per_variant(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks()
    with sqlite3.connect(db_path) as conn:
        first = conn.execute("select count(*) from page_chunks").fetchone()[0]

    rebuild_page_chunks()
    with sqlite3.connect(db_path) as conn:
        second = conn.execute("select count(*) from page_chunks").fetchone()[0]

    assert first == second


def test_clean_recuts_a_variant(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks("small", build_chunker("recursive", size=600, overlap=0))
    with sqlite3.connect(db_path) as conn:
        wide = conn.execute(
            "select count(*) from page_chunks where variant = 'small'"
        ).fetchone()[0]

    rebuild_page_chunks(
        "small", build_chunker("recursive", size=200, overlap=0), clean=True
    )
    with sqlite3.connect(db_path) as conn:
        narrow = conn.execute(
            "select count(*) from page_chunks where variant = 'small'"
        ).fetchone()[0]

    assert narrow > wide


def test_backfill_fills_spans_without_touching_text(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks()
    with sqlite3.connect(db_path) as conn:
        conn.execute("update page_chunks set start_char = null, end_char = null")
        before = [
            row[0] for row in conn.execute("select text from page_chunks order by id")
        ]

    backfill_chunk_spans()

    with sqlite3.connect(db_path) as conn:
        missing = conn.execute(
            "select count(*) from page_chunks where start_char is null"
        ).fetchone()[0]
        after = [
            row[0] for row in conn.execute("select text from page_chunks order by id")
        ]

    assert missing == 0
    assert before == after


def test_projection_maps_gold_spans_onto_a_finer_variant(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks()
    rebuild_page_chunks("small", build_chunker("recursive", size=200, overlap=0))

    with sqlite3.connect(db_path) as conn:
        gold = conn.execute(
            "select id from page_chunks where variant = 'base' order by chunk_index"
        ).fetchone()[0]

    rows = [{"question_id": "q1", "chunk_id": gold}]
    projected = project_relevance(rows, "base", "small")

    assert projected, "a gold span must map onto at least one finer chunk"
    assert all(row["chunk_id"].startswith("page-castle:small:") for row in projected)
    assert project_relevance(rows, "base", "base") == rows


def test_base_variant_chunker_is_the_frozen_legacy_strategy():
    assert base_variant_chunker().name == "legacy"


def _write_page_fixture(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key,
              title text,
              source text not null,
              page_kind text not null
            );
            create table page_markdown_content (
              page_id text primary key references page_metadata(id) on delete cascade,
              markdown text not null
            );
            create table page_sources (
              source text primary key,
              language text not null
            );
            create table page_chunks (
              id text primary key,
              page_id text not null references page_metadata(id) on delete cascade,
              chunk_index integer not null,
              heading_path text,
              text text not null,
              char_count integer not null,
              unique(page_id, chunk_index)
            );
            create table eval_questions (
              id text primary key,
              question text not null,
              answer text not null,
              question_type text not null,
              question_language text not null,
              approved integer not null default 1
            );
            create table eval_relevant_chunks (
              question_id text not null references eval_questions(id) on delete cascade,
              chunk_id text not null references page_chunks(id) on delete cascade,
              primary key (question_id, chunk_id)
            );

            insert into page_metadata (id, title, source, page_kind)
            values ('page-castle', 'Castle', 'fixture', 'prose');
            insert into page_sources (source, language) values ('fixture', 'sl');
            """
        )
        conn.execute(
            "insert into page_markdown_content (page_id, markdown) values (?, ?)",
            ("page-castle", MARKDOWN),
        )
