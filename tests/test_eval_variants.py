"""Per-variant eval sets and the deterministic sampling that keeps them cheap."""

from __future__ import annotations

import sqlite3

from src.eval.evaluate import load_eval_rows
from src.eval.generate_dataset import (
    QuestionCandidate,
    _eligible_unprocessed_chunks,
    _stratified_sample,
    insert_questions,
)


def test_stratified_sample_is_deterministic_and_spreads_over_strata():
    chunks = [
        {
            "id": f"page-{source}-{index}",
            "text": "x" * (300 + index * 400),
            "source": source,
            "page_kind": kind,
        }
        for source in ("wikipedia", "castle")
        for kind in ("prose", "listing")
        for index in range(10)
    ]

    picked = _stratified_sample(chunks, 12)
    again = _stratified_sample(chunks, 12)

    assert len(picked) == 12
    assert [chunk["id"] for chunk in picked] == [chunk["id"] for chunk in again]
    assert len({chunk["source"] for chunk in picked}) == 2
    assert len({chunk["page_kind"] for chunk in picked}) == 2


def test_stratified_sample_returns_everything_when_asked_for_more():
    chunks = [
        {"id": "a", "text": "x" * 400, "source": "s", "page_kind": "prose"},
        {"id": "b", "text": "x" * 400, "source": "s", "page_kind": "prose"},
    ]

    assert _stratified_sample(chunks, 5) == chunks


def test_eligible_chunks_and_questions_are_scoped_per_variant(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_two_variant_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    base_chunks = _eligible_unprocessed_chunks(None, "base")
    small_chunks = _eligible_unprocessed_chunks(None, "small")

    assert [chunk["id"] for chunk in base_chunks] == ["page-1:0"]
    assert [chunk["id"] for chunk in small_chunks] == ["page-1:small:0"]

    candidate = QuestionCandidate(
        question="Where does the castle stand?",
        answer="Above the river.",
        question_language="en",
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        insert_questions(conn, "page-1:0", [("direct_short", candidate)], "base")
        insert_questions(conn, "page-1:small:0", [("direct_short", candidate)], "small")

    base_questions, base_relevance = load_eval_rows(variant="base")
    small_questions, small_relevance = load_eval_rows(variant="small")
    all_questions, _ = load_eval_rows()

    assert len(base_questions) == 1
    assert len(small_questions) == 1
    assert len(all_questions) == 2
    assert [row["chunk_id"] for row in base_relevance] == ["page-1:0"]
    assert [row["chunk_id"] for row in small_relevance] == ["page-1:small:0"]


def _write_two_variant_fixture(path):
    text = (
        "The castle stands above the river and was first mentioned in 895, "
        "which makes it one of the oldest fortifications in the valley and a "
        "reference point for dating the settlements around it. Records from the "
        "estate list harvests, repairs and disputes across several centuries, "
        "and the ledgers kept by the monks remain the best surviving account of "
        "daily life on the holding. The eastern wing was rebuilt in 1541 after "
        "an earthquake, and the chapel along the northern range was added in "
        "the same campaign of works."
    )
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key,
              title text,
              source text not null,
              page_kind text not null
            );
            create table page_sources (
              source text primary key,
              language text not null
            );
            create table page_chunks (
              id text primary key,
              page_id text not null,
              chunk_index integer not null,
              variant text not null default 'base',
              heading_path text,
              text text not null,
              char_count integer not null,
              start_char integer,
              end_char integer,
              unique(page_id, variant, chunk_index)
            );
            create table eval_questions (
              id text primary key,
              question text not null,
              answer text not null,
              question_type text not null,
              question_language text not null,
              approved integer not null default 1,
              variant text not null default 'base'
            );
            create table eval_relevant_chunks (
              question_id text not null references eval_questions(id) on delete cascade,
              chunk_id text not null references page_chunks(id) on delete cascade,
              primary key (question_id, chunk_id)
            );

            insert into page_metadata (id, title, source, page_kind)
            values ('page-1', 'Castle', 'fixture', 'prose');
            insert into page_sources (source, language) values ('fixture', 'sl');
            """
        )
        conn.executemany(
            """
            insert into page_chunks (
              id, page_id, chunk_index, variant, heading_path, text, char_count,
              start_char, end_char
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "page-1:0",
                    "page-1",
                    0,
                    "base",
                    "Castle",
                    text,
                    len(text),
                    0,
                    len(text),
                ),
                (
                    "page-1:small:0",
                    "page-1",
                    0,
                    "small",
                    "Castle",
                    text,
                    len(text),
                    0,
                    len(text),
                ),
            ],
        )
