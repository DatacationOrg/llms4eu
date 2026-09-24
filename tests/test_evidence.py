import sqlite3

import pytest

from src.eval.evidence import matching_chunks, relabel
from src.preprocess.chunker import rebuild_page_chunks

QUOTE = "The castle was first mentioned in 1256."


def test_quote_split_by_a_boundary_labels_both_sides():
    chunks = {
        "p:0": "Intro. The castle was first mentioned in 1256.",
        "p:1": "It was rebuilt after the fire of 1590.   Later text.",
        "p:2": "Unrelated.",
    }
    quote = (
        "The castle was first mentioned in 1256. It was rebuilt after the fire of 1590."
    )

    assert matching_chunks(quote, chunks) == ["p:0", "p:1"]


def test_rechunk_refuses_until_every_question_is_anchored(page_db):
    _seed(page_db, with_evidence=False)

    with pytest.raises(SystemExit, match="no evidence quote"):
        rebuild_page_chunks(rechunk_all=True)


def test_rechunk_then_relabel_moves_labels_to_the_new_chunks(page_db):
    _seed(page_db, with_evidence=True)

    rebuild_page_chunks(rechunk_all=True)
    relabel()

    with sqlite3.connect(page_db) as conn:
        labels = conn.execute(
            "select c.text from eval_relevant_chunks r join page_chunks c on c.id = r.chunk_id"
        ).fetchall()
    assert len(labels) == 1
    assert QUOTE in labels[0][0]


def _seed(path, with_evidence):
    markdown = f"# Castle\n\n{QUOTE} " + "Filler sentence here. " * 20
    with sqlite3.connect(path) as conn:
        conn.execute(
            "insert into page_metadata (id, source, url, fetched_at) "
            "values ('page', 'fixture', 'https://example.test', '2026-01-01')"
        )
        conn.execute(
            "insert into page_markdown_content values ('page', ?)", (markdown,)
        )
        conn.execute(
            "insert into page_chunks values ('page:0', 'page', 0, 'Castle', 'old', 3)"
        )
        conn.execute(
            "insert into eval_questions (id, question, answer, question_type, "
            "question_language) values ('q', 'When?', '1256', 'same_language', 'en')"
        )
        conn.execute("insert into eval_relevant_chunks values ('q', 'page:0')")
        if with_evidence:
            conn.execute("insert into eval_evidence values ('q', 'page', ?)", (QUOTE,))
