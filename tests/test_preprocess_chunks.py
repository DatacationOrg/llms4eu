import sqlite3

from src.indexing.chunk_text import (
    MetadataContextChunkText,
    PageChunk,
    TitleHeadingChunkText,
    chunk_text_representation,
)
from src.preprocess.chunks import chunk_markdown, rebuild_page_chunks


def test_chunk_markdown_preserves_heading_path():
    markdown = """
# Castle

Intro text with enough factual detail about the castle and its location.

## History

The castle was first mentioned in 895. It stands above the Sava river.
"""

    chunks = chunk_markdown(markdown, target_chars=120, max_chars=180, min_chars=20)

    assert len(chunks) == 2
    assert chunks[0].heading_path == "Castle"
    assert chunks[1].heading_path == "Castle > History"
    assert "895" in chunks[1].text


def test_chunk_markdown_splits_long_paragraph():
    markdown = "# Title\n\n" + "Sentence about Rajhenburg. " * 80

    chunks = chunk_markdown(markdown, target_chars=300, max_chars=450, min_chars=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 500 for chunk in chunks)


def test_embedding_text_uses_chunk_context():
    text = TitleHeadingChunkText().text_for_embedding(
        PageChunk(
            id="chunk-1",
            page_id="page-1",
            title="Castle",
            heading_path="History",
            text="Full chunk text.",
        )
    )

    assert text == "Castle\nHistory\nFull chunk text."


def test_metadata_context_embedding_text_uses_page_metadata():
    text = MetadataContextChunkText().text_for_embedding(
        PageChunk(
            id="chunk-1",
            page_id="page-1",
            title="Castle",
            heading_path="History",
            text="Full chunk text.",
            source="encyclopedia",
            language="sl",
            page_kind="prose",
        )
    )

    assert text == (
        "Document: Castle\n"
        "Section: History\n"
        "Source language: sl\n"
        "Source collection: encyclopedia\n"
        "Document type: prose\n"
        "Content:\nFull chunk text."
    )
    assert chunk_text_representation("v2").name == "metadata_context_chunk"


def test_rebuild_page_chunks_appends_new_pages_without_dropping_labels(page_db):
    _seed_labeled_chunk(page_db)

    rebuild_page_chunks()

    with sqlite3.connect(page_db) as conn:
        labeled = conn.execute(
            """
            select count(*)
            from eval_relevant_chunks
            where question_id = 'question-existing'
              and chunk_id = 'page-existing:0'
            """
        ).fetchone()[0]
        chunk_ids = [
            row[0]
            for row in conn.execute("select id from page_chunks order by id").fetchall()
        ]

    assert labeled == 1
    assert "page-existing:0" in chunk_ids
    assert "page-new:0" in chunk_ids


def _seed_labeled_chunk(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            insert into page_metadata (id, source, url, fetched_at, title, page_kind)
            values ('page-existing', 'fixture', 'https://example.test/existing',
                    '2026-01-01T00:00:00+00:00', 'Existing', 'ok'),
                   ('page-new', 'fixture', 'https://example.test/new',
                    '2026-01-01T00:00:00+00:00', 'New', 'ok');
            insert into page_markdown_content (page_id, markdown)
            values ('page-existing', '# Existing\n\nExisting page markdown.'),
                   ('page-new', '# New\n\nNew page markdown with enough text.');
            insert into page_chunks (
              id, page_id, chunk_index, heading_path, text, char_count
            ) values (
              'page-existing:0', 'page-existing', 0, 'Existing', 'Existing chunk.', 15
            );
            insert into eval_questions (
              id, question, answer, question_type, question_language
            ) values (
              'question-existing', 'Existing question?', 'Existing answer.',
              'direct_short', 'en'
            );
            insert into eval_relevant_chunks (question_id, chunk_id)
            values ('question-existing', 'page-existing:0');
            """
        )
