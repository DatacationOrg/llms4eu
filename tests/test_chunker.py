import sqlite3

from src.indexing.documents import PageChunk, text_for_embedding
from src.preprocess.chunker import chunk_markdown, rebuild_page_chunks


def test_chunk_markdown_preserves_heading_path():
    markdown = """
# Castle

Intro text with enough factual detail about the castle and its location.

## History

The castle was first mentioned in 895. It stands above the Sava river.
"""

    chunks = chunk_markdown(markdown, size=120, overlap=0, min_chars=20)

    assert len(chunks) == 2
    assert chunks[0].heading_path == "Castle"
    assert chunks[1].heading_path == "Castle > History"
    assert "895" in chunks[1].text


def test_chunk_markdown_splits_long_paragraph():
    markdown = "# Title\n\n" + "Sentence about Rajhenburg. " * 80

    chunks = chunk_markdown(markdown, size=300, overlap=0, min_chars=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_chunk_markdown_overlaps_trailing_paragraphs():
    paragraphs = [f"Paragraph {i} about the castle." for i in range(6)]
    markdown = "# T\n\n" + "\n\n".join(paragraphs)

    chunks = chunk_markdown(markdown, size=90, overlap=40, min_chars=0)

    assert len(chunks) > 1
    for before, after in zip(chunks, chunks[1:]):
        assert after.text.split("\n\n")[0] == before.text.split("\n\n")[-1]
        assert len(after.text) <= 90


def test_chunk_markdown_merges_short_chunks_instead_of_dropping_them():
    markdown = "# Castle\n\n" + "Long history sentence. " * 10 + "\n\n## Note\n\nShort."

    chunks = chunk_markdown(markdown, size=1000, overlap=0, min_chars=50)

    assert len(chunks) == 1
    assert chunks[0].text.endswith("Castle > Note\n\nShort.")


def test_embedding_text_is_title_breadcrumbs_then_chunk():
    text = text_for_embedding(
        PageChunk(
            id="chunk-1",
            page_id="page-1",
            title="Castle",
            heading_path="History",
            text="Full chunk text.",
        )
    )

    assert text == "Castle\nHistory\nFull chunk text."


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
