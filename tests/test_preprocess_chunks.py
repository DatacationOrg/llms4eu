from src.indexing.chunk_text import PageChunk, TitleHeadingChunkText
from src.preprocess.chunks import chunk_markdown


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
