import sqlite3

import src.vector_store.chunks as chunk_vectors
from src.vector_store.chunks import (
    collection_ready,
    enabled_provider_names,
    query_chunk_vectors,
    query_chunk_vectors_batch,
    rebuild_chunk_collection,
)


class StubIndexer:
    name = "qwen"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "castle" in text.lower() else [0.0, 1.0] for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "castle" in text.lower() else [0.0, 1.0] for text in texts
        ]


def test_chunk_collection_rebuild_readiness_and_query(monkeypatch, tmp_path):
    db_path = tmp_path / "raw_pages.db"
    _write_chunk_fixture(db_path)

    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    monkeypatch.setattr(
        "src.vector_store.chunks.build_indexer", lambda *_: StubIndexer()
    )

    assert not collection_ready("qwen")

    rebuild_chunk_collection("qwen")

    assert collection_ready("qwen")
    hits = query_chunk_vectors("qwen", "castle history", limit=1)
    assert [hit.id for hit in hits] == ["chunk-castle"]
    assert hits[0].text == "Canonical castle chunk."

    batch_hits = query_chunk_vectors_batch(
        "qwen",
        ["castle history", "forest trail"],
        limit=1,
    )
    assert [hit.id for hit in batch_hits[0]] == ["chunk-castle"]
    assert [hit.id for hit in batch_hits[1]] == ["chunk-forest"]

    monkeypatch.setitem(chunk_vectors.CONFIG, "query_batch_size", 1)
    small_batch_hits = query_chunk_vectors_batch(
        "qwen",
        ["castle history", "forest trail"],
        limit=1,
    )
    assert [hit.id for hit in small_batch_hits[0]] == ["chunk-castle"]
    assert [hit.id for hit in small_batch_hits[1]] == ["chunk-forest"]


def test_enabled_provider_names_comes_from_indexing_config(monkeypatch):
    monkeypatch.setattr(chunk_vectors, "INDEXING_PROVIDERS", ("qwen", "english"))

    assert enabled_provider_names() == ["english", "qwen"]


def _write_chunk_fixture(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key,
              title text,
              source text not null
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
            insert into page_metadata (id, title, source)
            values ('page-castle', 'Castle Page', 'fixture'),
                   ('page-forest', 'Forest Page', 'fixture');
            insert into page_chunks (
              id, page_id, chunk_index, heading_path, text, char_count
            )
            values ('chunk-castle', 'page-castle', 0, 'History', 'Canonical castle chunk.', 23),
                   ('chunk-forest', 'page-forest', 0, 'Trail', 'Canonical forest chunk.', 23);
            """
        )
