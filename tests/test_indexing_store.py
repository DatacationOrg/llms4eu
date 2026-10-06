import sqlite3

import src.indexing.store as chunk_vectors
from src.retrieval.retrievers import sparse as sparse_module
from src.retrieval.retrievers.sparse import SparseRetriever
from src.indexing.store import (
    collection_ready,
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


def test_chunk_collection_rebuild_readiness_and_query(monkeypatch, page_db):
    _seed_chunks(page_db)
    monkeypatch.setattr("src.indexing.store.build_indexer", lambda *_: StubIndexer())

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

    with sqlite3.connect(page_db) as conn:
        conn.execute(
            "update page_chunks set text = 'Rechunked.' where id = 'chunk-castle'"
        )
    assert not collection_ready("qwen")
    rebuild_chunk_collection("qwen")

    monkeypatch.setitem(chunk_vectors.CONFIG, "query_batch_size", 1)
    small_batch_hits = query_chunk_vectors_batch(
        "qwen",
        ["castle history", "forest trail"],
        limit=1,
    )
    assert [hit.id for hit in small_batch_hits[0]] == ["chunk-castle"]
    assert [hit.id for hit in small_batch_hits[1]] == ["chunk-forest"]


def test_chunk_collection_stores_page_metadata(monkeypatch, page_db):
    _seed_chunks(page_db)
    monkeypatch.setattr("src.indexing.store.build_indexer", lambda *_: StubIndexer())

    rebuild_chunk_collection("qwen")

    assert collection_ready("qwen")
    hits = query_chunk_vectors("qwen", "castle history", limit=1)
    assert [hit.id for hit in hits] == ["chunk-castle"]

    collection = chunk_vectors._client().get_collection(
        chunk_vectors._collection_name("qwen")
    )
    stored = collection.get(ids=["chunk-castle"], include=["metadatas"])
    assert stored["metadatas"][0] == {
        "chunk_index": 0,
        "id": "chunk-castle",
        "language": "en",
        "page_id": "page-castle",
        "page_kind": "prose",
        "source": "fixture",
        "title": "Castle Page",
    }


def test_sparse_indexes_chunk_text_not_page_metadata(page_db):
    """BM25 must not match on a source name; only dense retrieval sees metadata."""
    _seed_chunks(page_db)
    sparse_module._corpus.cache_clear()

    assert SparseRetriever().retrieve("fixture", limit=2) == []
    assert {hit.id for hit in SparseRetriever().retrieve("castle", limit=2)} == {
        "chunk-castle"
    }
    sparse_module._corpus.cache_clear()


def _seed_chunks(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            insert into page_sources (source, language) values ('fixture', 'en');
            insert into page_metadata (id, source, url, fetched_at, title, page_kind)
            values ('page-castle', 'fixture', 'https://example.test/castle',
                    '2026-01-01T00:00:00+00:00', 'Castle Page', 'prose'),
                   ('page-forest', 'fixture', 'https://example.test/forest',
                    '2026-01-01T00:00:00+00:00', 'Forest Page', 'prose');
            insert into page_chunks (
              id, page_id, chunk_index, heading_path, text, char_count
            )
            values ('chunk-castle', 'page-castle', 0, 'History', 'Canonical castle chunk.', 23),
                   ('chunk-forest', 'page-forest', 0, 'Trail', 'Canonical forest chunk.', 23);
            """
        )
