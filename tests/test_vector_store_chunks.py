import sqlite3

import src.vector_store.chunks as chunk_vectors
from src.retrieval.retrievers import sparse as sparse_module
from src.retrieval.retrievers.sparse import SparseRetriever
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


def test_v2_chunk_collection_is_isolated_and_contains_metadata(monkeypatch, tmp_path):
    db_path = tmp_path / "raw_pages.db"
    _write_chunk_fixture(db_path)

    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    monkeypatch.setattr(
        "src.vector_store.chunks.build_indexer", lambda *_: StubIndexer()
    )

    rebuild_chunk_collection("qwen")
    rebuild_chunk_collection("qwen", "v2")

    assert collection_ready("qwen")
    assert collection_ready("qwen", "v2")
    assert chunk_vectors._collection_name("qwen", "v1") != (
        chunk_vectors._collection_name("qwen", "v2")
    )
    hits = query_chunk_vectors("qwen", "castle history", limit=1, version="v2")
    assert [hit.id for hit in hits] == ["chunk-castle"]

    collection = chunk_vectors._client().get_collection(
        chunk_vectors._collection_name("qwen", "v2")
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
        "variant": "base",
    }


def test_enabled_provider_names_comes_from_indexing_config(monkeypatch):
    monkeypatch.setattr(chunk_vectors, "INDEXING_PROVIDERS", ("qwen", "english"))

    assert enabled_provider_names() == ["english", "qwen"]


def test_v2_sparse_indexes_metadata_without_changing_v1(monkeypatch, tmp_path):
    db_path = tmp_path / "raw_pages.db"
    _write_chunk_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    sparse_module._corpus.cache_clear()

    legacy_hits = SparseRetriever(chunk_version="v1").retrieve("fixture", limit=2)
    contextual_hits = SparseRetriever(chunk_version="v2").retrieve(
        "fixture",
        limit=2,
    )

    assert legacy_hits == []
    assert {hit.id for hit in contextual_hits} == {"chunk-castle", "chunk-forest"}
    sparse_module._corpus.cache_clear()


def _write_chunk_fixture(path):
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
              page_id text not null references page_metadata(id) on delete cascade,
              chunk_index integer not null,
              variant text not null default 'base',
              heading_path text,
              text text not null,
              char_count integer not null,
              start_char integer,
              end_char integer,
              unique(page_id, variant, chunk_index)
            );
                 insert into page_sources (source, language) values ('fixture', 'en');
                 insert into page_metadata (id, title, source, page_kind)
                 values ('page-castle', 'Castle Page', 'fixture', 'prose'),
                     ('page-forest', 'Forest Page', 'fixture', 'prose');
            insert into page_chunks (
              id, page_id, chunk_index, heading_path, text, char_count
            )
            values ('chunk-castle', 'page-castle', 0, 'History', 'Canonical castle chunk.', 23),
                   ('chunk-forest', 'page-forest', 0, 'Trail', 'Canonical forest chunk.', 23);
            """
        )
