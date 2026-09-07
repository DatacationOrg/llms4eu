import sqlite3

import pytest

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
    }


def test_chunk_variants_get_isolated_collections(monkeypatch, tmp_path):
    db_path = tmp_path / "raw_pages.db"
    _write_chunk_fixture(db_path)
    with sqlite3.connect(db_path) as conn:
        # Same page, a different cut: ids differ only by the variant segment.
        conn.execute(
            """
            insert into page_chunks (
              id, page_id, chunk_index, variant, heading_path, text, char_count
            ) values ('page-castle:tok256:0', 'page-castle', 0, 'tok256',
                      'History', 'A castle chunk cut smaller.', 27)
            """
        )

    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    monkeypatch.setattr(
        "src.vector_store.chunks.build_indexer", lambda *_: StubIndexer()
    )

    rebuild_chunk_collection("qwen")
    rebuild_chunk_collection("qwen", "v1", "tok256")

    # The base collection keeps its historical name; the variant gets its own.
    assert chunk_vectors._collection_name("qwen", "v1") != (
        chunk_vectors._collection_name("qwen", "v1", "tok256")
    )
    assert collection_ready("qwen")
    assert collection_ready("qwen", "v1", "tok256")

    # Readiness is scoped to the variant, so a global chunk count cannot make
    # one collection look stale because another variant exists.
    assert chunk_vectors._chunk_count("base") == 2
    assert chunk_vectors._chunk_count("tok256") == 1

    hits = query_chunk_vectors("qwen", "castle history", limit=5, variant="tok256")
    assert [hit.id for hit in hits] == ["page-castle:tok256:0"]


def test_unbuilt_variant_fails_with_a_build_command(monkeypatch, tmp_path):
    db_path = tmp_path / "raw_pages.db"
    _write_chunk_fixture(db_path)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    monkeypatch.setattr(
        "src.vector_store.chunks.build_indexer", lambda *_: StubIndexer()
    )

    with pytest.raises(RuntimeError, match="No missing chunks in SQLite"):
        rebuild_chunk_collection("qwen", "v1", "missing")


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
              page_kind text not null,
              language text
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


def test_a_collection_built_at_another_sequence_length_is_not_ready(
    monkeypatch, tmp_path
):
    """A length change alters the vectors but not the chunk count.

    Nothing in a count comparison can see that `page_chunks_qwen_chunk` holds
    512-length vectors while the provider is now configured for 32768, so the
    length has to be recorded and checked.
    """
    db_path = tmp_path / "raw_pages.db"
    _write_chunk_fixture(db_path)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    class ShortIndexer(StubIndexer):
        max_seq_length = 512

    class LongIndexer(StubIndexer):
        max_seq_length = 32768

    monkeypatch.setattr(
        "src.vector_store.chunks.build_indexer", lambda *_: ShortIndexer()
    )
    rebuild_chunk_collection("qwen")
    assert collection_ready("qwen")

    # Same chunks, same count, longer read length: the stored vectors are stale.
    monkeypatch.setattr(
        "src.vector_store.chunks.build_indexer", lambda *_: LongIndexer()
    )
    assert not collection_ready("qwen")

    rebuild_chunk_collection("qwen")
    assert collection_ready("qwen")


def test_a_collection_with_no_recorded_length_is_treated_as_stale(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "raw_pages.db"
    _write_chunk_fixture(db_path)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    # No max_seq_length attribute at all: nothing is written to metadata.
    monkeypatch.setattr(
        "src.vector_store.chunks.build_indexer", lambda *_: StubIndexer()
    )
    rebuild_chunk_collection("qwen")

    class LongIndexer(StubIndexer):
        max_seq_length = 32768

    monkeypatch.setattr(
        "src.vector_store.chunks.build_indexer", lambda *_: LongIndexer()
    )

    # Provenance unknown beats serving vectors of unknown origin.
    assert not collection_ready("qwen")
