import numpy as np

from src.db.dataset import DEFAULT_SIZE, vectors_path
from src.indexing import store
from src.indexing.store import collection_ready, query_chunk_vectors_batch
from src.retrieval.retrievers import sparse as sparse_module
from src.retrieval.retrievers.sparse import SparseRetriever


class StubIndexer:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "castle" in t.lower() else [0.0, 1.0] for t in texts]


def test_vector_search_over_stored_embeddings(monkeypatch, tiny_dataset):
    monkeypatch.setattr(store, "build_indexer", lambda *_: StubIndexer())
    store._index.cache_clear()
    assert collection_ready("qwen")

    monkeypatch.setitem(store.CONFIG, "query_batch_size", 1)
    hits = query_chunk_vectors_batch(
        "qwen", ["castle history", "forest trail"], limit=1
    )
    assert [h.id.split(":")[0] for h in hits[0]] == ["enwiki/Q1"]
    assert [h.id.split(":")[0] for h in hits[1]] == ["enwiki/Q2"]
    assert hits[0][0].text == "Canonical castle chunk."
    store._index.cache_clear()


def test_a_missing_vector_makes_the_index_not_ready(tiny_dataset):
    file = vectors_path("qwen", DEFAULT_SIZE)
    vectors = np.load(file)
    vectors[1] = np.nan
    np.save(file, vectors)
    assert not collection_ready("qwen")
    assert not collection_ready("nemotron")  # no file at all


def test_sparse_indexes_chunk_text_not_titles(tiny_dataset):
    """BM25 must not match on a page title; only dense retrieval sees it."""
    sparse_module._corpus.cache_clear()
    assert SparseRetriever().retrieve("page", limit=2) == []
    assert [
        h.id.split(":")[0] for h in SparseRetriever().retrieve("castle", limit=2)
    ] == ["enwiki/Q1"]
    sparse_module._corpus.cache_clear()
