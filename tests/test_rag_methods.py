import pytest

from src.retrieval import methods
from src.shared.indexers import build_indexer, provider_names


def test_retriever_catalog_generates_public_names():
    """One name per shape; the rest of the catalog is the same generator."""
    names = methods.list_retrievers()

    assert "sparse" in names
    assert "qwen_hybrid_rerank" in names
    assert "nemotron_hybrid_rerank" in names
    assert "sparse_hybrid" not in names


def test_nemotron_embedding_provider_uses_retrieval_defaults():
    indexer = build_indexer("nemotron")

    assert "nemotron" in provider_names()
    assert indexer.name == "nemotron"
    assert indexer.model_name == "nvidia/Nemotron-3-Embed-1B-BF16"
    assert indexer.max_seq_length == 4096
    assert indexer.dtype == "bfloat16"
    assert indexer.attn_implementation == "sdpa"


def test_ensure_retriever_ready_accepts_ready_vector_provider(monkeypatch):
    monkeypatch.setattr(methods, "collection_ready", lambda name: True)
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    methods.ensure_retriever_ready("stub")


def test_ensure_retriever_ready_reports_missing_index(monkeypatch):
    monkeypatch.setattr(methods, "collection_ready", lambda name: False)
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    with pytest.raises(methods.MissingRetrieverIndexes) as exc:
        methods.ensure_retriever_ready("stub")

    assert exc.value.missing == {"stub": "stub"}
    assert "uv run python -m src.indexing.chunks --method stub" in str(exc.value)


def test_reranker_uses_discovered_defaults(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    retriever = methods.build_retriever("stub_rerank")

    assert retriever.max_length == 2048


def test_hybrid_rerank_wraps_a_vector_and_sparse_pair(monkeypatch):
    """The compound name must match what it actually composes."""
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["nemotron"])

    retriever = methods.build_retriever("nemotron_hybrid_rerank")
    hybrid = retriever.base_retriever
    vector, sparse = hybrid.retrievers

    assert retriever.name == "nemotron_hybrid_rerank"
    assert hybrid.name == "nemotron_hybrid"
    assert vector.provider == "nemotron"
    assert sparse.name == "sparse"
