import pytest

from src.retrieval import methods
from src.shared.indexers import AzureEmbeddingIndexer


def test_retriever_catalog_generates_public_names():
    names = methods.list_retrievers()

    assert "sparse" in names
    assert "qwen4b" in names
    assert "qwen4b_hybrid" in names
    assert "qwen4b_rerank" in names
    assert "qwen4b_hybrid_rerank" in names
    assert "qwen4b_chunk" not in names
    assert "sparse_hybrid" not in names


def test_build_retriever_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown retriever"):
        methods.build_retriever("missing")


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
    assert retriever.prompt_name is None
    assert retriever.prompt is None


def test_azure_embedding_batch_size_defaults_to_rpm_friendly_max(monkeypatch):
    monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://example.test")
    monkeypatch.setenv("AZURE_AI_API_KEY", "key")
    monkeypatch.setenv("AZURE_EMBEDDING_MODEL", "model")

    assert AzureEmbeddingIndexer.from_env().batch_size == 96
    assert (
        AzureEmbeddingIndexer.from_env({"azure_embedding_batch_size": 128}).batch_size
        == 96
    )
