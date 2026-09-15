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
    assert indexer.max_seq_length == 2048
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


def test_indexer_refuses_a_model_below_the_shared_sequence_limit(monkeypatch):
    """A model that truncates earlier than its rivals would skew every comparison."""

    class ShortModel:
        max_seq_length = 256

    monkeypatch.setattr(
        "src.shared.indexers.load_embedder", lambda *_, **__: ShortModel()
    )
    indexer = build_indexer("qwen", {"embedding_max_seq_length": 2048})

    with pytest.raises(ValueError, match="caps out at 256 tokens"):
        indexer._model()


def test_indexer_applies_the_shared_limit_to_a_model_that_can_reach_it(monkeypatch):
    class LongModel:
        max_seq_length = 32768

    monkeypatch.setattr(
        "src.shared.indexers.load_embedder", lambda *_, **__: LongModel()
    )
    model = build_indexer("nemotron", {"embedding_max_seq_length": 2048})._model()

    assert model.max_seq_length == 2048
