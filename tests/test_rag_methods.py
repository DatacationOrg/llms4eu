import pytest

from src.retrieval import methods
from src.retrieval.retrievers.sparse import SparseRetriever


class StubProvider:
    ensured = False

    def build_retriever(self, name):
        return SparseRetriever(name=name)

    def ensure_ready(self):
        self.ensured = True


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


def test_ensure_retriever_ready_calls_vector_provider(monkeypatch):
    provider = StubProvider()
    monkeypatch.setattr(methods, "build_provider", lambda name: provider)
    monkeypatch.setattr(methods, "provider_names", lambda: ["stub"])

    methods.ensure_retriever_ready("stub")

    assert provider.ensured


def test_reranker_uses_discovered_defaults(monkeypatch):
    monkeypatch.setattr(methods, "build_provider", lambda name: StubProvider())
    monkeypatch.setattr(methods, "provider_names", lambda: ["stub"])

    retriever = methods.build_retriever("stub_rerank")

    assert retriever.max_length == 2048
    assert retriever.prompt_name is None
    assert retriever.prompt is None
