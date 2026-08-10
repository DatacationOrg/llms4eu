import pytest

from src.retrieval import methods
from src.shared.indexers import (
    Nemotron3EmbedIndexer,
    build_indexer,
    provider_names,
)


def test_retriever_catalog_generates_public_names():
    names = methods.list_retrievers()

    assert "sparse" in names
    assert "qwen_agentic" in names
    assert "qwen_hybrid_agentic" in names
    assert "qwen4b" in names
    assert "qwen4b_hybrid" in names
    assert "qwen4b_rerank" in names
    assert "qwen4b_hybrid_rerank" in names
    assert "nemotron" in names
    assert "nemotron_hybrid" in names
    assert "nemotron_rerank" in names
    assert "nemotron_hybrid_rerank" in names
    assert "nemotron_hybrid_agentic" in names
    assert "sparse_v2" in names
    assert "qwen4b_hybrid_rerank_v2" in names
    assert "nemotron_hybrid_agentic_v2" in names
    assert "qwen4b_chunk" not in names
    assert "sparse_hybrid" not in names


def test_retriever_catalog_has_no_remote_methods():
    """Every registered method must run offline."""
    names = methods.list_retrievers()

    assert not any("cohere" in name for name in names)
    assert not any("embed_v4" in name for name in names)
    assert not any(name.startswith("azure") for name in names)
    assert "azure" not in provider_names()


def test_build_retriever_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown retriever"):
        methods.build_retriever("missing")


def test_nemotron_embedding_provider_uses_retrieval_defaults():
    indexer = build_indexer("nemotron")

    assert "nemotron" in provider_names()
    assert isinstance(indexer, Nemotron3EmbedIndexer)
    assert indexer.model_name == "nvidia/Nemotron-3-Embed-1B-BF16"
    assert indexer.max_seq_length == 4096
    assert indexer.dtype == "bfloat16"
    assert indexer.attn_implementation == "sdpa"


def test_ensure_retriever_ready_accepts_ready_vector_provider(monkeypatch):
    monkeypatch.setattr(methods, "collection_ready", lambda *_: True)
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    methods.ensure_retriever_ready("stub")


def test_ensure_retriever_ready_reports_missing_index(monkeypatch):
    monkeypatch.setattr(methods, "collection_ready", lambda *_: False)
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    with pytest.raises(methods.MissingRetrieverIndexes) as exc:
        methods.ensure_retriever_ready("stub")

    assert exc.value.missing == {"stub": "stub"}
    assert "uv run python -m src.indexing.chunks --method stub" in str(exc.value)


def test_variant_retriever_reports_variant_index_command(monkeypatch):
    monkeypatch.setattr(methods, "collection_ready", lambda *_: False)
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    with pytest.raises(methods.MissingRetrieverIndexes) as exc:
        methods.ensure_retriever_ready("stub_hybrid_rerank", variant="tok512")

    assert exc.value.missing == {"stub_hybrid_rerank": "stub#tok512"}
    assert "--method stub --chunk-variant tok512" in str(exc.value)


def test_variant_retriever_keeps_base_names_and_suffixes_others(monkeypatch):
    monkeypatch.setattr(methods, "collection_ready", lambda *_: True)
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    base = methods.build_retriever("sparse_rerank")
    variant = methods.build_retriever("sparse_rerank", variant="tok512")

    assert base.name == "sparse_rerank"
    assert variant.name == "sparse_rerank_tok512"
    assert variant.base_retriever.chunk_variant == "tok512"


def test_v2_retriever_reports_versioned_index_command(monkeypatch):
    monkeypatch.setattr(methods, "collection_ready", lambda *_: False)
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    with pytest.raises(methods.MissingRetrieverIndexes) as exc:
        methods.ensure_retriever_ready("stub_hybrid_rerank_v2")

    assert exc.value.missing == {"stub_hybrid_rerank_v2": "stub@v2"}
    assert (
        "uv run python -m src.indexing.chunks --method stub --chunk-version v2"
        in str(exc.value)
    )


def test_reranker_uses_discovered_defaults(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    retriever = methods.build_retriever("stub_rerank")

    assert retriever.max_length == 2048
    assert retriever.prompt_name is None
    assert retriever.prompt is None


def test_nemotron_agent_uses_hybrid_reranked_nemotron_chunks(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["nemotron"])

    retriever = methods.build_retriever("nemotron_hybrid_agentic")

    assert retriever.name == "nemotron_hybrid_agentic"
    assert retriever.initial_limit == 10
    assert retriever.limit_step == 5
    assert retriever.base_retriever.name == "nemotron_hybrid_rerank"
    assert retriever.base_retriever.base_retriever.name == "nemotron_hybrid"


def test_qwen_agent_uses_internal_qwen_embedding_index(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["qwen"])

    retriever = methods.build_retriever("qwen_hybrid_agentic")

    assert retriever.name == "qwen_hybrid_agentic"
    assert retriever.base_retriever.name == "qwen_hybrid_rerank"
    hybrid = retriever.base_retriever.base_retriever
    assert hybrid.name == "qwen_hybrid"
    assert hybrid.retrievers[0].provider == "qwen"


def test_agentic_retrievers_use_the_configured_local_judge(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["nemotron"])

    retriever = methods.build_retriever("nemotron_hybrid_agentic")

    assert retriever.judge is not None
    assert retriever.judge.model_id == methods.CONFIG["agentic_judge_model"]
    assert retriever.judge.method == methods.CONFIG["agentic_judge_structured_method"]


def test_v2_agent_uses_v2_dense_and_sparse_representations(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["nemotron"])

    retriever = methods.build_retriever("nemotron_hybrid_agentic_v2")
    hybrid = retriever.base_retriever.base_retriever
    vector, sparse = hybrid.retrievers

    assert retriever.name == "nemotron_hybrid_agentic_v2"
    assert hybrid.name == "nemotron_hybrid_v2"
    assert vector.chunk_version == "v2"
    assert sparse.chunk_version == "v2"
