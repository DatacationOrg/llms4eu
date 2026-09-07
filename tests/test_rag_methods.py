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
    # The model declares 32768; it was previously pinned to 4096 for no reason
    # the model imposes.
    assert indexer.max_seq_length == 32768
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
    monkeypatch.setitem(methods.CONFIG, "agentic_judge_provider", "ollama")

    retriever = methods.build_retriever("nemotron_hybrid_agentic")

    assert retriever.judge is not None
    assert retriever.judge.model_id == methods.CONFIG["agentic_judge_model"]
    assert retriever.judge.method == methods.CONFIG["agentic_judge_structured_method"]


def test_agentic_retrievers_use_the_azure_judge_when_configured(monkeypatch):
    """One provider switch moves the whole agentic family, DCI included."""
    from src.shared.llm import AzureFoundryStructuredLlm

    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["nemotron"])
    monkeypatch.setitem(methods.CONFIG, "agentic_judge_provider", "azure")
    monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://x.test/openai/v1/")
    monkeypatch.setenv("AZURE_AI_API_KEY", "k")
    monkeypatch.setenv("AZURE_AI_MODEL", "DeepSeek-V4-Pro")

    for name in ("nemotron_hybrid_agentic", "nemotron_hybrid_agentic_tools", "dci"):
        judge = methods.build_retriever(name).judge
        assert isinstance(judge, AzureFoundryStructuredLlm), name
        assert judge.model == "DeepSeek-V4-Pro"
    assert (
        methods.build_retriever("dci").judge.max_tokens
        == methods.CONFIG["dci_num_predict"]
    )


def test_v2_agent_uses_v2_dense_and_sparse_representations(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["nemotron"])

    retriever = methods.build_retriever("nemotron_hybrid_agentic_v2")
    hybrid = retriever.base_retriever.base_retriever
    vector, sparse = hybrid.retrievers

    assert retriever.name == "nemotron_hybrid_agentic_v2"
    assert hybrid.name == "nemotron_hybrid_v2"
    assert vector.chunk_version == "v2"
    assert sparse.chunk_version == "v2"


def test_named_reranker_generates_its_own_methods(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    names = methods.list_retrievers()

    assert "stub_rerank_4b" in names
    assert "stub_hybrid_rerank_4b" in names
    assert "stub_hybrid_rerank_4b_v2" in names
    # The default rung keeps its own names, so adding a reranker cannot silently
    # redefine what an already-published `*_rerank` number measured.
    assert "stub_rerank" in names
    assert "stub_hybrid_rerank" in names


def test_named_reranker_swaps_only_the_cross_encoder(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    default = methods.build_retriever("stub_hybrid_rerank")
    scaled = methods.build_retriever("stub_hybrid_rerank_4b")

    assert default.model_name == "Qwen/Qwen3-Reranker-0.6B"
    assert scaled.model_name == "Qwen/Qwen3-Reranker-4B"
    # Everything the rung does not name is inherited, so a score difference
    # between the two rows is the model and not the harness.
    assert scaled.max_length == default.max_length
    assert scaled.candidate_limit == default.candidate_limit
    assert scaled.device == default.device
    assert scaled.base_retriever.name == default.base_retriever.name
    # ...and what it does name is overridden.
    assert scaled.batch_size == 4
    assert scaled.dtype == "bfloat16"
    assert default.dtype is None


def test_unknown_reranker_is_rejected():
    with pytest.raises(ValueError, match="Unknown reranker: 70b"):
        methods._reranker_settings("70b")


def test_reranker_rung_cannot_set_an_unknown_key(monkeypatch):
    """A typo in a rung's config must fail loudly rather than be ignored.

    A silently dropped `max_lenght` would leave the rung reading 2048 tokens while
    the report claimed otherwise, which is indistinguishable from a real result.
    """
    monkeypatch.setitem(methods.CONFIG, "rerankers", {"typo": {"max_lenght": 512}})

    with pytest.raises(ValueError, match="unknown keys: max_lenght"):
        methods._reranker_settings("typo")


def test_reasoning_rungs_generate_parallel_agentic_methods():
    """Every declared rung adds a method; the default names stay untouched."""
    names = set(methods.list_retrievers())
    for rung in methods._reasoning_levels():
        assert f"qwen_hybrid_agentic_{rung}" in names
        assert f"qwen_hybrid_agentic_tools_{rung}" in names
        assert f"dci_{rung}" in names
    # Adding a rung must not remove or rename what was already measured.
    assert "qwen_hybrid_agentic" in names
    assert "qwen_hybrid_agentic_tools" in names
    assert "dci" in names


def test_reasoning_rung_reaches_the_judge_and_default_is_unchanged(monkeypatch):
    """The suffix is not cosmetic: it must change the judge's reasoning effort.

    Reasoning rungs are an Ollama-judge axis; the Azure judge ignores them.
    """
    monkeypatch.setitem(methods.CONFIG, "agentic_judge_provider", "ollama")
    high = methods.build_retriever("qwen_hybrid_agentic_tools_high")
    default = methods.build_retriever("qwen_hybrid_agentic_tools")

    assert high.judge.reasoning == "high"
    assert default.judge.reasoning == methods.CONFIG["agentic_judge_reasoning"]
    assert high.name.endswith("_high")


def test_reasoning_rung_pairs_with_the_plain_reranked_baseline():
    """There is no `*_rerank_high`, so the rung must come off before pairing.

    Without this the diagnostics would look for a method that does not exist and
    report no pairs at all, which reads identically to an agent that changed
    nothing.
    """
    from src.eval.agentic_diagnostics import baseline_for_agent

    assert baseline_for_agent("qwen_hybrid_agentic_high") == "qwen_hybrid_rerank"
    assert baseline_for_agent("qwen_hybrid_agentic_tools_high") == "qwen_hybrid_rerank"
    assert (
        baseline_for_agent("qwen_hybrid_agentic_tools_high_v2")
        == "qwen_hybrid_rerank_v2"
    )
