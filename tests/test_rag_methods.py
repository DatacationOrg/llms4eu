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
    assert "sparse_v2" not in names
    assert "qwen4b_hybrid_rerank_v2" not in names
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


def test_agent_uses_the_same_dense_and_sparse_version(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["nemotron"])

    retriever = methods.build_retriever("nemotron_hybrid_agentic")
    hybrid = retriever.base_retriever.base_retriever
    vector, sparse = hybrid.retrievers

    assert retriever.name == "nemotron_hybrid_agentic"
    assert hybrid.name == "nemotron_hybrid"
    assert vector.chunk_version == "v1"
    assert sparse.chunk_version == "v1"


def test_named_reranker_generates_its_own_methods(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["stub"])

    names = methods.list_retrievers()

    assert "stub_rerank_4b" in names
    assert "stub_hybrid_rerank_4b" in names
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


def test_judge_and_geo_agent_methods_are_registered():
    names = methods.list_retrievers()

    assert "qwen_hybrid_agentic_gptoss" in names
    assert "qwen_hybrid_agentic_tools_gemma" in names
    assert "dci_gptoss" in names
    assert "qwen_hybrid_agentic_geo" in names
    assert "qwen_hybrid_agentic_tools_geo" in names
    assert "qwen_hybrid_rerank_geo" in names
    assert methods.method_requires_geo("qwen_hybrid_agentic_geo")
    assert not methods.method_requires_geo("qwen_hybrid_agentic")
    assert methods.method_provider("qwen8b_hybrid_rerank_4b") == "qwen8b"
    assert methods.method_provider("sparse") is None


def test_judge_suffix_selects_that_judges_settings():
    config = methods._judge_config(
        judge="gptoss", method_key="agentic_tools_structured_method"
    )
    assert config["agentic_judge_provider"] == "ollama"
    assert config["agentic_judge_model"] == "gpt-oss:20b"
    assert config["agentic_judge_structured_method"] == "json_schema"
    assert methods._judge_config(judge="gemma")["agentic_judge_reasoning"] is False
    with pytest.raises(ValueError, match="Unknown judge"):
        methods._judge_suffix("claude")


def test_named_judge_supplies_the_model_but_keeps_method_pinned_settings():
    dci = methods._judge_config(
        {"agentic_judge_num_predict": methods.CONFIG["dci_num_predict"]},
        judge="gptoss",
        method_key="dci_structured_method",
    )
    assert dci["agentic_judge_model"] == "gpt-oss:20b"
    assert dci["agentic_judge_num_predict"] == methods.CONFIG["dci_num_predict"]
    # Without a pin the judge's own budget applies.
    plain = methods._judge_config(judge="gptoss")
    assert plain["agentic_judge_num_predict"] == 1024
    # The judge's per-schema method still beats the method's default one.
    tools = methods._judge_config(
        {"agentic_judge_structured_method": "function_calling"},
        judge="gptoss",
        method_key="agentic_tools_structured_method",
    )
    assert tools["agentic_judge_structured_method"] == "json_schema"
    # An explicit reasoning rung is not overridden by the judge's reasoning.
    assert (
        methods._judge_config(reasoning="high", judge="gptoss")[
            "agentic_judge_reasoning"
        ]
        == "high"
    )


def test_plain_tool_agent_has_no_gazetteer_and_the_geo_one_does(monkeypatch):
    monkeypatch.setattr(methods, "enabled_provider_names", lambda: ["qwen"])
    monkeypatch.setitem(methods.CONFIG, "agentic_judge_provider", "azure")
    monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://x.test/openai/v1/")
    monkeypatch.setenv("AZURE_AI_API_KEY", "k")
    monkeypatch.setenv("AZURE_AI_MODEL", "m")

    assert methods.build_retriever("qwen_hybrid_agentic_tools").gazetteer is None
    geo = methods.build_retriever("qwen_hybrid_agentic_tools_geo")
    assert geo.gazetteer is not None and geo.variant == "base"
    assert "qwen_hybrid_rerank_geo_strict" in methods._specs()
    assert methods.method_requires_geo("qwen_hybrid_rerank_geo_strict")
