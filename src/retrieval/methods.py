from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from src.db.pages import load_chunk_locations, located_page_share, page_ids_in_scope
from src.indexing.chunk_text import LEGACY_CHUNK_VERSION
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.retrieval.base import Retriever
from src.retrieval.retrievers.agentic import AgenticRetriever, default_judge
from src.retrieval.retrievers.agentic_tools import AgenticToolRetriever
from src.retrieval.retrievers.dci import DirectCorpusRetriever
from src.retrieval.retrievers.fusion import WeightedScoreFusionRetriever
from src.retrieval.retrievers.geo import GeoScopedRetriever
from src.retrieval.retrievers.rerank import CrossEncoderRerankRetriever
from src.retrieval.retrievers.sparse import SparseRetriever
from src.retrieval.retrievers.vector_chunks import VectorChunkRetriever
from src.shared.env import load_yaml
from src.shared.geo_resolver import Gazetteer, LazyResolver, LlmGazetteerResolver
from src.shared.geo_scope import GeoScope
from src.shared.geocode import NominatimGeocoder
from src.shared.wikidata import WikidataClient
from src.vector_store.chunks import (
    collection_has_geo_metadata,
    collection_ready,
    enabled_provider_names,
)

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


@dataclass(frozen=True)
class RetrieverSpec:
    """Lazy catalog entry for one public retrieval method."""

    name: str
    build: Callable[[], Retriever]
    provider: str | None = None
    chunk_version: str = LEGACY_CHUNK_VERSION
    variant: str = BASE_CHUNK_VARIANT
    # Geo methods filter on metadata keys a pre-geo collection does not carry.
    requires_geo: bool = False

    @property
    def requires_index(self) -> bool:
        return self.provider is not None


class MissingRetrieverIndexes(RuntimeError):
    def __init__(self, missing: dict[str, str]) -> None:
        self.missing = missing
        requirements = sorted(set(missing.values()))
        commands = "\n".join(
            _index_build_command(requirement) for requirement in requirements
        )
        methods = ", ".join(sorted(missing))
        super().__init__(
            "Missing vector indexes for retrieval methods: "
            f"{methods}\nBuild them first:\n{commands}"
        )


def list_retrievers(variant: str = BASE_CHUNK_VARIANT) -> list[str]:
    return sorted(_specs(variant))


def build_retriever(name: str, variant: str = BASE_CHUNK_VARIANT) -> Retriever:
    # The variant is passed in rather than encoded in the name, so a chunking
    # ablation reuses the same method list and the report can put one method's
    # score for every variant in one row.
    specs = _specs(variant)
    if name not in specs:
        raise ValueError(f"Unknown retriever: {name}")
    return specs[name].build()


def ensure_retriever_ready(name: str, variant: str = BASE_CHUNK_VARIANT) -> None:
    ensure_retrievers_ready([name], variant)


def ensure_retrievers_ready(
    names: list[str],
    variant: str = BASE_CHUNK_VARIANT,
) -> None:
    missing = missing_retriever_indexes(names, variant)
    if missing:
        raise MissingRetrieverIndexes(missing)


def missing_retriever_indexes(
    names: list[str],
    variant: str = BASE_CHUNK_VARIANT,
) -> dict[str, str]:
    specs = _specs(variant)
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f"Unknown retriever: {', '.join(unknown)}")

    missing = {}
    for name in names:
        spec = specs[name]
        if spec.provider is None:
            continue
        ready = (
            collection_has_geo_metadata(spec.provider, spec.chunk_version, spec.variant)
            if spec.requires_geo
            else _collection_ready(spec.provider, spec.chunk_version, spec.variant)
        )
        if not ready:
            missing[name] = _index_requirement(
                spec.provider, spec.chunk_version, spec.variant
            )
    return missing


def method_provider(name: str) -> str | None:
    """The embedding provider a method name carries, longest match first.

    None for methods that load no embedder (sparse, dci). Used to group methods
    so a comparison never holds two 15 GB embedders resident at once.
    """
    for provider in sorted(enabled_provider_names(), key=len, reverse=True):
        if name == provider or name.startswith(f"{provider}_"):
            return provider
    return None


def method_requires_geo(name: str, variant: str = BASE_CHUNK_VARIANT) -> bool:
    return _specs(variant)[name].requires_geo


def _specs(variant: str = BASE_CHUNK_VARIANT) -> dict[str, RetrieverSpec]:
    specs = {
        "sparse": RetrieverSpec(
            "sparse",
            lambda: _sparse(LEGACY_CHUNK_VERSION, variant),
            variant=variant,
        ),
        "sparse_rerank": RetrieverSpec(
            "sparse_rerank",
            lambda: _build_sparse_rerank(LEGACY_CHUNK_VERSION, variant),
            variant=variant,
        ),
        # Direct Corpus Interaction needs no vector index, only the BM25
        # shortlist and the on-disk workspace.
        "dci": RetrieverSpec(
            "dci",
            lambda: _dci(variant=variant),
            variant=variant,
        ),
    }
    for k in CONFIG["dci_k_sweep"]:
        specs[f"dci_k{k}"] = RetrieverSpec(
            f"dci_k{k}",
            lambda documents=k: _dci(max_documents=documents, variant=variant),
            variant=variant,
        )
    for judge in _judge_names():
        specs[f"dci_{judge}"] = RetrieverSpec(
            f"dci_{judge}",
            lambda j=judge: _dci(variant=variant, judge=j),
            variant=variant,
        )
    for rung, effort in _reasoning_levels().items():
        specs[f"dci_{rung}"] = RetrieverSpec(
            f"dci_{rung}",
            lambda level=effort: _dci(variant=variant, reasoning=level),
            variant=variant,
        )
        for k in CONFIG["dci_k_sweep"]:
            specs[f"dci_k{k}_{rung}"] = RetrieverSpec(
                f"dci_k{k}_{rung}",
                lambda documents=k, level=effort: _dci(
                    max_documents=documents, variant=variant, reasoning=level
                ),
                variant=variant,
            )
    for provider_name in enabled_provider_names():
        specs.update(_provider_specs(provider_name, LEGACY_CHUNK_VERSION, variant))
    return specs


def _agentic_providers() -> frozenset[str]:
    """Providers that get `_hybrid_agentic` and `_hybrid_agentic_tools` methods.

    Config rather than a literal set, because which embedder an agent sits on top
    of is an experiment choice: adding a provider should not require editing the
    registry to measure an agent against it. Every name here must also be an
    enabled embedding provider, or its methods would have no index to read.
    """
    configured = CONFIG.get("agentic_providers")
    return frozenset(configured or ("qwen", "nemotron"))


def _provider_specs(
    provider_name: str,
    chunk_version: str,
    variant: str = BASE_CHUNK_VARIANT,
) -> dict[str, RetrieverSpec]:
    suffix = _version_suffix(chunk_version)
    specs = {
        f"{provider_name}{suffix}": RetrieverSpec(
            f"{provider_name}{suffix}",
            lambda provider=provider_name, version=chunk_version: _vector(
                provider, version, variant
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
        ),
        f"{provider_name}_hybrid{suffix}": RetrieverSpec(
            f"{provider_name}_hybrid{suffix}",
            lambda provider=provider_name, version=chunk_version: _hybrid(
                provider, version, variant
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
        ),
        f"{provider_name}_rerank{suffix}": RetrieverSpec(
            f"{provider_name}_rerank{suffix}",
            lambda provider=provider_name, version=chunk_version: _vector_rerank(
                provider, version, variant
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
        ),
        f"{provider_name}_hybrid_rerank{suffix}": RetrieverSpec(
            f"{provider_name}_hybrid_rerank{suffix}",
            lambda provider=provider_name, version=chunk_version: _hybrid_rerank(
                provider, version, variant
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
        ),
    }
    # Geo-aware: the hybrid stage over-fetched and re-scored by geography (soft;
    # unknown footprint neutral). Both with and without the reranker, so the
    # reranker's share of any lift stays measurable. `_strict` keeps the
    # filter-then-widen shape the 2026-09-08 run measured, for comparison.
    for rerank in (False, True):
        name = f"{provider_name}_hybrid{'_rerank' if rerank else ''}_geo{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version, rr=rerank: _geo(
                provider, version, variant, rerank=rr
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
            requires_geo=True,
        )
    name = f"{provider_name}_hybrid_rerank_geo_strict{suffix}"
    specs[name] = RetrieverSpec(
        name,
        lambda provider=provider_name, version=chunk_version: _geo(
            provider, version, variant, rerank=True, mode="strict"
        ),
        provider=provider_name,
        chunk_version=chunk_version,
        variant=variant,
        requires_geo=True,
    )
    # One extra pair per configured reranker, so which cross-encoder does the
    # reranking becomes a method choice rather than a config edit that silently
    # redefines what every existing `*_rerank*` name measured.
    for reranker in _extra_rerankers():
        name = f"{provider_name}_rerank_{reranker}{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version, key=reranker: (
                _vector_rerank(provider, version, variant, key)
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
        )
        name = f"{provider_name}_hybrid_rerank_{reranker}{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version, key=reranker: (
                _hybrid_rerank(provider, version, variant, key)
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
        )
    if provider_name == "qwen":
        name = f"qwen_agentic{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda version=chunk_version: _qwen_agentic(version, variant),
            provider="qwen",
            chunk_version=chunk_version,
            variant=variant,
        )
        for rung, effort in _reasoning_levels().items():
            name = f"qwen_agentic_{rung}{suffix}"
            specs[name] = RetrieverSpec(
                name,
                lambda version=chunk_version, level=effort: _qwen_agentic(
                    version, variant, reasoning=level
                ),
                provider="qwen",
                chunk_version=chunk_version,
                variant=variant,
            )
    if provider_name in _agentic_providers():
        name = f"{provider_name}_hybrid_agentic{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version: _hybrid_agentic(
                provider, version, variant
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
        )
        name = f"{provider_name}_hybrid_agentic_tools{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version: _hybrid_agentic_tools(
                provider, version, variant
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
        )
        # The same two agents at each declared reasoning rung. Both families get
        # every rung, because the plain loop is the control that says whether a
        # rung moved the tools or moved the judge underneath them.
        for rung, effort in _reasoning_levels().items():
            name = f"{provider_name}_hybrid_agentic_{rung}{suffix}"
            specs[name] = RetrieverSpec(
                name,
                lambda provider=provider_name, version=chunk_version, level=effort: (
                    _hybrid_agentic(provider, version, variant, reasoning=level)
                ),
                provider=provider_name,
                chunk_version=chunk_version,
                variant=variant,
            )
            name = f"{provider_name}_hybrid_agentic_tools_{rung}{suffix}"
            specs[name] = RetrieverSpec(
                name,
                lambda provider=provider_name, version=chunk_version, level=effort: (
                    _hybrid_agentic_tools(provider, version, variant, reasoning=level)
                ),
                provider=provider_name,
                chunk_version=chunk_version,
                variant=variant,
            )
        # The judge LLM as its own axis: the same two agents under each judge
        # declared in `agentic_judges`, so DeepSeek against gpt-oss against gemma
        # is a method choice in one table rather than three config edits and
        # three runs whose checkpoints cannot tell each other apart.
        for judge in _judge_names():
            name = f"{provider_name}_hybrid_agentic_{judge}{suffix}"
            specs[name] = RetrieverSpec(
                name,
                lambda provider=provider_name, version=chunk_version, j=judge: (
                    _hybrid_agentic(provider, version, variant, judge=j)
                ),
                provider=provider_name,
                chunk_version=chunk_version,
                variant=variant,
            )
            name = f"{provider_name}_hybrid_agentic_tools_{judge}{suffix}"
            specs[name] = RetrieverSpec(
                name,
                lambda provider=provider_name, version=chunk_version, j=judge: (
                    _hybrid_agentic_tools(provider, version, variant, judge=j)
                ),
                provider=provider_name,
                chunk_version=chunk_version,
                variant=variant,
            )
        # Agents on the geo-scoped first stage, paired with `_hybrid_rerank_geo`
        # by the diagnostics the same way the plain agents pair with `_hybrid_rerank`.
        name = f"{provider_name}_hybrid_agentic_geo{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version: _hybrid_agentic(
                provider, version, variant, geo=True
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
            requires_geo=True,
        )
        name = f"{provider_name}_hybrid_agentic_tools_geo{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version: _hybrid_agentic_tools(
                provider, version, variant, geo=True
            ),
            provider=provider_name,
            chunk_version=chunk_version,
            variant=variant,
            requires_geo=True,
        )
    return specs


def _sparse(
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> SparseRetriever:
    return SparseRetriever(
        name=f"sparse{_version_suffix(chunk_version)}",
        k1=CONFIG["sparse_k1"],
        b=CONFIG["sparse_b"],
        chunk_version=chunk_version,
        variant=variant,
    )


def _build_sparse_rerank(
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> CrossEncoderRerankRetriever:
    suffix = _version_suffix(chunk_version)
    return _reranker(f"sparse_rerank{suffix}", _sparse(chunk_version, variant))


def _vector(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> Retriever:
    return VectorChunkRetriever(
        name=f"{provider_name}{_version_suffix(chunk_version)}",
        provider=provider_name,
        chunk_version=chunk_version,
        variant=variant,
    )


def _hybrid(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> WeightedScoreFusionRetriever:
    suffix = _version_suffix(chunk_version)
    return WeightedScoreFusionRetriever(
        name=f"{provider_name}_hybrid{suffix}",
        retrievers=(
            _vector(provider_name, chunk_version, variant),
            _sparse(chunk_version, variant),
        ),
        candidate_limit=CONFIG["rerank_candidate_limit"],
        weights=(CONFIG["hybrid_vector_weight"], CONFIG["hybrid_sparse_weight"]),
    )


def _geo(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
    rerank: bool = True,
    mode: str = "soft",
) -> GeoScopedRetriever:
    suffix = _version_suffix(chunk_version)
    strict = "_strict" if mode == "strict" else ""
    name = f"{provider_name}_hybrid{'_rerank' if rerank else ''}_geo{strict}{suffix}"

    def build_stage(scope: GeoScope | None) -> Retriever:
        vector = replace(
            _vector(provider_name, chunk_version, variant),
            where=scope.chroma_where() if scope else None,
        )
        sparse = replace(
            _sparse(chunk_version, variant),
            allowed_page_ids=page_ids_in_scope(scope) if scope else None,
        )
        hybrid = WeightedScoreFusionRetriever(
            name=f"{provider_name}_hybrid{suffix}",
            retrievers=(vector, sparse),
            candidate_limit=CONFIG["rerank_candidate_limit"],
            weights=(CONFIG["hybrid_vector_weight"], CONFIG["hybrid_sparse_weight"]),
        )
        return _reranker(name, hybrid) if rerank else hybrid

    return GeoScopedRetriever(
        name=name,
        resolver=LazyResolver(default_resolver),
        build_stage=build_stage,
        chunk_locations=load_chunk_locations,
        mode=mode,
        boost_weight=CONFIG[
            "geo_strict_boost_weight" if mode == "strict" else "geo_boost_weight"
        ],
        decay_km=CONFIG["geo_decay_km"],
        overfetch=CONFIG["geo_overfetch"],
        max_scope_share=CONFIG["geo_max_scope_share"],
        scope_share=located_page_share,
        min_candidates=CONFIG["geo_min_candidates"],
        include_null=bool(CONFIG.get("geo_include_null", False)),
    )


def default_gazetteer(config: dict | None = None) -> Gazetteer:
    """The place-name resolver shared by geo methods and the agent's geo tools."""
    config = config or CONFIG
    hint = config.get("geo_country_hint") or None
    return Gazetteer(
        geocoder=NominatimGeocoder(country_codes=hint),
        wikidata=WikidataClient(),
        default_radius_km=config["geo_default_radius_km"],
        country_hint=hint,
        include_null=bool(config.get("geo_include_null", False)),
    )


def default_resolver(config: dict | None = None) -> LlmGazetteerResolver:
    """Question -> scope. The LLM names the place; `default_gazetteer` places it."""
    config = config or CONFIG
    judge = default_judge(
        _judge_config(
            {
                "agentic_judge_provider": config["geo_resolver_provider"],
                "agentic_judge_structured_method": config["geo_structured_method"],
                "agentic_judge_num_predict": 512,
            }
        )
    )
    return LlmGazetteerResolver(llm=judge, gazetteer=default_gazetteer(config))


def _vector_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
    reranker: str | None = None,
) -> CrossEncoderRerankRetriever:
    suffix = _version_suffix(chunk_version)
    return _reranker(
        f"{provider_name}_rerank{_reranker_suffix(reranker)}{suffix}",
        _vector(provider_name, chunk_version, variant),
        reranker,
    )


def _hybrid_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
    reranker: str | None = None,
) -> CrossEncoderRerankRetriever:
    suffix = _version_suffix(chunk_version)
    return _reranker(
        f"{provider_name}_hybrid_rerank{_reranker_suffix(reranker)}{suffix}",
        _hybrid(provider_name, chunk_version, variant),
        reranker,
    )


def _qwen_agentic(
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
    reasoning: str | None = None,
) -> AgenticRetriever:
    suffix = _version_suffix(chunk_version)
    return AgenticRetriever(
        name=f"qwen_agentic{_reasoning_suffix(reasoning)}{suffix}",
        base_retriever=_vector_rerank("qwen", chunk_version, variant),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
        judge=default_judge(_judge_config(reasoning=reasoning)),
    )


def _hybrid_agentic(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
    reasoning: str | None = None,
    judge: str | None = None,
    geo: bool = False,
) -> AgenticRetriever:
    suffix = _version_suffix(chunk_version)
    tail = (
        f"{_reasoning_suffix(reasoning)}{_judge_suffix(judge)}{'_geo' if geo else ''}"
    )
    return AgenticRetriever(
        name=f"{provider_name}_hybrid_agentic{tail}{suffix}",
        base_retriever=(
            _geo(provider_name, chunk_version, variant, rerank=True)
            if geo
            else _hybrid_rerank(provider_name, chunk_version, variant)
        ),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
        judge=default_judge(_judge_config(reasoning=reasoning, judge=judge)),
    )


def _hybrid_agentic_tools(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
    reasoning: str | None = None,
    judge: str | None = None,
    geo: bool = False,
) -> AgenticToolRetriever:
    suffix = _version_suffix(chunk_version)
    tail = (
        f"{_reasoning_suffix(reasoning)}{_judge_suffix(judge)}{'_geo' if geo else ''}"
    )
    return AgenticToolRetriever(
        name=f"{provider_name}_hybrid_agentic_tools{tail}{suffix}",
        base_retriever=(
            _geo(provider_name, chunk_version, variant, rerank=True)
            if geo
            else _hybrid_rerank(provider_name, chunk_version, variant)
        ),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_tools_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
        max_tool_calls=CONFIG["agentic_tools_max_tool_calls"],
        section_limit=CONFIG["agentic_tools_section_limit"],
        search_limit=CONFIG["agentic_tools_search_limit"],
        variant=variant,
        # The geo tools (find_pages_near, pages_in_region) belong to the `_geo`
        # methods only, so the plain tool agent keeps measuring what it did.
        gazetteer=default_gazetteer() if geo else None,
        judge=default_judge(
            _judge_config(
                {
                    "agentic_judge_structured_method": CONFIG[
                        "agentic_tools_structured_method"
                    ]
                },
                reasoning=reasoning,
                judge=judge,
                method_key="agentic_tools_structured_method",
            )
        ),
    )


def _dci(
    max_documents: int | None = None,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
    reasoning: str | None = None,
    judge: str | None = None,
) -> DirectCorpusRetriever:
    documents = (
        CONFIG["dci_max_documents"] if max_documents is None else int(max_documents)
    )
    suffix = "" if max_documents is None else f"_k{documents}"
    return DirectCorpusRetriever(
        name=f"dci{suffix}{_reasoning_suffix(reasoning)}{_judge_suffix(judge)}",
        variant=variant,
        shortlist_retriever=_sparse(chunk_version, variant),
        shortlist_k=CONFIG["dci_shortlist_k"],
        max_documents=documents,
        max_steps=CONFIG["dci_max_steps"],
        search_limit=CONFIG["dci_search_limit"],
        read_limit=CONFIG["dci_read_limit"],
        page_expansion_limit=CONFIG["dci_page_expansion_limit"],
        max_answer_retries=CONFIG["dci_max_answer_retries"],
        judge_retries=CONFIG["agentic_judge_retries"],
        judge=default_judge(
            _judge_config(
                {
                    "agentic_judge_structured_method": CONFIG["dci_structured_method"],
                    "agentic_judge_num_predict": CONFIG["dci_num_predict"],
                },
                reasoning=reasoning,
                judge=judge,
                method_key="dci_structured_method",
            )
        ),
    )


def _version_suffix(chunk_version: str) -> str:
    return "" if chunk_version == LEGACY_CHUNK_VERSION else f"_{chunk_version}"


def _reasoning_suffix(reasoning: str | None) -> str:
    return "" if reasoning is None else f"_{reasoning}"


def _collection_ready(provider: str, chunk_version: str, variant: str) -> bool:
    return collection_ready(provider, chunk_version, variant)


def _index_requirement(provider: str, chunk_version: str, variant: str) -> str:
    requirement = provider
    if chunk_version != LEGACY_CHUNK_VERSION:
        requirement += f"@{chunk_version}"
    if variant != BASE_CHUNK_VARIANT:
        requirement += f"#{variant}"
    return requirement


def _index_build_command(requirement: str) -> str:
    head, _, variant = requirement.partition("#")
    provider, _, chunk_version = head.partition("@")
    command = f"  uv run python -m src.indexing.chunks --method {provider}"
    if chunk_version:
        command += f" --chunk-version {chunk_version}"
    if variant:
        command += f" --variant {variant}"
    return command


def _extra_rerankers() -> tuple[str, ...]:
    """Named rerankers beyond the default, in config order."""
    return tuple(CONFIG.get("rerankers") or ())


def _reasoning_levels() -> dict[str, str]:
    """Named judge reasoning rungs beyond the configured default, in config order.

    An agent's reasoning effort is a benchmark axis: the same loop at `low` and at
    `high` is two measurements, not a setting to pick once. Declared in config so
    a rung is added without editing the registry, and suffixed so the default
    method names keep measuring exactly what they measured before.
    """
    return dict(CONFIG.get("agentic_reasoning_levels") or {})


def _judge_config(
    *overrides: dict,
    reasoning: str | None = None,
    judge: str | None = None,
    method_key: str = "agentic_judge_structured_method",
) -> dict:
    """CONFIG with per-method judge overrides, a reasoning rung, and a named judge.

    A named judge (`agentic_judges` in config) supplies the model and, through
    `method_key`, the structured-output method for this schema: the working
    method is a property of the (model, schema) pair. Per-method overrides
    (DCI's `dci_num_predict`, a reasoning rung) stay in force under any judge,
    so `dci` and `dci_gptoss` differ in the model alone.
    """
    config = dict(CONFIG)
    pinned: set[str] = set()
    for override in overrides:
        config.update(override)
        pinned |= set(override)
    if reasoning is not None:
        config["agentic_judge_reasoning"] = reasoning
        pinned.add("agentic_judge_reasoning")
    if judge is not None:
        settings = dict(_judges()[judge])
        if method_key in settings:
            # The judge's per-schema method beats the method's default one.
            settings["agentic_judge_structured_method"] = settings[method_key]
            pinned.discard("agentic_judge_structured_method")
        # Everything else a method pinned (DCI's output budget, an explicit
        # reasoning rung) survives the judge swap; the judge supplies the model.
        config.update({k: v for k, v in settings.items() if k not in pinned})
    return config


def _judges() -> dict[str, dict]:
    """Named judge LLMs beyond the configured default, in config order."""
    return dict(CONFIG.get("agentic_judges") or {})


def _judge_names() -> tuple[str, ...]:
    return tuple(_judges())


def _judge_suffix(judge: str | None) -> str:
    if judge is None:
        return ""
    if judge not in _judges():
        raise ValueError(
            f"Unknown judge: {judge}. Configured: {', '.join(_judges()) or 'none'}"
        )
    return f"_{judge}"


def _reranker_suffix(reranker: str | None) -> str:
    return "" if reranker is None else f"_{reranker}"


def _reranker_settings(reranker: str | None) -> dict:
    """The default reranker settings, with one named reranker's overrides applied.

    Inheriting rather than restating means a named rung differs from the default
    only in what it names: `max_length` and the candidate limit cannot drift
    between rungs, so a score difference is the model and not the harness.
    """
    settings = {
        "model": CONFIG["reranker_model"],
        "batch_size": CONFIG["reranker_batch_size"],
        "device": CONFIG["reranker_device"],
        "max_length": CONFIG["reranker_max_length"],
        "local_files_only": CONFIG["reranker_local_files_only"],
        "prompt_name": CONFIG["reranker_prompt_name"],
        "prompt": CONFIG["reranker_prompt"],
        "trust_remote_code": CONFIG.get("reranker_trust_remote_code", False),
        "dtype": CONFIG.get("reranker_dtype"),
    }
    if reranker is None:
        return settings
    configured = CONFIG.get("rerankers") or {}
    if reranker not in configured:
        raise ValueError(
            f"Unknown reranker: {reranker}. "
            f"Configured: {', '.join(sorted(configured)) or 'none'}"
        )
    unknown = set(configured[reranker]) - set(settings)
    if unknown:
        raise ValueError(
            f"Reranker {reranker} sets unknown keys: {', '.join(sorted(unknown))}"
        )
    return {**settings, **configured[reranker]}


def _reranker(
    name: str,
    base_retriever: Retriever,
    reranker: str | None = None,
) -> CrossEncoderRerankRetriever:
    settings = _reranker_settings(reranker)
    return CrossEncoderRerankRetriever(
        name=name,
        model_name=settings["model"],
        base_retriever=base_retriever,
        candidate_limit=CONFIG["rerank_candidate_limit"],
        device=settings["device"],
        max_length=settings["max_length"],
        local_files_only=settings["local_files_only"],
        batch_size=settings["batch_size"],
        prompt_name=settings["prompt_name"],
        prompt=settings["prompt"],
        trust_remote_code=settings["trust_remote_code"],
        dtype=settings["dtype"],
    )
