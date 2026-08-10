from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.indexing.chunk_text import (
    BASE_CHUNK_VARIANT,
    CONTEXTUAL_CHUNK_VERSION,
    LEGACY_CHUNK_VERSION,
)
from src.retrieval.base import Retriever
from src.retrieval.retrievers.agentic import AgenticRetriever, default_judge
from src.retrieval.retrievers.agentic_tools import AgenticToolRetriever
from src.retrieval.retrievers.dci import DirectCorpusRetriever
from src.retrieval.retrievers.fusion import WeightedScoreFusionRetriever
from src.retrieval.retrievers.rerank import CrossEncoderRerankRetriever
from src.retrieval.retrievers.sparse import SparseRetriever
from src.retrieval.retrievers.vector_chunks import VectorChunkRetriever
from src.shared.env import load_yaml
from src.vector_store.chunks import collection_ready, enabled_provider_names

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


@dataclass(frozen=True)
class RetrieverSpec:
    """Lazy catalog entry for one public retrieval method."""

    name: str
    build: Callable[[], Retriever]
    provider: str | None = None
    chunk_version: str = LEGACY_CHUNK_VERSION
    chunk_variant: str = BASE_CHUNK_VARIANT

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


def list_retrievers() -> list[str]:
    return sorted(_specs())


def build_retriever(name: str, variant: str = BASE_CHUNK_VARIANT) -> Retriever:
    """Build one method. `variant` selects which chunking it retrieves over.

    Variants are not separate catalog entries: the method names stay stable and
    the variant is passed in, so a chunking ablation reuses the same method list.
    """
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
        if spec.provider is not None and not _collection_ready(
            spec.provider,
            spec.chunk_version,
            spec.chunk_variant,
        ):
            missing[name] = _index_requirement(
                spec.provider, spec.chunk_version, spec.chunk_variant
            )
    return missing


def _specs(variant: str = BASE_CHUNK_VARIANT) -> dict[str, RetrieverSpec]:
    specs = {
        "sparse": RetrieverSpec(
            "sparse",
            lambda: _sparse(LEGACY_CHUNK_VERSION, variant),
            chunk_variant=variant,
        ),
        "sparse_rerank": RetrieverSpec(
            "sparse_rerank",
            lambda: _build_sparse_rerank(LEGACY_CHUNK_VERSION, variant),
            chunk_variant=variant,
        ),
        "sparse_v2": RetrieverSpec(
            "sparse_v2",
            lambda: _sparse(CONTEXTUAL_CHUNK_VERSION, variant),
            chunk_version=CONTEXTUAL_CHUNK_VERSION,
            chunk_variant=variant,
        ),
        "sparse_rerank_v2": RetrieverSpec(
            "sparse_rerank_v2",
            lambda: _build_sparse_rerank(CONTEXTUAL_CHUNK_VERSION, variant),
            chunk_version=CONTEXTUAL_CHUNK_VERSION,
            chunk_variant=variant,
        ),
        # Direct Corpus Interaction needs no vector index, only the BM25
        # shortlist and the on-disk workspace.
        "dci": RetrieverSpec(
            "dci",
            lambda: _dci(chunk_variant=variant),
            chunk_variant=variant,
        ),
    }
    for k in CONFIG["dci_k_sweep"]:
        specs[f"dci_k{k}"] = RetrieverSpec(
            f"dci_k{k}",
            lambda documents=k: _dci(max_documents=documents, chunk_variant=variant),
            chunk_variant=variant,
        )
    for provider_name in enabled_provider_names():
        specs.update(_provider_specs(provider_name, LEGACY_CHUNK_VERSION, variant))
        specs.update(_provider_specs(provider_name, CONTEXTUAL_CHUNK_VERSION, variant))
    return specs


def _provider_specs(
    provider_name: str,
    chunk_version: str,
    variant: str = BASE_CHUNK_VARIANT,
) -> dict[str, RetrieverSpec]:
    suffix = _version_suffix(chunk_version)

    def spec(name: str, build: Callable[[], Retriever]) -> RetrieverSpec:
        return RetrieverSpec(
            name,
            build,
            provider=provider_name,
            chunk_version=chunk_version,
            chunk_variant=variant,
        )

    provider, version = provider_name, chunk_version
    specs = {
        f"{provider_name}{suffix}": spec(
            f"{provider_name}{suffix}",
            lambda: _vector(provider, version, variant),
        ),
        f"{provider_name}_hybrid{suffix}": spec(
            f"{provider_name}_hybrid{suffix}",
            lambda: _hybrid(provider, version, variant),
        ),
        f"{provider_name}_rerank{suffix}": spec(
            f"{provider_name}_rerank{suffix}",
            lambda: _vector_rerank(provider, version, variant),
        ),
        f"{provider_name}_hybrid_rerank{suffix}": spec(
            f"{provider_name}_hybrid_rerank{suffix}",
            lambda: _hybrid_rerank(provider, version, variant),
        ),
    }
    if provider_name == "qwen":
        name = f"qwen_agentic{suffix}"
        specs[name] = spec(name, lambda: _qwen_agentic(version, variant))
    if provider_name in {"qwen", "nemotron"}:
        name = f"{provider_name}_hybrid_agentic{suffix}"
        specs[name] = spec(name, lambda: _hybrid_agentic(provider, version, variant))
        name = f"{provider_name}_hybrid_agentic_tools{suffix}"
        specs[name] = spec(
            name, lambda: _hybrid_agentic_tools(provider, version, variant)
        )
    return specs


def _sparse(
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> SparseRetriever:
    return SparseRetriever(
        name=f"sparse{_version_suffix(chunk_version)}{_variant_suffix(chunk_variant)}",
        k1=CONFIG["sparse_k1"],
        b=CONFIG["sparse_b"],
        chunk_version=chunk_version,
        chunk_variant=chunk_variant,
    )


def _build_sparse_rerank(
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> CrossEncoderRerankRetriever:
    return _reranker(
        _method_name("sparse_rerank", chunk_version, chunk_variant),
        _sparse(chunk_version, chunk_variant),
    )


def _vector(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> Retriever:
    return VectorChunkRetriever(
        name=_method_name(provider_name, chunk_version, chunk_variant),
        provider=provider_name,
        chunk_version=chunk_version,
        chunk_variant=chunk_variant,
    )


def _hybrid(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> WeightedScoreFusionRetriever:
    return WeightedScoreFusionRetriever(
        name=_method_name(f"{provider_name}_hybrid", chunk_version, chunk_variant),
        retrievers=(
            _vector(provider_name, chunk_version, chunk_variant),
            _sparse(chunk_version, chunk_variant),
        ),
        candidate_limit=CONFIG["rerank_candidate_limit"],
        weights=(CONFIG["hybrid_vector_weight"], CONFIG["hybrid_sparse_weight"]),
    )


def _vector_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> CrossEncoderRerankRetriever:
    return _reranker(
        _method_name(f"{provider_name}_rerank", chunk_version, chunk_variant),
        _vector(provider_name, chunk_version, chunk_variant),
    )


def _hybrid_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> CrossEncoderRerankRetriever:
    return _reranker(
        _method_name(f"{provider_name}_hybrid_rerank", chunk_version, chunk_variant),
        _hybrid(provider_name, chunk_version, chunk_variant),
    )


def _qwen_agentic(
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> AgenticRetriever:
    return AgenticRetriever(
        name=_method_name("qwen_agentic", chunk_version, chunk_variant),
        base_retriever=_vector_rerank("qwen", chunk_version, chunk_variant),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
        judge=default_judge(CONFIG),
    )


def _hybrid_agentic(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> AgenticRetriever:
    return AgenticRetriever(
        name=_method_name(
            f"{provider_name}_hybrid_agentic", chunk_version, chunk_variant
        ),
        base_retriever=_hybrid_rerank(provider_name, chunk_version, chunk_variant),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
        judge=default_judge(CONFIG),
    )


def _hybrid_agentic_tools(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> AgenticToolRetriever:
    return AgenticToolRetriever(
        name=_method_name(
            f"{provider_name}_hybrid_agentic_tools", chunk_version, chunk_variant
        ),
        base_retriever=_hybrid_rerank(provider_name, chunk_version, chunk_variant),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_tools_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
        max_tool_calls=CONFIG["agentic_tools_max_tool_calls"],
        section_limit=CONFIG["agentic_tools_section_limit"],
        search_limit=CONFIG["agentic_tools_search_limit"],
        judge=default_judge(
            {
                **CONFIG,
                "agentic_judge_structured_method": CONFIG[
                    "agentic_tools_structured_method"
                ],
            }
        ),
    )


def _dci(
    max_documents: int | None = None,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> DirectCorpusRetriever:
    documents = (
        CONFIG["dci_max_documents"] if max_documents is None else int(max_documents)
    )
    suffix = "" if max_documents is None else f"_k{documents}"
    return DirectCorpusRetriever(
        name=f"dci{suffix}{_variant_suffix(chunk_variant)}",
        shortlist_retriever=_sparse(chunk_version, chunk_variant),
        shortlist_k=CONFIG["dci_shortlist_k"],
        max_documents=documents,
        chunk_variant=chunk_variant,
        max_steps=CONFIG["dci_max_steps"],
        search_limit=CONFIG["dci_search_limit"],
        read_limit=CONFIG["dci_read_limit"],
        judge_retries=CONFIG["agentic_judge_retries"],
        judge=default_judge(
            {
                **CONFIG,
                "agentic_judge_structured_method": CONFIG["dci_structured_method"],
                "agentic_judge_num_predict": CONFIG["dci_num_predict"],
            }
        ),
    )


def _version_suffix(chunk_version: str) -> str:
    return "" if chunk_version == LEGACY_CHUNK_VERSION else f"_{chunk_version}"


def _variant_suffix(chunk_variant: str) -> str:
    return "" if chunk_variant == BASE_CHUNK_VARIANT else f"_{chunk_variant}"


def _method_name(base: str, chunk_version: str, chunk_variant: str) -> str:
    """Report name for a built retriever.

    The base variant keeps the historical name, so existing reports and
    checkpoints resume unchanged; other variants get a suffix so a chunking
    ablation can hold several cuts of the same method in one run.
    """
    return f"{base}{_version_suffix(chunk_version)}{_variant_suffix(chunk_variant)}"


def _collection_ready(
    provider: str,
    chunk_version: str,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> bool:
    return collection_ready(provider, chunk_version, chunk_variant)


def _index_requirement(
    provider: str,
    chunk_version: str,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> str:
    requirement = provider
    if chunk_version != LEGACY_CHUNK_VERSION:
        requirement += f"@{chunk_version}"
    if chunk_variant != BASE_CHUNK_VARIANT:
        requirement += f"#{chunk_variant}"
    return requirement


def _index_build_command(requirement: str) -> str:
    provider, _, rest = requirement.partition("@")
    provider, _, variant_only = provider.partition("#")
    chunk_version, _, chunk_variant = rest.partition("#")
    chunk_variant = chunk_variant or variant_only
    command = f"  uv run python -m src.indexing.chunks --method {provider}"
    if chunk_version:
        command += f" --chunk-version {chunk_version}"
    if chunk_variant:
        command += f" --chunk-variant {chunk_variant}"
    return command


def _reranker(name: str, base_retriever: Retriever) -> CrossEncoderRerankRetriever:
    return CrossEncoderRerankRetriever(
        name=name,
        model_name=CONFIG["reranker_model"],
        base_retriever=base_retriever,
        candidate_limit=CONFIG["rerank_candidate_limit"],
        device=CONFIG["reranker_device"],
        max_length=CONFIG["reranker_max_length"],
        local_files_only=CONFIG["reranker_local_files_only"],
        batch_size=CONFIG["reranker_batch_size"],
        prompt_name=CONFIG["reranker_prompt_name"],
        prompt=CONFIG["reranker_prompt"],
    )
