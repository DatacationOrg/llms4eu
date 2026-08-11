from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.indexing.chunk_text import CONTEXTUAL_CHUNK_VERSION, LEGACY_CHUNK_VERSION
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


def build_retriever(name: str) -> Retriever:
    specs = _specs()
    if name not in specs:
        raise ValueError(f"Unknown retriever: {name}")
    return specs[name].build()


def ensure_retriever_ready(name: str) -> None:
    ensure_retrievers_ready([name])


def ensure_retrievers_ready(names: list[str]) -> None:
    missing = missing_retriever_indexes(names)
    if missing:
        raise MissingRetrieverIndexes(missing)


def missing_retriever_indexes(names: list[str]) -> dict[str, str]:
    specs = _specs()
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f"Unknown retriever: {', '.join(unknown)}")

    missing = {}
    for name in names:
        spec = specs[name]
        if spec.provider is not None and not _collection_ready(
            spec.provider,
            spec.chunk_version,
        ):
            missing[name] = _index_requirement(spec.provider, spec.chunk_version)
    return missing


def _specs() -> dict[str, RetrieverSpec]:
    specs = {
        "sparse": RetrieverSpec("sparse", _sparse),
        "sparse_rerank": RetrieverSpec(
            "sparse_rerank",
            _build_sparse_rerank,
        ),
        "sparse_v2": RetrieverSpec(
            "sparse_v2",
            lambda: _sparse(CONTEXTUAL_CHUNK_VERSION),
            chunk_version=CONTEXTUAL_CHUNK_VERSION,
        ),
        "sparse_rerank_v2": RetrieverSpec(
            "sparse_rerank_v2",
            lambda: _build_sparse_rerank(CONTEXTUAL_CHUNK_VERSION),
            chunk_version=CONTEXTUAL_CHUNK_VERSION,
        ),
        # Direct Corpus Interaction needs no vector index, only the BM25
        # shortlist and the on-disk workspace.
        "dci": RetrieverSpec("dci", _dci),
    }
    for k in CONFIG["dci_k_sweep"]:
        specs[f"dci_k{k}"] = RetrieverSpec(
            f"dci_k{k}",
            lambda documents=k: _dci(max_documents=documents),
        )
    for provider_name in enabled_provider_names():
        specs.update(_provider_specs(provider_name, LEGACY_CHUNK_VERSION))
        specs.update(_provider_specs(provider_name, CONTEXTUAL_CHUNK_VERSION))
    return specs


def _provider_specs(
    provider_name: str,
    chunk_version: str,
) -> dict[str, RetrieverSpec]:
    suffix = _version_suffix(chunk_version)
    specs = {
        f"{provider_name}{suffix}": RetrieverSpec(
            f"{provider_name}{suffix}",
            lambda provider=provider_name, version=chunk_version: _vector(
                provider, version
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
        f"{provider_name}_hybrid{suffix}": RetrieverSpec(
            f"{provider_name}_hybrid{suffix}",
            lambda provider=provider_name, version=chunk_version: _hybrid(
                provider, version
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
        f"{provider_name}_rerank{suffix}": RetrieverSpec(
            f"{provider_name}_rerank{suffix}",
            lambda provider=provider_name, version=chunk_version: _vector_rerank(
                provider, version
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
        f"{provider_name}_hybrid_rerank{suffix}": RetrieverSpec(
            f"{provider_name}_hybrid_rerank{suffix}",
            lambda provider=provider_name, version=chunk_version: _hybrid_rerank(
                provider, version
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
    }
    if provider_name == "qwen":
        name = f"qwen_agentic{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda version=chunk_version: _qwen_agentic(version),
            provider="qwen",
            chunk_version=chunk_version,
        )
    if provider_name in {"qwen", "nemotron"}:
        name = f"{provider_name}_hybrid_agentic{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version: _hybrid_agentic(
                provider, version
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        )
        name = f"{provider_name}_hybrid_agentic_tools{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version: _hybrid_agentic_tools(
                provider, version
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        )
    return specs


def _sparse(chunk_version: str = LEGACY_CHUNK_VERSION) -> SparseRetriever:
    return SparseRetriever(
        name=f"sparse{_version_suffix(chunk_version)}",
        k1=CONFIG["sparse_k1"],
        b=CONFIG["sparse_b"],
        chunk_version=chunk_version,
    )


def _build_sparse_rerank(
    chunk_version: str = LEGACY_CHUNK_VERSION,
) -> CrossEncoderRerankRetriever:
    suffix = _version_suffix(chunk_version)
    return _reranker(f"sparse_rerank{suffix}", _sparse(chunk_version))


def _vector(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
) -> Retriever:
    return VectorChunkRetriever(
        name=f"{provider_name}{_version_suffix(chunk_version)}",
        provider=provider_name,
        chunk_version=chunk_version,
    )


def _hybrid(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
) -> WeightedScoreFusionRetriever:
    suffix = _version_suffix(chunk_version)
    return WeightedScoreFusionRetriever(
        name=f"{provider_name}_hybrid{suffix}",
        retrievers=(
            _vector(provider_name, chunk_version),
            _sparse(chunk_version),
        ),
        candidate_limit=CONFIG["rerank_candidate_limit"],
        weights=(CONFIG["hybrid_vector_weight"], CONFIG["hybrid_sparse_weight"]),
    )


def _vector_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
) -> CrossEncoderRerankRetriever:
    suffix = _version_suffix(chunk_version)
    return _reranker(
        f"{provider_name}_rerank{suffix}",
        _vector(provider_name, chunk_version),
    )


def _hybrid_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
) -> CrossEncoderRerankRetriever:
    suffix = _version_suffix(chunk_version)
    return _reranker(
        f"{provider_name}_hybrid_rerank{suffix}",
        _hybrid(provider_name, chunk_version),
    )


def _qwen_agentic(
    chunk_version: str = LEGACY_CHUNK_VERSION,
) -> AgenticRetriever:
    suffix = _version_suffix(chunk_version)
    return AgenticRetriever(
        name=f"qwen_agentic{suffix}",
        base_retriever=_vector_rerank("qwen", chunk_version),
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
) -> AgenticRetriever:
    suffix = _version_suffix(chunk_version)
    return AgenticRetriever(
        name=f"{provider_name}_hybrid_agentic{suffix}",
        base_retriever=_hybrid_rerank(provider_name, chunk_version),
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
) -> AgenticToolRetriever:
    suffix = _version_suffix(chunk_version)
    return AgenticToolRetriever(
        name=f"{provider_name}_hybrid_agentic_tools{suffix}",
        base_retriever=_hybrid_rerank(provider_name, chunk_version),
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
) -> DirectCorpusRetriever:
    documents = (
        CONFIG["dci_max_documents"] if max_documents is None else int(max_documents)
    )
    suffix = "" if max_documents is None else f"_k{documents}"
    return DirectCorpusRetriever(
        name=f"dci{suffix}",
        shortlist_retriever=_sparse(chunk_version),
        shortlist_k=CONFIG["dci_shortlist_k"],
        max_documents=documents,
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


def _collection_ready(provider: str, chunk_version: str) -> bool:
    if chunk_version == LEGACY_CHUNK_VERSION:
        return collection_ready(provider)
    return collection_ready(provider, chunk_version)


def _index_requirement(provider: str, chunk_version: str) -> str:
    if chunk_version == LEGACY_CHUNK_VERSION:
        return provider
    return f"{provider}@{chunk_version}"


def _index_build_command(requirement: str) -> str:
    provider, _, chunk_version = requirement.partition("@")
    command = f"  uv run python -m src.indexing.chunks --method {provider}"
    if chunk_version:
        command += f" --chunk-version {chunk_version}"
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
