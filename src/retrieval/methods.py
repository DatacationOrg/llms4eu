from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.indexing.chunk_text import CONTEXTUAL_CHUNK_VERSION, LEGACY_CHUNK_VERSION
from src.retrieval.base import Retriever
from src.retrieval.retrievers.agentic import AgenticRetriever
from src.retrieval.retrievers.fusion import WeightedScoreFusionRetriever
from src.retrieval.retrievers.rerank import (
    AzureCohereRerankRetriever,
    CrossEncoderRerankRetriever,
)
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
        "sparse_rerank_cohere": RetrieverSpec(
            "sparse_rerank_cohere",
            _build_sparse_cohere_rerank,
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
        "sparse_rerank_cohere_v2": RetrieverSpec(
            "sparse_rerank_cohere_v2",
            lambda: _build_sparse_cohere_rerank(CONTEXTUAL_CHUNK_VERSION),
            chunk_version=CONTEXTUAL_CHUNK_VERSION,
        ),
    }
    for provider_name in enabled_provider_names():
        public_name = "embed_v4" if provider_name == "azure" else provider_name
        specs.update(_provider_specs(provider_name, LEGACY_CHUNK_VERSION, public_name))
        specs.update(
            _provider_specs(provider_name, CONTEXTUAL_CHUNK_VERSION, public_name)
        )
    return specs


def _provider_specs(
    provider_name: str,
    chunk_version: str,
    public_name: str | None = None,
) -> dict[str, RetrieverSpec]:
    public_name = public_name or provider_name
    suffix = _version_suffix(chunk_version)
    specs = {
        f"{public_name}{suffix}": RetrieverSpec(
            f"{public_name}{suffix}",
            lambda provider=provider_name, version=chunk_version, name=public_name: (
                _vector(provider, version, name)
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
        f"{public_name}_hybrid{suffix}": RetrieverSpec(
            f"{public_name}_hybrid{suffix}",
            lambda provider=provider_name, version=chunk_version, name=public_name: (
                _hybrid(provider, version, name)
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
        f"{public_name}_rerank{suffix}": RetrieverSpec(
            f"{public_name}_rerank{suffix}",
            lambda provider=provider_name, version=chunk_version, name=public_name: (
                _vector_rerank(provider, version, name)
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
        f"{public_name}_hybrid_rerank{suffix}": RetrieverSpec(
            f"{public_name}_hybrid_rerank{suffix}",
            lambda provider=provider_name, version=chunk_version, name=public_name: (
                _hybrid_rerank(provider, version, name)
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
        f"{public_name}_rerank_cohere{suffix}": RetrieverSpec(
            f"{public_name}_rerank_cohere{suffix}",
            lambda provider=provider_name, version=chunk_version, name=public_name: (
                _vector_cohere_rerank(provider, version, name)
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        ),
        f"{public_name}_hybrid_rerank_cohere{suffix}": RetrieverSpec(
            f"{public_name}_hybrid_rerank_cohere{suffix}",
            lambda provider=provider_name, version=chunk_version, name=public_name: (
                _hybrid_cohere_rerank(provider, version, name)
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
    if provider_name in {"qwen", "nemotron", "azure"}:
        name = f"{public_name}_hybrid_agentic{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version, public=public_name: (
                _hybrid_agentic(provider, version, public)
            ),
            provider=provider_name,
            chunk_version=chunk_version,
        )
    if provider_name == "azure":
        name = f"{public_name}_hybrid_agentic_cohere{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda provider=provider_name, version=chunk_version, public=public_name: (
                _hybrid_cohere_agentic(provider, version, public)
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


def _build_sparse_cohere_rerank(
    chunk_version: str = LEGACY_CHUNK_VERSION,
) -> AzureCohereRerankRetriever:
    suffix = _version_suffix(chunk_version)
    return _cohere_reranker(
        f"sparse_rerank_cohere{suffix}",
        _sparse(chunk_version),
    )


def _vector(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    public_name: str | None = None,
) -> Retriever:
    public_name = public_name or provider_name
    return VectorChunkRetriever(
        name=f"{public_name}{_version_suffix(chunk_version)}",
        provider=provider_name,
        chunk_version=chunk_version,
    )


def _hybrid(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    public_name: str | None = None,
) -> WeightedScoreFusionRetriever:
    public_name = public_name or provider_name
    suffix = _version_suffix(chunk_version)
    return WeightedScoreFusionRetriever(
        name=f"{public_name}_hybrid{suffix}",
        retrievers=(
            _vector(provider_name, chunk_version, public_name),
            _sparse(chunk_version),
        ),
        candidate_limit=CONFIG["rerank_candidate_limit"],
        weights=(CONFIG["hybrid_vector_weight"], CONFIG["hybrid_sparse_weight"]),
    )


def _vector_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    public_name: str | None = None,
) -> CrossEncoderRerankRetriever:
    public_name = public_name or provider_name
    suffix = _version_suffix(chunk_version)
    return _reranker(
        f"{public_name}_rerank{suffix}",
        _vector(provider_name, chunk_version, public_name),
    )


def _hybrid_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    public_name: str | None = None,
) -> CrossEncoderRerankRetriever:
    public_name = public_name or provider_name
    suffix = _version_suffix(chunk_version)
    return _reranker(
        f"{public_name}_hybrid_rerank{suffix}",
        _hybrid(provider_name, chunk_version, public_name),
    )


def _vector_cohere_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    public_name: str | None = None,
) -> AzureCohereRerankRetriever:
    public_name = public_name or provider_name
    suffix = _version_suffix(chunk_version)
    return _cohere_reranker(
        f"{public_name}_rerank_cohere{suffix}",
        _vector(provider_name, chunk_version, public_name),
    )


def _hybrid_cohere_rerank(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    public_name: str | None = None,
) -> AzureCohereRerankRetriever:
    public_name = public_name or provider_name
    suffix = _version_suffix(chunk_version)
    return _cohere_reranker(
        f"{public_name}_hybrid_rerank_cohere{suffix}",
        _hybrid(provider_name, chunk_version, public_name),
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
    )


def _hybrid_agentic(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    public_name: str | None = None,
) -> AgenticRetriever:
    public_name = public_name or provider_name
    suffix = _version_suffix(chunk_version)
    return AgenticRetriever(
        name=f"{public_name}_hybrid_agentic{suffix}",
        base_retriever=_hybrid_rerank(provider_name, chunk_version, public_name),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
    )


def _hybrid_cohere_agentic(
    provider_name: str,
    chunk_version: str = LEGACY_CHUNK_VERSION,
    public_name: str | None = None,
) -> AgenticRetriever:
    public_name = public_name or provider_name
    suffix = _version_suffix(chunk_version)
    return AgenticRetriever(
        name=f"{public_name}_hybrid_agentic_cohere{suffix}",
        base_retriever=_hybrid_cohere_rerank(
            provider_name,
            chunk_version,
            public_name,
        ),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
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


def _cohere_reranker(
    name: str,
    base_retriever: Retriever,
) -> AzureCohereRerankRetriever:
    return AzureCohereRerankRetriever(
        name=name,
        model_name=os.getenv(
            "AZURE_COHERE_RERANK_MODEL",
            CONFIG["cohere_reranker_model"],
        ),
        endpoint=os.environ["AZURE_COHERE_RERANK_ENDPOINT"],
        api_key=os.environ["AZURE_COHERE_RERANK_API_KEY"],
        base_retriever=base_retriever,
        candidate_limit=CONFIG["rerank_candidate_limit"],
        timeout_seconds=CONFIG["cohere_reranker_timeout_seconds"],
        retries=CONFIG["cohere_reranker_retries"],
    )
