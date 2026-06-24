from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.retrieval.base import Retriever
from src.retrieval.retrievers.agentic import AgenticRetriever
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

    @property
    def requires_index(self) -> bool:
        return self.provider is not None


class MissingRetrieverIndexes(RuntimeError):
    def __init__(self, missing: dict[str, str]) -> None:
        self.missing = missing
        providers = sorted(set(missing.values()))
        commands = "\n".join(
            f"  uv run python -m src.indexing.chunks --method {provider}"
            for provider in providers
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
        provider_name = specs[name].provider
        if provider_name is not None and not collection_ready(provider_name):
            missing[name] = provider_name
    return missing


def _specs() -> dict[str, RetrieverSpec]:
    specs = {
        "sparse": RetrieverSpec("sparse", _sparse),
        "sparse_rerank": RetrieverSpec(
            "sparse_rerank",
            _build_sparse_rerank,
        ),
    }
    for provider_name in enabled_provider_names():
        specs.update(_provider_specs(provider_name))
    return specs


def _provider_specs(provider_name: str) -> dict[str, RetrieverSpec]:
    specs = {
        provider_name: RetrieverSpec(
            provider_name,
            lambda provider=provider_name: _vector(provider),
            provider=provider_name,
        ),
        f"{provider_name}_hybrid": RetrieverSpec(
            f"{provider_name}_hybrid",
            lambda provider=provider_name: _hybrid(provider),
            provider=provider_name,
        ),
        f"{provider_name}_rerank": RetrieverSpec(
            f"{provider_name}_rerank",
            lambda provider=provider_name: _vector_rerank(provider),
            provider=provider_name,
        ),
        f"{provider_name}_hybrid_rerank": RetrieverSpec(
            f"{provider_name}_hybrid_rerank",
            lambda provider=provider_name: _hybrid_rerank(provider),
            provider=provider_name,
        ),
    }
    if provider_name == "qwen":
        specs["qwen_agentic"] = RetrieverSpec(
            "qwen_agentic",
            _qwen_agentic,
            provider="qwen",
        )
        specs["qwen_hybrid_agentic"] = RetrieverSpec(
            "qwen_hybrid_agentic",
            _qwen_hybrid_agentic,
            provider="qwen",
        )
    return specs


def _sparse() -> SparseRetriever:
    return SparseRetriever(k1=CONFIG["sparse_k1"], b=CONFIG["sparse_b"])


def _build_sparse_rerank() -> CrossEncoderRerankRetriever:
    return _reranker("sparse_rerank", _sparse())


def _vector(provider_name: str) -> Retriever:
    return VectorChunkRetriever(name=provider_name, provider=provider_name)


def _hybrid(provider_name: str) -> WeightedScoreFusionRetriever:
    return WeightedScoreFusionRetriever(
        name=f"{provider_name}_hybrid",
        retrievers=(_vector(provider_name), _sparse()),
        candidate_limit=CONFIG["rerank_candidate_limit"],
        weights=(CONFIG["hybrid_vector_weight"], CONFIG["hybrid_sparse_weight"]),
    )


def _vector_rerank(provider_name: str) -> CrossEncoderRerankRetriever:
    return _reranker(f"{provider_name}_rerank", _vector(provider_name))


def _hybrid_rerank(provider_name: str) -> CrossEncoderRerankRetriever:
    return _reranker(f"{provider_name}_hybrid_rerank", _hybrid(provider_name))


def _qwen_agentic() -> AgenticRetriever:
    return AgenticRetriever(
        name="qwen_agentic",
        base_retriever=_vector("qwen"),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
    )


def _qwen_hybrid_agentic() -> AgenticRetriever:
    return AgenticRetriever(
        name="qwen_hybrid_agentic",
        base_retriever=_hybrid("qwen"),
        judge_retries=CONFIG["agentic_judge_retries"],
        max_attempts=CONFIG["agentic_max_attempts"],
        min_sufficient_chunks=CONFIG["agentic_min_sufficient_chunks"],
        initial_limit=CONFIG["agentic_initial_limit"],
        limit_step=CONFIG["agentic_limit_step"],
        max_limit=CONFIG["agentic_max_limit"],
    )


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
