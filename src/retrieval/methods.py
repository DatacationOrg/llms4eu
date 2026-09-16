from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.retrieval.base import Retriever
from src.retrieval.retrievers.fusion import WeightedScoreFusionRetriever
from src.retrieval.retrievers.rerank import CrossEncoderRerankRetriever
from src.retrieval.retrievers.sparse import SparseRetriever
from src.retrieval.retrievers.vector import VectorChunkRetriever
from src.shared.env import load_yaml
from src.indexing.store import collection_ready, enabled_provider_names

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


@dataclass(frozen=True)
class RetrieverSpec:
    """Lazy catalog entry for one public retrieval method."""

    name: str
    build: Callable[[], Retriever]
    provider: str | None = None


class MissingRetrieverIndexes(RuntimeError):
    def __init__(self, missing: dict[str, str]) -> None:
        self.missing = missing
        commands = "\n".join(
            f"  uv run python -m src.indexing --method {provider}"
            for provider in sorted(set(missing.values()))
        )
        super().__init__(
            "Missing vector indexes for retrieval methods: "
            f"{', '.join(sorted(missing))}\nBuild them first:\n{commands}"
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

    return {
        name: specs[name].provider
        for name in names
        if specs[name].provider and not collection_ready(specs[name].provider)
    }


def _specs() -> dict[str, RetrieverSpec]:
    specs = {
        "sparse": RetrieverSpec("sparse", SparseRetriever),
        "sparse_rerank": RetrieverSpec(
            "sparse_rerank",
            lambda: _reranker("sparse_rerank", SparseRetriever()),
        ),
    }
    builders = {
        "": _vector,
        "_hybrid": _hybrid,
        "_rerank": _vector_rerank,
        "_hybrid_rerank": _hybrid_rerank,
    }
    for provider in enabled_provider_names():
        for tag, build in builders.items():
            name = f"{provider}{tag}"
            specs[name] = RetrieverSpec(
                name,
                lambda build=build, provider=provider: build(provider),
                provider=provider,
            )
    return specs


def _vector(provider_name: str) -> Retriever:
    return VectorChunkRetriever(name=provider_name, provider=provider_name)


def _hybrid(provider_name: str) -> WeightedScoreFusionRetriever:
    return WeightedScoreFusionRetriever(
        name=f"{provider_name}_hybrid",
        retrievers=(_vector(provider_name), SparseRetriever()),
        candidate_limit=CONFIG["rerank_candidate_limit"],
        weights=(CONFIG["hybrid_vector_weight"], CONFIG["hybrid_sparse_weight"]),
    )


def _vector_rerank(provider_name: str) -> CrossEncoderRerankRetriever:
    return _reranker(f"{provider_name}_rerank", _vector(provider_name))


def _hybrid_rerank(provider_name: str) -> CrossEncoderRerankRetriever:
    return _reranker(f"{provider_name}_hybrid_rerank", _hybrid(provider_name))


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
    )
