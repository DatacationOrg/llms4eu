from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.retrieval.base import Retriever
from src.retrieval.retrievers.geo import GeoRetriever, page_points, place_of
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
        name: provider
        for name in names
        if (provider := specs[name].provider) and not collection_ready(provider)
    }


def _specs() -> dict[str, RetrieverSpec]:
    specs = {"sparse": RetrieverSpec("sparse", SparseRetriever)}
    for suffix in CONFIG["rerankers"]:
        name = f"sparse_{suffix}"
        specs[name] = RetrieverSpec(
            name,
            lambda name=name, suffix=suffix: _reranker(name, suffix, SparseRetriever()),
        )
    for provider in enabled_provider_names():
        stages: dict[str, Callable[[], Retriever]] = {
            provider: lambda provider=provider: _vector(provider),
            f"{provider}_hybrid": lambda provider=provider: _hybrid(provider),
        }
        for stage_name, stage in list(stages.items()):
            for suffix in CONFIG["rerankers"]:
                name = f"{stage_name}_{suffix}"
                stages[name] = lambda name=name, suffix=suffix, stage=stage: _reranker(
                    name, suffix, stage()
                )
        name = f"{provider}_hybrid_rerank_geo"
        stages[name] = lambda name=name, stage=stages[f"{provider}_hybrid_rerank"]: (
            GeoRetriever(
                name=name,
                stage=stage(),
                place_of=place_of,
                page_points=page_points,
                weight=CONFIG["geo_weight"],
                overfetch=CONFIG["geo_overfetch"],
            )
        )
        for name, build in stages.items():
            specs[name] = RetrieverSpec(name, build, provider=provider)
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


def _reranker(
    name: str, suffix: str, base_retriever: Retriever
) -> CrossEncoderRerankRetriever:
    return CrossEncoderRerankRetriever(
        name=name,
        base_retriever=base_retriever,
        candidate_limit=CONFIG["rerank_candidate_limit"],
        device=CONFIG["reranker_device"],
        max_length=CONFIG["reranker_max_length"],
        local_files_only=CONFIG["reranker_local_files_only"],
        batch_size=CONFIG["reranker_batch_size"],
        **CONFIG["rerankers"][suffix],
    )
