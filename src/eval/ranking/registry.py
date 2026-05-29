from __future__ import annotations

from pathlib import Path

from src.eval.ranking.base import RankingMethod
from src.eval.ranking.bm25 import Bm25Method
from src.eval.ranking.cross_encoder import CrossEncoderRerankMethod
from src.eval.ranking.fusion import ReciprocalRankFusionMethod
from src.eval.ranking.vector import VectorMethod
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).parents[1] / "config.yaml")
VECTOR_METHODS = ("english", "qwen", "azure")
CONTENT_MODES = ("chunk", "summary", "chunk_summary")

BM25 = Bm25Method()
VECTORS = {
    f"{indexer}_{content_mode}": VectorMethod(
        name=f"{indexer}_{content_mode}",
        indexer=indexer,
        content_mode=content_mode,
    )
    for indexer in VECTOR_METHODS
    for content_mode in CONTENT_MODES
}
HYBRIDS = {
    name: ReciprocalRankFusionMethod(name=f"{name}_bm25", methods=(method, BM25))
    for name, method in VECTORS.items()
}


def _reranker(name: str, base_method: RankingMethod) -> CrossEncoderRerankMethod:
    return CrossEncoderRerankMethod(
        name=name,
        model_name=CONFIG["reranker_model"],
        base_method=base_method,
        candidate_limit=CONFIG["default_candidate_limit"],
        batch_size=CONFIG.get("reranker_batch_size", 64),
    )


METHODS: dict[str, RankingMethod] = {
    BM25.name: BM25,
    **VECTORS,
    **{method.name: method for method in HYBRIDS.values()},
    **{
        f"{name}_rerank": _reranker(f"{name}_rerank", method)
        for name, method in VECTORS.items()
    },
    **{
        f"{name}_rerank_hybrid": _reranker(f"{name}_rerank_hybrid", method)
        for name, method in HYBRIDS.items()
    },
}


def get_ranking_method(method_name: str) -> RankingMethod:
    if method_name not in METHODS:
        raise NameError(f"Ranking method {method_name} is unknown.")
    return METHODS[method_name]


def available_ranking_methods() -> set[str]:
    return set(METHODS)
