from __future__ import annotations

from pathlib import Path

from src.eval.ranking.base import RankingMethod
from src.eval.ranking.bm25 import Bm25Method
from src.eval.ranking.cross_encoder import CrossEncoderRerankMethod
from src.eval.ranking.fusion import ReciprocalRankFusionMethod
from src.eval.ranking.vector import VectorMethod
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).parents[1] / "config.yaml")

VECTOR = VectorMethod()
BM25 = Bm25Method()
VECTOR_BM25 = ReciprocalRankFusionMethod(
    name="vector_bm25",
    methods=(VECTOR, BM25),
)

METHODS: dict[str, RankingMethod] = {
    VECTOR.name: VECTOR,
    BM25.name: BM25,
    VECTOR_BM25.name: VECTOR_BM25,
    "qwen3_rerank": CrossEncoderRerankMethod(
        name="qwen3_rerank",
        model_name=CONFIG["reranker_model"],
        base_method=VECTOR,
        candidate_limit=CONFIG["default_candidate_limit"],
    ),
    "qwen3_rerank_hybrid": CrossEncoderRerankMethod(
        name="qwen3_rerank_hybrid",
        model_name=CONFIG["reranker_model"],
        base_method=VECTOR_BM25,
        candidate_limit=CONFIG["default_candidate_limit"],
    ),
}


def get_ranking_method(method_name: str) -> RankingMethod:
    if method_name not in METHODS:
        raise NameError(f"Ranking method {method_name} is unknown.")
    return METHODS[method_name]


def available_ranking_methods() -> set[str]:
    return set(METHODS)
