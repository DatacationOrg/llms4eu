from dataclasses import dataclass

from src.retrieval.base import RankedChunk
from src.retrieval.retrievers.fusion import WeightedScoreFusionRetriever


@dataclass(frozen=True)
class StaticMethod:
    name: str
    chunks: list[RankedChunk]

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        return self.chunks[:limit]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        return {
            index: self.retrieve(query, limit) for index, query in enumerate(queries)
        }


def test_weighted_score_fusion_normalizes_each_method_before_combining():
    vector = StaticMethod(
        "vector",
        [
            RankedChunk("a", 0.9, "a"),
            RankedChunk("b", 0.8, "b"),
        ],
    )
    sparse = StaticMethod(
        "sparse",
        [
            RankedChunk("b", 100.0, "b"),
            RankedChunk("a", 10.0, "a"),
        ],
    )
    method = WeightedScoreFusionRetriever(
        "hybrid",
        retrievers=(vector, sparse),
        candidate_limit=30,
        weights=(0.7, 0.3),
    )

    result = method.retrieve("query", 2)

    assert [chunk.id for chunk in result] == ["a", "b"]
    assert result[0].score == 0.7
    assert result[1].score == 0.3


def test_weighted_score_fusion_batches_queries():
    vector = StaticMethod(
        "vector",
        [
            RankedChunk("a", 0.9, "a"),
            RankedChunk("b", 0.8, "b"),
        ],
    )
    sparse = StaticMethod(
        "sparse",
        [
            RankedChunk("b", 100.0, "b"),
            RankedChunk("a", 10.0, "a"),
        ],
    )
    method = WeightedScoreFusionRetriever(
        "hybrid",
        retrievers=(vector, sparse),
        candidate_limit=30,
        weights=(0.7, 0.3),
    )

    result = method.retrieve_batch(["first", "second"], 2)

    assert [chunk.id for chunk in result[0]] == ["a", "b"]
    assert [chunk.id for chunk in result[1]] == ["a", "b"]
