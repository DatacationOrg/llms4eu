from __future__ import annotations

from dataclasses import dataclass

from src.eval.ranking.base import RankedChunk, RankingMethod


@dataclass(frozen=True)
class ReciprocalRankFusionMethod:
    name: str
    methods: tuple[RankingMethod, ...]
    candidate_limit: int = 50
    k: int = 60

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        by_id: dict[str, RankedChunk] = {}
        scores: dict[str, float] = {}
        for method in self.methods:
            for rank, chunk in enumerate(
                method.retrieve(query, self.candidate_limit),
                start=1,
            ):
                by_id[chunk.id] = chunk
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (self.k + rank)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [
            RankedChunk(id=chunk_id, score=scores[chunk_id], text=by_id[chunk_id].text)
            for chunk_id in ranked_ids[:limit]
        ]

    def retrieve_many(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        fused: dict[int, dict[str, float]] = {
            index: {} for index in range(len(queries))
        }
        chunks_by_query: dict[int, dict[str, RankedChunk]] = {
            index: {} for index in range(len(queries))
        }
        for method in self.methods:
            rankings = _retrieve_many(method, queries, self.candidate_limit)
            for query_index, chunks in rankings.items():
                for rank, chunk in enumerate(chunks, start=1):
                    chunks_by_query[query_index][chunk.id] = chunk
                    fused[query_index][chunk.id] = fused[query_index].get(
                        chunk.id, 0.0
                    ) + 1 / (self.k + rank)
        return {
            query_index: [
                RankedChunk(
                    id=chunk_id,
                    score=scores[chunk_id],
                    text=chunks_by_query[query_index][chunk_id].text,
                )
                for chunk_id in sorted(scores, key=scores.get, reverse=True)[:limit]
            ]
            for query_index, scores in fused.items()
        }


def _retrieve_many(
    method: RankingMethod,
    queries: list[str],
    limit: int,
) -> dict[int, list[RankedChunk]]:
    if hasattr(method, "retrieve_many"):
        return method.retrieve_many(queries, limit)
    return {index: method.retrieve(query, limit) for index, query in enumerate(queries)}
