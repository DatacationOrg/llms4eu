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
