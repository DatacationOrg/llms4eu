from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.base import RankedChunk, Retriever


@dataclass(frozen=True)
class WeightedScoreFusionRetriever:
    """Combine retriever scores after per-query min-max normalization."""

    name: str
    retrievers: tuple[Retriever, ...]
    candidate_limit: int
    weights: tuple[float, ...]

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        by_id: dict[str, RankedChunk] = {}
        scores: dict[str, float] = {}
        for weight, retriever in zip(self.weights, self.retrievers, strict=True):
            chunks = retriever.retrieve(query, self.candidate_limit)
            by_id.update({chunk.id: chunk for chunk in chunks})
            for chunk_id, score in _normalized_scores(chunks).items():
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * score

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [
            RankedChunk(id=chunk_id, score=scores[chunk_id], text=by_id[chunk_id].text)
            for chunk_id in ranked_ids[:limit]
        ]

    def retrieve_batch(
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
        for weight, retriever in zip(self.weights, self.retrievers, strict=True):
            rankings = retriever.retrieve_batch(queries, self.candidate_limit)
            for query_index, chunks in rankings.items():
                chunks_by_query[query_index].update(
                    {chunk.id: chunk for chunk in chunks}
                )
                for chunk_id, score in _normalized_scores(chunks).items():
                    fused[query_index][chunk_id] = (
                        fused[query_index].get(chunk_id, 0.0) + weight * score
                    )
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


def _normalized_scores(chunks: list[RankedChunk]) -> dict[str, float]:
    if not chunks:
        return {}
    values = [chunk.score for chunk in chunks]
    low = min(values)
    high = max(values)
    if high == low:
        return {chunk.id: 1.0 for chunk in chunks}
    return {chunk.id: (chunk.score - low) / (high - low) for chunk in chunks}
