from __future__ import annotations

from dataclasses import dataclass

from src.eval.ranking.base import RankedChunk
from src.indexing.chunks import search_chunk_vectors, search_chunk_vectors_batch


@dataclass(frozen=True)
class VectorMethod:
    name: str
    indexer: str
    content_mode: str = "chunk_summary"

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        return [
            RankedChunk(id=chunk.id, score=chunk.score, text=chunk.text)
            for chunk in search_chunk_vectors(
                query,
                limit,
                method=self.indexer,
                content_mode=self.content_mode,
            )
        ]

    def retrieve_many(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        rankings = search_chunk_vectors_batch(
            queries,
            limit,
            method=self.indexer,
            content_mode=self.content_mode,
        )
        return {
            index: [
                RankedChunk(id=chunk.id, score=chunk.score, text=chunk.text)
                for chunk in chunks
            ]
            for index, chunks in rankings.items()
        }
