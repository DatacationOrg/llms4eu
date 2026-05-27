from __future__ import annotations

from dataclasses import dataclass

from src.eval.ranking.base import RankedChunk
from src.eval.vector_index import search_chunk_vectors


@dataclass(frozen=True)
class VectorMethod:
    name: str = "vector"

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        return [
            RankedChunk(id=chunk.id, score=chunk.score, text=chunk.text)
            for chunk in search_chunk_vectors(query, limit)
        ]
