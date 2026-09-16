from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.base import RankedChunk
from src.indexing.store import query_chunk_vectors, query_chunk_vectors_batch


@dataclass(frozen=True)
class VectorChunkRetriever:
    """Retriever adapter for one Chroma chunk-vector provider."""

    name: str
    provider: str

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        return query_chunk_vectors(self.provider, query, limit)

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        return query_chunk_vectors_batch(self.provider, queries, limit)
