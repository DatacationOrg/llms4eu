from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.retrieval.base import RankedChunk

if TYPE_CHECKING:
    from src.vector_store.chunks import ScoredChunk


@dataclass(frozen=True)
class VectorChunkRetriever:
    """Retriever adapter for one Chroma chunk-vector provider."""

    name: str
    provider: str

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        chunks = _query_chunk_vectors()(self.provider, query, limit)
        return [_ranked_chunk(chunk) for chunk in chunks]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        rankings = _query_chunk_vectors_batch()(self.provider, queries, limit)
        return {
            index: [_ranked_chunk(chunk) for chunk in chunks]
            for index, chunks in rankings.items()
        }


def _ranked_chunk(chunk: ScoredChunk) -> RankedChunk:
    return RankedChunk(id=chunk.id, score=chunk.score, text=chunk.text)


def _query_chunk_vectors() -> Callable[[str, str, int], list[ScoredChunk]]:
    from src.vector_store.chunks import query_chunk_vectors

    return query_chunk_vectors


def _query_chunk_vectors_batch() -> Callable[
    [str, list[str], int],
    dict[int, list[ScoredChunk]],
]:
    from src.vector_store.chunks import query_chunk_vectors_batch

    return query_chunk_vectors_batch
