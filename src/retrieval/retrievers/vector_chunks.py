from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.indexing.chunk_text import LEGACY_CHUNK_VERSION
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.retrieval.base import RankedChunk

if TYPE_CHECKING:
    from src.vector_store.chunks import ScoredChunk


@dataclass(frozen=True)
class VectorChunkRetriever:
    """Retriever adapter for one Chroma chunk-vector provider."""

    name: str
    provider: str
    chunk_version: str = LEGACY_CHUNK_VERSION
    variant: str = BASE_CHUNK_VARIANT
    # Chroma metadata filter (`GeoScope.chroma_where()`); None for no filter.
    where: dict | None = None

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        chunks = _query_chunk_vectors()(
            self.provider,
            query,
            limit,
            self.chunk_version,
            self.variant,
            where=self.where,
        )
        return [_ranked_chunk(chunk) for chunk in chunks]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        rankings = _query_chunk_vectors_batch()(
            self.provider,
            queries,
            limit,
            self.chunk_version,
            self.variant,
            where=self.where,
        )
        return {
            index: [_ranked_chunk(chunk) for chunk in chunks]
            for index, chunks in rankings.items()
        }


def _ranked_chunk(chunk: ScoredChunk) -> RankedChunk:
    return RankedChunk(id=chunk.id, score=chunk.score, text=chunk.text)


def _query_chunk_vectors() -> Callable[..., list[ScoredChunk]]:
    from src.vector_store.chunks import query_chunk_vectors

    return query_chunk_vectors


def _query_chunk_vectors_batch() -> Callable[..., dict[int, list[ScoredChunk]]]:
    from src.vector_store.chunks import query_chunk_vectors_batch

    return query_chunk_vectors_batch
