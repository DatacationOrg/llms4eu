from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RankedChunk:
    """Chunk result returned by every retriever implementation."""

    id: str
    score: float
    text: str


class Retriever(Protocol):
    """Common retrieval contract used by RAG and eval."""

    name: str

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]: ...

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]: ...


def retrieve_batch_default(
    retriever: Retriever,
    queries: list[str],
    limit: int,
) -> dict[int, list[RankedChunk]]:
    return {
        index: retriever.retrieve(query, limit) for index, query in enumerate(queries)
    }
