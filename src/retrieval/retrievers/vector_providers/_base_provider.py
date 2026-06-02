from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.retrievers.vector_chunks import VectorChunkRetriever
from src.vector_store.chunks import collection_ready, rebuild_chunk_collection


@dataclass(frozen=True)
class ChunkVectorProvider:
    """Base provider for regenerable chunk-vector collections."""

    provider: str

    def build_retriever(self, name: str) -> VectorChunkRetriever:
        return VectorChunkRetriever(name=name, provider=self.provider)

    def ensure_ready(self) -> None:
        if collection_ready(self.provider):
            return
        self._before_rebuild()
        print(f"Building missing chunk vector collection for {self.provider}.")
        rebuild_chunk_collection(self.provider)

    def _before_rebuild(self) -> None:
        pass
