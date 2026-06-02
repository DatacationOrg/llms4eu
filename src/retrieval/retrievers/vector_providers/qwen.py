from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.retrievers.vector_providers._base_provider import ChunkVectorProvider


@dataclass(frozen=True)
class QwenProvider(ChunkVectorProvider):
    provider: str = "qwen"
