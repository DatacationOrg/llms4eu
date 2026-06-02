from __future__ import annotations

from typing import Protocol

from src.retrieval.base import Retriever


class VectorProvider(Protocol):
    """Lazy binding from provider name to retriever plus readiness checks."""

    def build_retriever(self, name: str) -> Retriever: ...

    def ensure_ready(self) -> None: ...
