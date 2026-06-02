from __future__ import annotations

from src.retrieval.retrievers.vector_providers.azure import AzureProvider
from src.retrieval.retrievers.vector_providers.base import VectorProvider
from src.retrieval.retrievers.vector_providers.english import EnglishProvider
from src.retrieval.retrievers.vector_providers.qwen import QwenProvider
from src.retrieval.retrievers.vector_providers.qwen4b import Qwen4BProvider

PROVIDER_CLASSES: dict[str, type[VectorProvider]] = {
    "azure": AzureProvider,
    "english": EnglishProvider,
    "qwen": QwenProvider,
    "qwen4b": Qwen4BProvider,
}


def provider_names() -> list[str]:
    return sorted(PROVIDER_CLASSES)


def build_provider(name: str) -> VectorProvider:
    if name not in PROVIDER_CLASSES:
        raise ValueError(f"Unknown vector provider: {name}")
    return PROVIDER_CLASSES[name]()
