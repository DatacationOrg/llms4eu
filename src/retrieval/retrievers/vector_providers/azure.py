from __future__ import annotations

import os
from dataclasses import dataclass

from src.retrieval.retrievers.vector_providers._base_provider import ChunkVectorProvider
from src.shared.env import load_local_env

REQUIRED_ENV = ("AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY", "AZURE_EMBEDDING_MODEL")


@dataclass(frozen=True)
class AzureProvider(ChunkVectorProvider):
    provider: str = "azure"

    def _before_rebuild(self) -> None:
        load_local_env()
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "Azure retrieval requires these environment variables: "
                + ", ".join(missing)
            )
        if not _confirm()(
            "Do you want to also run Azure embeddings and spend quota? [yes/no]: ",
            default=False,
        ):
            raise RuntimeError("Azure retrieval setup cancelled.")


def _confirm():
    try:
        from src.shared.cli import confirm as shared_confirm
    except ImportError:
        return _local_confirm
    return lambda message, default=False: shared_confirm(message)


def _local_confirm(message: str, default: bool = False) -> bool:
    answer = input(f"{message} [yes/no] ").strip().casefold()
    if not answer:
        return default
    return answer == "yes"
