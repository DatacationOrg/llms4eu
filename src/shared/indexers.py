from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import httpx

from src.shared.embed import embed_texts, load_embedder

__all__ = [
    "AzureEmbeddingIndexer",
    "EmbeddingIndexer",
    "EnglishMiniLmIndexer",
    "QwenMultilingualIndexer",
    "build_indexer",
]


class EmbeddingIndexer(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class SentenceTransformerIndexer:
    name: str
    model_name: str
    local_files_only: bool = True
    batch_size: int | None = None
    max_seq_length: int | None = None
    show_progress_bar: bool = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(
            self._model(),
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(self._model(), texts, prompt_name=self._query_prompt())

    @cache
    def _model(self):
        model = load_embedder(self.model_name, local_files_only=self.local_files_only)
        if self.max_seq_length is not None:
            model.max_seq_length = int(self.max_seq_length)
        return model

    def _query_prompt(self) -> str | None:
        prompts = self._model().prompts or {}
        return "query" if "query" in prompts else None


@dataclass(frozen=True)
class EnglishMiniLmIndexer(SentenceTransformerIndexer):
    name: str = "english"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class QwenMultilingualIndexer(SentenceTransformerIndexer):
    name: str = "qwen"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int | None = 8
    max_seq_length: int | None = 512
    show_progress_bar: bool = True


@dataclass(frozen=True)
class AzureEmbeddingIndexer:
    name: str = "azure"
    endpoint: str | None = None
    api_key: str | None = None
    model: str | None = None
    batch_size: int = 16
    timeout_seconds: int = 120
    retries: int = 6

    @classmethod
    def from_env(cls, config: dict | None = None) -> AzureEmbeddingIndexer:
        config = config or {}
        return cls(
            endpoint=os.environ["AZURE_AI_ENDPOINT"],
            api_key=os.environ["AZURE_AI_API_KEY"],
            model=os.environ["AZURE_EMBEDDING_MODEL"],
            batch_size=config.get("azure_embedding_batch_size", 16),
            retries=config.get("azure_embedding_retries", 6),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = self._post_with_retries(client, batch)
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in data)
                print(
                    f"embedded {min(start + self.batch_size, len(texts))}/{len(texts)}"
                )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def _endpoint(self) -> str:
        return (self.endpoint or os.environ["AZURE_AI_ENDPOINT"]).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key or os.environ['AZURE_AI_API_KEY']}",
            "Content-Type": "application/json",
        }

    def _model(self) -> str:
        return self.model or os.environ["AZURE_EMBEDDING_MODEL"]

    def _post_with_retries(
        self,
        client: httpx.Client,
        batch: list[str],
    ) -> httpx.Response:
        for attempt in range(self.retries + 1):
            response = client.post(
                f"{self._endpoint()}/embeddings",
                headers=self._headers(),
                json={"model": self._model(), "input": batch},
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response
            if attempt == self.retries:
                response.raise_for_status()
            wait_seconds = _retry_after(response) or min(2**attempt, 60)
            print(f"azure embedding rate limited; retrying in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
        raise RuntimeError("unreachable azure retry state")


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_indexer(name: str, config: dict | None = None) -> EmbeddingIndexer:
    config = config or {}
    if name == "english":
        return EnglishMiniLmIndexer(
            model_name=config.get(
                "english_embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        )
    if name == "qwen":
        return QwenMultilingualIndexer(
            model_name=config.get("qwen_embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
            max_seq_length=config.get("embedding_max_seq_length", 512),
            local_files_only=config.get("qwen_local_files_only", True),
        )
    if name == "azure":
        return AzureEmbeddingIndexer.from_env(config)
    raise ValueError(f"Unknown indexer: {name}")
