from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import cache
from typing import Callable, Protocol

import httpx

from src.shared.embed import embed_texts, load_embedder
from src.shared.embedding_cache import cached_embeddings

__all__ = [
    "AzureEmbeddingIndexer",
    "EmbeddingIndexer",
    "EnglishMiniLmIndexer",
    "EmbeddingProviderSpec",
    "Qwen4BIndexer",
    "QwenMultilingualIndexer",
    "build_indexer",
    "provider_names",
]


class EmbeddingIndexer(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class AzureEmbeddingRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingProviderSpec:
    name: str
    build: Callable[[dict], EmbeddingIndexer]
    remote: bool = False
    max_batch_size: int | None = None
    required_env: tuple[str, ...] = ()


@dataclass(frozen=True)
class SentenceTransformerIndexer:
    name: str
    model_name: str
    local_files_only: bool = True
    batch_size: int | None = None
    max_seq_length: int | None = None
    show_progress_bar: bool = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return cached_embeddings(
            provider=self.name,
            model=self.model_name,
            kind="document",
            texts=texts,
            embed_missing=lambda missing: embed_texts(
                self._model(),
                missing,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
            ),
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return cached_embeddings(
            provider=self.name,
            model=self.model_name,
            kind="query",
            texts=texts,
            embed_missing=lambda missing: embed_texts(
                self._model(), missing, prompt_name=self._query_prompt()
            ),
        )

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
class Qwen4BIndexer(SentenceTransformerIndexer):
    name: str = "qwen4b"
    model_name: str = "Qwen/Qwen3-Embedding-4B"
    batch_size: int | None = 1
    max_seq_length: int | None = 512
    show_progress_bar: bool = True


@dataclass(frozen=True)
class AzureEmbeddingIndexer:
    name: str = "azure"
    endpoint: str | None = None
    api_key: str | None = None
    model: str | None = None
    batch_size: int = 96
    timeout_seconds: int = 120
    retries: int = 6
    retry_backoff_max_seconds: int = 60

    @classmethod
    def from_env(cls, config: dict | None = None) -> AzureEmbeddingIndexer:
        config = config or {}
        return cls(
            endpoint=os.environ["AZURE_AI_ENDPOINT"],
            api_key=os.environ["AZURE_AI_API_KEY"],
            model=os.environ["AZURE_EMBEDDING_MODEL"],
            batch_size=min(config.get("azure_embedding_batch_size", 96), 96),
            timeout_seconds=config.get("azure_embedding_timeout_seconds", 120),
            retries=config.get("azure_embedding_retries", 6),
            retry_backoff_max_seconds=config.get(
                "azure_embedding_retry_backoff_max_seconds",
                60,
            ),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return cached_embeddings(
            provider=self.name,
            model=self._model(),
            kind="document",
            texts=texts,
            embed_missing=self._embed_uncached,
        )

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
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
        return cached_embeddings(
            provider=self.name,
            model=self._model(),
            kind="query",
            texts=texts,
            embed_missing=self._embed_uncached,
        )

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
                _raise_for_azure_error(response, batch_size=len(batch))
                return response
            if attempt == self.retries:
                _raise_for_azure_error(response, batch_size=len(batch))
            wait_seconds = _retry_after(response) or min(
                2**attempt,
                self.retry_backoff_max_seconds,
            )
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


def _raise_for_azure_error(response: httpx.Response, batch_size: int) -> None:
    if response.status_code < 400:
        return
    message = _response_message(response)
    raise AzureEmbeddingRequestError(message)


def _response_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    return f"Azure embeddings failed with HTTP {response.status_code}: {payload}"


PROVIDER_SPECS: dict[str, EmbeddingProviderSpec] = {
    "english": EmbeddingProviderSpec(
        name="english",
        build=lambda config: EnglishMiniLmIndexer(
            model_name=config.get(
                "english_embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        ),
    ),
    "qwen": EmbeddingProviderSpec(
        name="qwen",
        build=lambda config: QwenMultilingualIndexer(
            model_name=config.get("qwen_embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
            batch_size=config.get("qwen_batch_size", 8),
            max_seq_length=config.get("embedding_max_seq_length", 512),
            local_files_only=config.get("qwen_local_files_only", True),
        ),
    ),
    "qwen4b": EmbeddingProviderSpec(
        name="qwen4b",
        build=lambda config: Qwen4BIndexer(
            model_name=config.get("qwen4b_embedding_model", "Qwen/Qwen3-Embedding-4B"),
            batch_size=config.get("qwen4b_batch_size", 1),
            max_seq_length=config.get("embedding_max_seq_length", 512),
            local_files_only=config.get("qwen4b_local_files_only", True),
        ),
    ),
    "azure": EmbeddingProviderSpec(
        name="azure",
        build=AzureEmbeddingIndexer.from_env,
        remote=True,
        max_batch_size=96,
        required_env=("AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY", "AZURE_EMBEDDING_MODEL"),
    ),
}


def provider_names() -> list[str]:
    return sorted(PROVIDER_SPECS)


def build_indexer(name: str, config: dict | None = None) -> EmbeddingIndexer:
    if name in PROVIDER_SPECS:
        return PROVIDER_SPECS[name].build(config or {})
    raise ValueError(f"Unknown indexer: {name}")
