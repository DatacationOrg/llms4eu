from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable, Protocol

from src.shared.embed import embed_texts, load_embedder
from src.shared.embedding_cache import cached_embeddings

__all__ = [
    "EmbeddingIndexer",
    "EnglishMiniLmIndexer",
    "EmbeddingProviderSpec",
    "Nemotron3EmbedIndexer",
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


@dataclass(frozen=True)
class EmbeddingProviderSpec:
    name: str
    build: Callable[[dict], EmbeddingIndexer]


@dataclass(frozen=True)
class SentenceTransformerIndexer:
    name: str
    model_name: str
    local_files_only: bool = True
    batch_size: int | None = None
    max_seq_length: int | None = None
    show_progress_bar: bool = False
    dtype: str | None = None
    attn_implementation: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return cached_embeddings(
            provider=self.name,
            model=self.model_name,
            kind="document",
            texts=texts,
            embed_missing=lambda missing: embed_texts(
                self._model(),
                missing,
                prompt_name=self._document_prompt(),
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
        model = load_embedder(
            self.model_name,
            local_files_only=self.local_files_only,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        )
        if self.max_seq_length is not None:
            model.max_seq_length = int(self.max_seq_length)
        return model

    def _query_prompt(self) -> str | None:
        prompts = self._model().prompts or {}
        return "query" if "query" in prompts else None

    def _document_prompt(self) -> str | None:
        prompts = self._model().prompts or {}
        return "document" if "document" in prompts else None


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
class Nemotron3EmbedIndexer(SentenceTransformerIndexer):
    name: str = "nemotron"
    model_name: str = "nvidia/Nemotron-3-Embed-1B-BF16"
    batch_size: int | None = 8
    max_seq_length: int | None = 4096
    show_progress_bar: bool = True
    dtype: str | None = "bfloat16"
    attn_implementation: str | None = "sdpa"


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
    "nemotron": EmbeddingProviderSpec(
        name="nemotron",
        build=lambda config: Nemotron3EmbedIndexer(
            model_name=config.get(
                "nemotron_embedding_model",
                "nvidia/Nemotron-3-Embed-1B-BF16",
            ),
            batch_size=config.get("nemotron_batch_size", 8),
            max_seq_length=config.get("nemotron_max_seq_length", 4096),
            local_files_only=config.get("nemotron_local_files_only", True),
            dtype=config.get("nemotron_dtype", "bfloat16"),
            attn_implementation=config.get(
                "nemotron_attn_implementation",
                "sdpa",
            ),
        ),
    ),
}


def provider_names() -> list[str]:
    return sorted(PROVIDER_SPECS)


def build_indexer(name: str, config: dict | None = None) -> EmbeddingIndexer:
    if name in PROVIDER_SPECS:
        return PROVIDER_SPECS[name].build(config or {})
    raise ValueError(f"Unknown indexer: {name}")
