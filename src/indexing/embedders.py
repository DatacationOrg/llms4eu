from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol


import torch
from sentence_transformers import SentenceTransformer

from src.indexing.cache import cached_embeddings
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

__all__ = [
    "EmbeddingIndexer",
    "SentenceTransformerIndexer",
    "build_indexer",
    "provider_names",
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
    max_seq_length: int = 0
    show_progress_bar: bool = True
    dtype: str | None = None
    attn_implementation: str | None = None
    revision: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return cached_embeddings(
            provider=self.name,
            model=self.model_name,
            kind="document",
            texts=texts,
            embed_missing=lambda missing: embed_texts(
                self._model(),
                missing,
                prompt_name=self._prompt_name("document"),
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
                self._model(), missing, prompt_name=self._prompt_name("query")
            ),
        )

    @cache
    def _model(self):
        model = load_embedder(
            self.model_name,
            local_files_only=self.local_files_only,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            revision=self.revision,
        )
        # One shared limit for every provider. A model that cannot reach it would be
        # compared truncated against untruncated rivals, so refuse rather than skew.
        supported = int(model.max_seq_length)
        if supported < self.max_seq_length:
            raise ValueError(
                f"{self.name} ({self.model_name}) caps out at {supported} tokens, under "
                f"the shared embedding_max_seq_length of {self.max_seq_length}. Raise the "
                "model's limit or drop it from the comparison."
            )
        model.max_seq_length = self.max_seq_length
        return model

    def _prompt_name(self, kind: str) -> str | None:
        prompts = self._model().prompts or {}
        return kind if kind in prompts else None


def provider_names() -> list[str]:
    return sorted(CONFIG["providers"])


def build_indexer(name: str, config: dict | None = None) -> EmbeddingIndexer:
    config = CONFIG if config is None else config
    if name not in config["providers"]:
        raise ValueError(f"Unknown indexer: {name}")
    return SentenceTransformerIndexer(
        name=name,
        max_seq_length=config["embedding_max_seq_length"],
        **config["providers"][name],
    )


def load_embedder(
    model_name: str,
    *,
    local_files_only: bool = True,
    dtype: str | None = None,
    attn_implementation: str | None = None,
    revision: str | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformers model from the local cache by default.

    Use local_files_only=False only for explicit model download/cache warmup.
    """
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["dtype"] = getattr(torch, dtype)
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    return SentenceTransformer(
        model_name,
        local_files_only=local_files_only,
        model_kwargs=model_kwargs,
        revision=revision,
    )


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    prompt_name: str | None = None,
    batch_size: int | None = None,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    # Normalized vectors make Qdrant cosine scores comparable across queries.
    # prompt_name can be e.g. "query" for instruction-aware models like Qwen3-Embedding.
    kwargs: dict = {
        "normalize_embeddings": True,
        "show_progress_bar": show_progress_bar,
    }
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    return model.encode(texts, **kwargs).tolist()
