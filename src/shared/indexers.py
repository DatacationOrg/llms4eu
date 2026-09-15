from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Protocol

from src.shared.embed import embed_texts, load_embedder
from src.shared.embedding_cache import cached_embeddings

DEFAULT_MAX_SEQ_LENGTH = 2048

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
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
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


# field -> (config key, default), or a literal for fields config never sets.
PROVIDER_FIELDS: dict[str, dict[str, object]] = {
    "qwen": {
        "model_name": ("qwen_embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        "batch_size": ("qwen_batch_size", 8),
        "local_files_only": ("qwen_local_files_only", True),
        "show_progress_bar": True,
    },
    "qwen4b": {
        "model_name": ("qwen4b_embedding_model", "Qwen/Qwen3-Embedding-4B"),
        "batch_size": ("qwen4b_batch_size", 1),
        "local_files_only": ("qwen4b_local_files_only", True),
        "show_progress_bar": True,
    },
    "nemotron": {
        "model_name": ("nemotron_embedding_model", "nvidia/Nemotron-3-Embed-1B-BF16"),
        "batch_size": ("nemotron_batch_size", 8),
        "local_files_only": ("nemotron_local_files_only", True),
        "dtype": ("nemotron_dtype", "bfloat16"),
        "attn_implementation": ("nemotron_attn_implementation", "sdpa"),
        "show_progress_bar": True,
    },
}


def provider_names() -> list[str]:
    return sorted(PROVIDER_FIELDS)


def build_indexer(name: str, config: dict | None = None) -> EmbeddingIndexer:
    if name not in PROVIDER_FIELDS:
        raise ValueError(f"Unknown indexer: {name}")
    config = config or {}
    kwargs = {
        field: config.get(*spec) if isinstance(spec, tuple) else spec
        for field, spec in PROVIDER_FIELDS[name].items()
    }
    return SentenceTransformerIndexer(
        name=name,
        max_seq_length=config.get("embedding_max_seq_length", DEFAULT_MAX_SEQ_LENGTH),
        **kwargs,
    )
