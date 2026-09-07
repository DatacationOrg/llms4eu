from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Callable

from src.shared.env import load_yaml
from src.shared.indexers import build_indexer

__all__ = [
    "huggingface_tokenizer",
    "provider_model_name",
    "provider_revision",
    "provider_token_limit",
    "token_counter",
]

INDEXING_CONFIG = load_yaml(
    Path(__file__).resolve().parents[1] / "indexing" / "config.yaml"
)


def provider_model_name(provider: str) -> str:
    """Model id an embedding provider tokenizes with.

    Reads the same config keys `build_indexer` uses, so a model override in
    `src/indexing/config.yaml` also moves the tokenizer.
    """
    return _spec(provider).model_name


def provider_revision(provider: str) -> str | None:
    """Cached snapshot an embedding provider is pinned to, if any.

    Every by-name lookup into the HF cache resolves through `refs/main`, and a
    model downloaded by explicit revision has no `refs/` directory at all — so
    without the pin, loading a tokenizer or reading a cached config falls through
    to the network and fails offline, while the weights sit on disk and the model
    itself loads fine. That asymmetry is why this is threaded separately rather
    than left to `from_pretrained` defaults.
    """
    return _spec(provider).revision


def provider_token_limit(provider: str) -> int:
    """Sequence length an embedding provider truncates at.

    The configured value is a ceiling the project chooses; the model's own
    `sentence_bert_config.json` is a floor it cannot exceed. MiniLM is the case
    that matters: it is configured at 512 but truncates at 256, so reporting the
    configured value alone would understate its truncation by half.
    """
    configured = _spec(provider).max_seq_length
    model_limit = _model_max_seq_length(
        provider_model_name(provider), provider_revision(provider)
    )
    candidates = [value for value in (configured, model_limit) if value]
    return int(min(candidates)) if candidates else 512


@cache
def _model_max_seq_length(model_name: str, revision: str | None = None) -> int | None:
    """`max_seq_length` from a cached sentence-transformers config, if any."""
    import json

    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(
        model_name, "sentence_bert_config.json", revision=revision
    )
    if not isinstance(path, str):
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))["max_seq_length"]
    except (OSError, ValueError, KeyError):
        return None
    # Some configs carry a sentinel rather than a real bound.
    return int(value) if 0 < int(value) < 1_000_000 else None


def huggingface_tokenizer(provider: str):
    """The provider's tokenizer, for splitters that take one directly."""
    return _tokenizer(provider_model_name(provider), provider_revision(provider))


def token_counter(provider: str) -> Callable[[str], int]:
    """Count tokens the way `provider` will count them when embedding.

    Loads only the tokenizer, never the model weights, so chunking stays cheap
    and needs no GPU.
    """
    tokenizer = _tokenizer(provider_model_name(provider), provider_revision(provider))

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def _spec(provider: str):
    return build_indexer(provider, INDEXING_CONFIG)


@cache
def _tokenizer(model_name: str, revision: str | None = None):
    from transformers import AutoTokenizer

    # Local-only, matching every indexer: chunking must never trigger a download.
    return AutoTokenizer.from_pretrained(
        model_name, local_files_only=True, revision=revision
    )
