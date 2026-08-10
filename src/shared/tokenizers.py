from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Callable

from src.shared.env import load_yaml
from src.shared.indexers import PROVIDER_SPECS

__all__ = [
    "huggingface_tokenizer",
    "provider_model_name",
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
    if provider not in PROVIDER_SPECS:
        raise ValueError(f"Unknown embedding provider: {provider}")
    indexer = PROVIDER_SPECS[provider].build(INDEXING_CONFIG)
    return indexer.model_name


def provider_token_limit(provider: str) -> int:
    """Sequence length an embedding provider truncates at."""
    if provider not in PROVIDER_SPECS:
        raise ValueError(f"Unknown embedding provider: {provider}")
    indexer = PROVIDER_SPECS[provider].build(INDEXING_CONFIG)
    return int(indexer.max_seq_length or 512)


def huggingface_tokenizer(provider: str):
    """The provider's tokenizer, for splitters that take one directly."""
    return _tokenizer(provider_model_name(provider))


def token_counter(provider: str) -> Callable[[str], int]:
    """Count tokens the way `provider` will count them when embedding.

    Loads only the tokenizer, never the model weights, so chunking stays cheap
    and needs no GPU.
    """
    tokenizer = _tokenizer(provider_model_name(provider))

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


@cache
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)
