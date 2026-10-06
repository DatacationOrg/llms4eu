from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.shared.env import load_yaml


@dataclass(frozen=True)
class FetchPagesConfig:
    workers: int
    domain_delay_seconds: float
    timeout_seconds: float
    retry_statuses: tuple[int, ...]
    browser_timeout_ms: int
    listing_min_items: int
    listing_min_context_words: int
    max_document_bytes: int
    max_document_pages: int
    document_extensions: tuple[str, ...]


@lru_cache(maxsize=1)
def fetch_pages_config() -> FetchPagesConfig:
    raw = load_yaml(Path(__file__).with_name("config.yaml"))["fetch_pages"]
    # Sequences become tuples so the frozen config stays immutable and hashable.
    return FetchPagesConfig(
        **{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in raw.items()
        }
    )
