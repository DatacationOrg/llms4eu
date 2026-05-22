from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.shared.env import load_yaml


@dataclass(frozen=True)
class FetchPagesConfig:
    db_path: str
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
    return FetchPagesConfig(
        db_path=str(raw["db_path"]),
        workers=int(raw["workers"]),
        domain_delay_seconds=float(raw["domain_delay_seconds"]),
        timeout_seconds=float(raw["timeout_seconds"]),
        retry_statuses=tuple(int(status) for status in raw["retry_statuses"]),
        browser_timeout_ms=int(raw["browser_timeout_ms"]),
        listing_min_items=int(raw["listing_min_items"]),
        listing_min_context_words=int(raw["listing_min_context_words"]),
        max_document_bytes=int(raw["max_document_bytes"]),
        max_document_pages=int(raw["max_document_pages"]),
        document_extensions=tuple(
            str(extension) for extension in raw["document_extensions"]
        ),
    )
