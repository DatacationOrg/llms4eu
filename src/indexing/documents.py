from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageChunk:
    """Canonical chunk fields used for embedding text construction."""

    id: str
    page_id: str
    heading_path: str | None
    text: str
    title: str | None = None
    chunk_index: int = 0
    source: str | None = None
    language: str | None = None
    page_kind: str | None = None


def text_for_embedding(chunk: PageChunk) -> str:
    """Embed the page title, heading breadcrumbs, and the chunk text as one document."""
    parts = [chunk.title or "", chunk.heading_path or "", chunk.text]
    return "\n".join(part for part in parts if part)
