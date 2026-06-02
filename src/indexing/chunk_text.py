from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PageChunk:
    """Canonical chunk fields used for embedding text construction."""

    id: str
    page_id: str
    heading_path: str | None
    text: str
    title: str | None = None


class ChunkTextRepresentation(Protocol):
    """Build provider-agnostic text sent to embedding models."""

    name: str

    def text_for_embedding(self, chunk: PageChunk) -> str: ...


@dataclass(frozen=True)
class TitleHeadingChunkText:
    """Embed title, heading path, and chunk text as one document."""

    name: str = "title_heading_chunk"

    def text_for_embedding(self, chunk: PageChunk) -> str:
        parts = [chunk.title or "", chunk.heading_path or "", chunk.text]
        return "\n".join(part for part in parts if part)
