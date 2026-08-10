from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


LEGACY_CHUNK_VERSION = "v1"
CONTEXTUAL_CHUNK_VERSION = "v2"
CHUNK_VERSIONS = (LEGACY_CHUNK_VERSION, CONTEXTUAL_CHUNK_VERSION)

# A chunk *variant* is a way of cutting pages into chunks (sizes, overlap, unit).
# A chunk *version* is how one chunk is turned into text for embedding. They are
# independent: any variant can be indexed with any version.
BASE_CHUNK_VARIANT = "base"


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
    variant: str = BASE_CHUNK_VARIANT


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


@dataclass(frozen=True)
class MetadataContextChunkText:
    """Situate a chunk with deterministic page metadata for retrieval."""

    name: str = "metadata_context_chunk"

    def text_for_embedding(self, chunk: PageChunk) -> str:
        context = [
            ("Document", chunk.title),
            ("Section", chunk.heading_path),
            ("Source language", chunk.language),
            ("Source collection", chunk.source),
            ("Document type", chunk.page_kind),
        ]
        lines = [f"{label}: {value}" for label, value in context if value]
        lines.append(f"Content:\n{chunk.text}")
        return "\n".join(lines)


def chunk_text_representation(version: str) -> ChunkTextRepresentation:
    if version == LEGACY_CHUNK_VERSION:
        return TitleHeadingChunkText()
    if version == CONTEXTUAL_CHUNK_VERSION:
        return MetadataContextChunkText()
    raise ValueError(f"Unknown chunk representation version: {version}")
