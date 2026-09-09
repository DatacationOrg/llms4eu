from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# One representation. v2 (page metadata lines) and v3 (v2 plus a location line)
# existed in 2026-07 and 2026-09 but were never measured in a published report
# and were removed on 2026-09-08; the version mechanism stays so a measured
# alternative can be added as its own collection later.
LEGACY_CHUNK_VERSION = "v1"
CHUNK_VERSIONS = (LEGACY_CHUNK_VERSION,)


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
    # The page's primary location (sql/eval.sql `page_locations`), when it has one.
    location_name: str | None = None
    country_code: str | None = None
    nuts2: str | None = None
    nuts3: str | None = None
    nuts3_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    @classmethod
    def from_row(cls, row) -> PageChunk:
        """Build from a `src.db.pages.CHUNK_ROWS_SQL` row (or any mapping).

        Geo columns are optional so a fixture without `page_locations` still
        loads; they come back None, which is exactly what an unlocated page is.
        """
        keys = set(row.keys())

        def get(key: str):
            return row[key] if key in keys else None

        return cls(
            id=row["id"],
            page_id=row["page_id"],
            heading_path=row["heading_path"],
            text=row["text"],
            title=get("title"),
            chunk_index=get("chunk_index") or 0,
            source=get("source"),
            language=get("language"),
            page_kind=get("page_kind"),
            location_name=get("location_name"),
            country_code=get("country_code"),
            nuts2=get("nuts2"),
            nuts3=get("nuts3"),
            nuts3_name=get("nuts3_name"),
            latitude=get("latitude"),
            longitude=get("longitude"),
        )


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


def chunk_text_representation(version: str) -> ChunkTextRepresentation:
    if version == LEGACY_CHUNK_VERSION:
        return TitleHeadingChunkText()
    raise ValueError(f"Unknown chunk representation version: {version}")
