"""Read-only page navigation tools for the agentic retriever.

A retrieved chunk always belongs to a source page, so the agent can move from a
chunk to that page's structure (`list_sections`) and to sibling chunks
(`search_in_page`). The geo tools (`find_pages_near`, `pages_in_region`) move
from a place to the pages located there. All of them return real
`page_chunks.id` values, which keeps whatever the agent finds scoreable against
the existing chunk-level qrels.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.db.pages import PageLocation
from src.db.pages import connect_pages as connect
from src.db.pages import pages_in_region, pages_near, region_code_column
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.shared.geocode import Coordinates

__all__ = [
    "LocatedPage",
    "PageChunkHit",
    "PageRef",
    "PageSection",
    "list_sections",
    "located_pages_in_region",
    "located_pages_near",
    "page_refs_for_chunks",
    "search_in_page",
]


@dataclass(frozen=True)
class PageRef:
    """Source page behind a retrieved chunk."""

    page_id: str
    title: str
    url: str
    source: str
    chunk_count: int


@dataclass(frozen=True)
class PageSection:
    heading_path: str
    chunk_count: int


@dataclass(frozen=True)
class PageChunkHit:
    id: str
    heading_path: str
    text: str


@dataclass(frozen=True)
class LocatedPage:
    """A page with a primary location, and its first chunk as the scoreable id."""

    page_id: str
    title: str
    place: str
    nuts3: str | None
    distance_km: float | None
    chunk_id: str
    chunk_text: str


def region_code_supported(code: str) -> bool:
    """Whether `pages_in_region` can answer for this code shape."""
    return region_code_column(code) is not None


def located_pages_near(
    coordinates: Coordinates,
    radius_km: float,
    limit: int = 10,
    variant: str = BASE_CHUNK_VARIANT,
) -> list[LocatedPage]:
    """Pages whose primary location lies within `radius_km`, nearest first."""
    hits = pages_near(coordinates, radius_km, limit=limit)
    return _located_pages(hits, variant)


def located_pages_in_region(
    code: str, limit: int = 20, variant: str = BASE_CHUNK_VARIANT
) -> list[LocatedPage]:
    """Pages whose primary location sits in a NUTS or ISO country code."""
    locations = pages_in_region(code, limit=limit)
    return _located_pages([(location, None) for location in locations], variant)


def _located_pages(
    hits: list[tuple[PageLocation, float | None]], variant: str
) -> list[LocatedPage]:
    heads = _first_chunks([location.page_id for location, _ in hits], variant)
    return [
        LocatedPage(
            page_id=location.page_id,
            title=heads[location.page_id]["title"],
            place=location.name or "",
            nuts3=location.nuts3,
            distance_km=None if distance is None else round(distance, 1),
            chunk_id=heads[location.page_id]["id"],
            chunk_text=heads[location.page_id]["text"],
        )
        for location, distance in hits
        if location.page_id in heads
    ]


def _first_chunks(page_ids: list[str], variant: str) -> dict[str, dict]:
    """Each page's chunk 0 of `variant` with its title: the id the agent can promote."""
    if not page_ids:
        return {}
    placeholders = ",".join("?" for _ in page_ids)
    with connect() as conn:
        rows = conn.execute(
            f"""
            select c.page_id, c.id, c.text, coalesce(m.title, '') as title
            from page_chunks c
            join page_metadata m on m.id = c.page_id
            where c.page_id in ({placeholders})
              and c.variant = ? and c.chunk_index = 0
            """,
            [*page_ids, variant],
        ).fetchall()
    return {row["page_id"]: dict(row) for row in rows}


def page_refs_for_chunks(chunk_ids: list[str]) -> dict[str, PageRef]:
    """Map each chunk id to its source page so the agent can name a page."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    with connect() as conn:
        rows = conn.execute(
            f"""
            select
              c.id as chunk_id,
              c.page_id as page_id,
              coalesce(m.title, '') as title,
              m.url as url,
              m.source as source,
              (
                select count(*) from page_chunks sibling
                where sibling.page_id = c.page_id
              ) as chunk_count
            from page_chunks c
            join page_metadata m on m.id = c.page_id
            where c.id in ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
    return {
        row["chunk_id"]: PageRef(
            page_id=row["page_id"],
            title=row["title"],
            url=row["url"],
            source=row["source"],
            chunk_count=row["chunk_count"],
        )
        for row in rows
    }


def list_sections(page_id: str, limit: int = 40) -> list[PageSection]:
    """Heading paths of one page, as a table of contents."""
    with connect() as conn:
        rows = conn.execute(
            """
            select
              coalesce(nullif(heading_path, ''), '(no heading)') as heading_path,
              count(*) as chunk_count,
              min(chunk_index) as first_index
            from page_chunks
            where page_id = ?
            group by heading_path
            order by first_index
            limit ?
            """,
            (page_id, limit),
        ).fetchall()
    return [
        PageSection(heading_path=row["heading_path"], chunk_count=row["chunk_count"])
        for row in rows
    ]


def search_in_page(page_id: str, term: str, limit: int = 5) -> list[PageChunkHit]:
    """Chunks of one page matching a term, ordered by document position.

    Matching is a case-insensitive substring over chunk text and heading path,
    so the agent gets back real chunk ids that the benchmark can score.
    """
    term = (term or "").strip()
    if not term:
        return []
    pattern = f"%{_escape_like(term)}%"
    with connect() as conn:
        rows = conn.execute(
            """
            select id, coalesce(heading_path, '') as heading_path, text
            from page_chunks
            where page_id = ?
              and (
                text like ? escape '\\'
                or coalesce(heading_path, '') like ? escape '\\'
              )
            order by chunk_index
            limit ?
            """,
            (page_id, pattern, pattern, limit),
        ).fetchall()
    return [
        PageChunkHit(
            id=row["id"],
            heading_path=row["heading_path"],
            text=row["text"],
        )
        for row in rows
    ]


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
