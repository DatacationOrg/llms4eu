from __future__ import annotations

from src.db.schemas.base import Row


class Chunk(Row):
    """A piece of a page, cut at its headings, at one of four target sizes. Filter by `size`."""

    file = "chunks.parquet"
    id: str  # <page id>:<size>:<n>
    page_id: str  # Page.id
    size: int  # target tokens, one of dataset.SIZES
    n: int  # position in the page, from 0
    title: str  # the page title
    # headings above the chunk, e.g. "Geschichte > Neuzeit"
    breadcrumb: str | None = None
    text: str
    tokens: int
    # Ling 3.1 Flash: one sentence on what the chunk adds to the page
    role: str | None = None


def represent(title: str, breadcrumb: str | None, text: str) -> str:
    """What gets embedded for a chunk: title, breadcrumb, text (as aihub-core does)."""
    return "\n".join(part for part in (title, breadcrumb, text) if part)
