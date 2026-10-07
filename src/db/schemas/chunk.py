from __future__ import annotations

from src.db.schemas.base import Row


class Chunk(Row):
    """A piece of a page, cut at its headings. Every page is cut at every size in
    `dataset.SIZES`; filter by `size`."""

    file = "chunks.parquet"
    id: str
    """Chunk id, `<page id>:<size>:<n>`."""
    page_id: str
    """The page it is cut from (`Page.id`)."""
    size: int
    """Target size in tokens, one of `dataset.SIZES`."""
    n: int
    """Position in the page at this size, from 0."""
    title: str
    """The page title."""
    breadcrumb: str | None = None
    """Headings all of the chunk sits under, e.g. `Geschichte > Neuzeit`; null when
    the chunk spans sections with no common heading."""
    text: str
    """The chunk's Markdown; headings below the breadcrumb stay inline."""
    tokens: int
    """Tokens in `text`: at most `size`, plus up to `MIN_TOKENS` for a merged tail."""
    role: str | None = None
    """Ling 3.1 Flash, one sentence in the page's language on what kind of information
    the chunk adds to its page; null until generated."""


def represent(title: str, breadcrumb: str | None, text: str) -> str:
    """What gets embedded for a chunk: title, breadcrumb, text (as aihub-core does)."""
    return "\n".join(part for part in (title, breadcrumb, text) if part)
