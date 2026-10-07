from __future__ import annotations

from src.db.schemas.base import Row


class Page(Row):
    """A Wikipedia page about an EU place: what the system searches over."""

    file = "wikipages.parquet"
    id: str  # <lang>wiki/<Wikidata id>, the page id every question file uses
    wikiname: str  # e.g. dewiki
    wikidata_id: str
    title: str
    url: str
    final_url: str  # after redirects
    in_language: str  # the wiki's language
    language: str  # the language the page declares
    language_ok: bool  # both agree
    country: str  # ISO 3166 alpha-2
    country_languages: list[str]
    latitude: float | None = None
    longitude: float | None = None
    sitelinks: int  # Wikipedias with a page on it: a popularity hint
    categories: list[str]  # castle, lake, cave, ...
    machine_generated: bool  # bot-written stub (mostly Swedish lakes)
    char_count: int
    word_count: int
    fetched_at: str
    text: str  # the page as Markdown, starting with `# <title>`
    summary: str | None = None  # Ling 3.1 Flash, at most 3 sentences / 300 chars
