from __future__ import annotations

from src.db.schemas.base import Category, Country, Lang, Row


class Page(Row):
    """A Wikipedia page about an EU place: what the system searches over."""

    file = "wikipages.parquet"
    id: str
    """Page id, `<wiki>/<Wikidata id>` (e.g. `dewiki/Q177125`); questions point here."""
    wikiname: str
    """The Wikipedia it is from, e.g. `dewiki`."""
    wikidata_id: str
    """The place's Wikidata item, e.g. `Q177125`."""
    title: str
    """Article title."""
    url: str
    """Article URL as requested."""
    final_url: str
    """Article URL after redirects."""
    in_language: Lang
    """Language of the Wikipedia it was taken from: the country's own language."""
    language: Lang
    """Language the fetched page declares (`<html lang>`)."""
    language_ok: bool
    """Whether `language` equals `in_language` and the page came from that wiki."""
    country: Country
    """Country the place is in."""
    country_languages: list[Lang]
    """The country's official languages; the article is in the first that has one."""
    latitude: float
    """Latitude of the place, from Wikidata."""
    longitude: float
    """Longitude of the place, from Wikidata (overseas territories included)."""
    sitelinks: int
    """Number of Wikipedias with a page on the place, a popularity hint (1 to 136)."""
    categories: list[Category]
    """Kinds of place it is listed as, one or more."""
    machine_generated: bool
    """Whether it is a bot-written stub (mostly Swedish lake register pages)."""
    char_count: int
    """Characters in `text`."""
    word_count: int
    """Whitespace-separated words in `text`."""
    fetched_at: str
    """When it was fetched, ISO 8601 with time zone."""
    text: str
    """The page as Markdown, starting with `# <title>`; evidence offsets index into it."""
    summary: str | None = None
    """Ling 3.1 Flash summary in the page's language, at most 3 sentences and 300
    characters; null until generated."""
