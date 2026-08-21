"""The committed dev-branch extractor, frozen so it can be voted against.

Why this exists. The `ours` entrant calls `src.scraping.page_extract`, which
imports `extract_markdown` from the working tree -- so every vote in the arena
describes whatever the working tree held at run time, and the working tree holds
five uncommitted fixes. That left the actual question unanswerable: is the
version on the branch (images/links/dedup/favor_precision on, plus a BeautifulSoup
pre-clean) better or worse than what replaced it?

This module reproduces `extract_markdown` exactly as it stands at the last commit
that touched it (2c1d8ef), and runs it through the *same* unchanged ladder in
page_extract, so the only difference between `ours` and `ours@committed` is the
five settings. It is a frozen copy on purpose: it must not follow further edits
to the working tree, or the comparison silently changes meaning.
"""

from __future__ import annotations

from contextlib import contextmanager

import trafilatura

from src.scraping.extract_markdown import _looks_like_web_chrome, clean_markdown


def _remove_common_junk(html: str) -> str:
    """The pre-clean, as committed. Kept verbatim, comments included below.

    Its measured cost was 104 zero-padded table cells on the German-language
    Wikipedia pages (a BeautifulSoup round-trip re-serialises the hidden sort-key
    spans so trafilatura no longer drops them) against a benefit of 14 fewer
    cookie/consent words.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "form", "button"]):
        tag.decompose()

    junk_terms = ("cookie", "cookies", "consent", "gdpr", "accessibility")
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        values: list[str] = []
        for attr in ("id", "class", "role", "aria-label"):
            value = tag.get(attr)
            if isinstance(value, list):
                values.extend(str(item) for item in value)
            elif value:
                values.append(str(value))
        haystack = " ".join(values).lower()
        if any(term in haystack for term in junk_terms):
            tag.decompose()

    return str(soup)


def extract_markdown_as_committed(html: str, url: str | None = None) -> str:
    cleaned_html = _remove_common_junk(html)
    markdown = trafilatura.extract(
        cleaned_html,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_images=True,
        include_links=True,
        include_formatting=True,
        deduplicate=True,
        favor_precision=True,
    )
    cleaned = clean_markdown(markdown or "")
    if cleaned and not _looks_like_web_chrome(cleaned):
        return cleaned

    fallback = trafilatura.baseline(cleaned_html)
    fallback_text = fallback[1] if fallback else ""
    fallback_markdown = clean_markdown(fallback_text)
    if fallback_markdown and not _looks_like_web_chrome(fallback_markdown):
        return fallback_markdown

    return cleaned


@contextmanager
def committed_settings():
    """Swap the ladder's extractor for the committed one, then put it back.

    page_extract binds `extract_markdown` at import time, so the name has to be
    replaced on that module rather than on extract_markdown itself.
    """
    from src.scraping import page_extract

    original = page_extract.extract_markdown
    page_extract.extract_markdown = extract_markdown_as_committed
    try:
        yield
    finally:
        page_extract.extract_markdown = original
