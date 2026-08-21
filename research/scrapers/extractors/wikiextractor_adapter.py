"""Wikiextractor-V2 adapter.

Wikiextractor consumes MediaWiki wikitext, not HTML, so it cannot be scored on
the same task as the HTML extractors. It is included as a wiki-slice-only
reference, and its rating is reported only within the wiki bucket.

Two caveats that belong in any write-up of the results:

1. **No template expansion.** Template expansion is Wikiextractor's real
   differentiator, but template *definitions* live in the dump header. Driving
   it from a single page's wikitext (MediaWiki `action=raw`) means templates
   cannot be expanded, so what is measured here is its wikitext cleaning only.
   A faithful comparison would need the Wikimedia Enterprise HTML/XML dumps.
2. **AGPL-3.0**, unlike the Apache/MIT/BSD licensing of every other entrant.

The upstream checkout lives in `.local/scraper-arena/vendor/` and is neither
vendored into this repo nor pip-installable.
"""

from __future__ import annotations

import sys
from functools import cache

from research.scrapers import store


VENDOR_SUBPATH = "vendor/Wikiextractor-V2"


@cache
def _extractor_class():
    """Import Wikiextractor-V2 from the local checkout and configure it."""
    root = store.data_dir() / VENDOR_SUBPATH
    if not root.is_dir():
        raise RuntimeError(
            f"Wikiextractor-V2 checkout missing at {root}. Clone it with: "
            "git clone --depth 1 "
            f"https://github.com/langtech-bsc/Wikiextractor-V2.git {root}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from wikiextractor.extract.extract import Extractor

    Extractor.generator = True  # return the doc instead of writing a file
    Extractor.to_txt = True
    Extractor.to_json = False
    Extractor.markdown = True  # section headers, comparable to the others
    Extractor.keepSections = True
    Extractor.keepLinks = False
    return Extractor


def _clean_title(page: dict) -> str:
    title = (page.get("title") or page["url"].rsplit("/", 1)[-1]).replace("_", " ")
    for suffix in (
        " - Wikipedia",
        " – Wikipedia",
        " — Wikipedia",
        " - Wikipédia",
        " – Wikipédia",
        " - Wikipedia, la enciclopedia libre",
    ):
        if title.endswith(suffix):
            title = title[: -len(suffix)]
            break
    return title


def extract_wikitext(wikitext: str, page: dict) -> str:
    Extractor = _extractor_class()
    Extractor.language = page.get("lang", "")

    extractor = Extractor(
        id=page["id"],
        revid="0",
        urlbase=page["url"].split("/wiki/")[0] + "/wiki/",
        title=_clean_title(page),
        page=[wikitext],
    )
    result = extractor.extract(None)
    if not result:
        return ""
    # generator + to_txt yields (id, title, url, language, text)
    return result[4] if isinstance(result, tuple) else str(result)
