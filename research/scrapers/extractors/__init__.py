"""Entrant registry for the arena.

An entrant is a (input variant, extraction function) pair. Keeping the input
variant explicit is what makes the raw-vs-rendered axis votable instead of a
hidden constant.

Output format note, restated in the report's methods: `ours`, `trafilatura*`,
`readability-lxml` and `scrapling-md` emit Markdown (the last two via markdownify,
configured to match trafilatura's ATX-heading, no-escape style); `resiliparse`,
`justext`, `goose3` and `html-text` emit plain text, because that is all they
produce. Stock trafilatura is deliberately set to Markdown so the `ours` vs
`trafilatura` matchup isolates our wrapper rather than a formatting difference.
The arena shows raw strings in a `<pre>` by default, which further flattens any
presentation advantage.

`resiliparse` runs with `alt_texts=False`, matching trafilatura's default of
omitting image alt text. That costs resiliparse ~2.8% of its output; it is a
cross-entrant parity choice, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# trafilatura emits ATX headings and no escaping. markdownify defaults to setext
# underlines (`Title\n=====`) and backslash-escapes `_`/`*`, which showed up as 415
# setext underlines and 384 stray escapes across the corpus. Humans vote on these
# strings, so without aligning this the arena would partly be scoring Markdown
# writers rather than extractors.
MARKDOWNIFY_OPTIONS = {
    "heading_style": "ATX",
    "escape_asterisks": False,
    "escape_underscores": False,
    "escape_misc": False,
}


def reset_extractor_state() -> None:
    """Clear cross-call global state before every extraction.

    This exists because of a real and severe bug. Our pipeline passes
    `deduplicate=True` to trafilatura, which drives a *process-global* LRU cache
    (`trafilatura.deduplication.LRU_TEST`): once a text segment has been seen
    MAX_REPETITIONS times it is dropped as a duplicate. Since the timing protocol
    calls each entrant several times per page, the 4th call onward returned
    progressively gutted text -- and the runner was storing the last call. On
    `en.wikisource.org/wiki/The_Federalist_Papers` that meant 17,028 chars
    collapsing to 618, a 96% loss. It also leaked *between* entrants, because
    `ours` and `ours@raw-only` run back-to-back over the same HTML.

    Benchmarking a page in isolation requires a cold cache, which is also how
    every other entrant behaves (none carries cross-call state). Note this makes
    `ours` behave as it would on the first page of a crawl; in production the warm
    cache additionally suppresses boilerplate repeated across pages, which this
    per-page comparison deliberately does not model.
    """
    try:
        from trafilatura.deduplication import LRU_TEST

        LRU_TEST.clear()
    except Exception:
        pass


LANGUAGE_STOPLISTS = {
    "en": "English",
    "nl": "Dutch",
    "de": "German",
    "sl": "Slovenian",
    "hu": "Hungarian",
    "es": "Spanish",
    "pl": "Polish",
}


@dataclass(frozen=True)
class Outcome:
    """What an entrant produced, plus which snapshot variant it actually used."""

    text: str
    used_variant: str


@dataclass(frozen=True)
class Entrant:
    name: str
    variants: tuple[str, ...]  # snapshot variants it may consume
    label: str  # human description, shown in the report
    layer: str  # extractor | floor | wiki
    run: Callable[[dict[str, str | None], dict], Outcome]
    wiki_only: bool = False

    def applies_to(self, page: dict) -> bool:
        return bool(page["is_wiki"]) if self.wiki_only else True


# --------------------------------------------------------------------------
# Extraction functions. Each takes {variant: text} plus the page row.
# --------------------------------------------------------------------------


def _run_ladder(
    inputs: dict[str, str | None], page: dict, variants: tuple[str, ...]
) -> Outcome:
    """Our real ladder, driven from cached HTML over the given input variants.

    `extract_page()` would call Playwright live on fallback; here the ladder's own
    decision logic runs against the cached snapshots instead. Same branches, no
    network. Which variants it may fall through to is the parameter, because that
    fallback is itself one of the things under test.
    """
    from datetime import datetime, timezone

    from src.scraping.page_extract import _extract_from_html
    from src.shared.schema import PageMetadata

    metadata = PageMetadata(
        id=page["id"],
        source=page["bucket"],
        url=page["url"],
        final_url=page["url"],
        fetched_at=datetime.now(timezone.utc),
        title=page.get("title"),
    )
    for variant in variants:
        html = inputs.get(variant)
        if not html:
            continue
        attempt = _extract_from_html(metadata, html, 30.0)
        if attempt.result and attempt.result.markdown_content:
            return Outcome(attempt.result.markdown_content.markdown, variant)
    return Outcome("", variants[0])


def _ours(inputs: dict[str, str | None], page: dict) -> Outcome:
    """Our ladder as it ships: raw first, rendered as the fallback."""
    return _run_ladder(inputs, page, ("raw", "rendered"))


def _ours_committed(inputs: dict[str, str | None], page: dict) -> Outcome:
    """`ours`, but with the extractor as it stands at the last commit.

    The five settings that the working tree changed (images, links, dedup,
    favor_precision, and the BeautifulSoup pre-clean) are the *only* difference:
    the ladder in page_extract is byte-identical between the two, so a vote here
    is a vote on the settings alone. See extractors/legacy_ours.py.
    """
    from research.scrapers.extractors.legacy_ours import committed_settings

    with committed_settings():
        return _ours(inputs, page)


def _ours_raw_only(inputs: dict[str, str | None], page: dict) -> Outcome:
    """Our ladder with the render fallback disabled -- isolates that fallback."""
    return _run_ladder(inputs, page, ("raw",))


def _trafilatura(variant: str):
    def run(inputs: dict[str, str | None], page: dict) -> Outcome:
        import trafilatura

        html = inputs.get(variant)
        if not html:
            return Outcome("", variant)
        text = trafilatura.extract(html, url=page["url"], output_format="markdown")
        return Outcome(text or "", variant)

    return run


def _resiliparse(variant: str):
    def run(inputs: dict[str, str | None], page: dict) -> Outcome:
        from resiliparse.extract.html2text import extract_plain_text

        html = inputs.get(variant)
        if not html:
            return Outcome("", variant)
        text = extract_plain_text(
            html,
            main_content=True,
            preserve_formatting=True,
            alt_texts=False,
            links=False,
            form_fields=False,
            noscript=False,
        )
        return Outcome(text or "", variant)

    return run


def _justext(inputs: dict[str, str | None], page: dict) -> Outcome:
    """Per-language stoplist -- the reason jusText is the multilingual probe."""
    import justext

    html = inputs.get("raw")
    if not html:
        return Outcome("", "raw")
    stoplist_name = LANGUAGE_STOPLISTS.get(page["lang"], "English")
    try:
        stoplist = justext.get_stoplist(stoplist_name)
    except Exception:
        stoplist = justext.get_stoplist("English")
    # `encoding=` is load-bearing defensively: jusText re-sniffs the encoding from
    # the document's own <meta charset> even when handed already-decoded UTF-8
    # bytes, so a page declaring iso-8859-15 gets mojibake and then fails stoplist
    # matching. No page in the current corpus declares a non-UTF-8 charset, and
    # output is byte-identical on all 100 with and without this, but a future
    # corpus addition would silently corrupt without it.
    paragraphs = justext.justext(
        html.encode("utf-8", errors="replace"), stoplist, encoding="utf-8"
    )
    kept = [p.text for p in paragraphs if not p.is_boilerplate]
    return Outcome("\n\n".join(kept), "raw")


def _readability(inputs: dict[str, str | None], page: dict) -> Outcome:
    """readability's summary is HTML; markdownify is the standard real pairing.

    `url=` is passed so relative hrefs resolve (worth ~6% of output, and matches
    what trafilatura and scrapling already receive). `html_partial=True` is
    load-bearing rather than cosmetic: it is identical on 97/100 pages but
    returns content where the default returns nothing on three
    (kormany.hu, rijksoverheid.nl, smithsonianmag.com).
    """
    from readability import Document
    from markdownify import markdownify

    html = inputs.get("raw")
    if not html:
        return Outcome("", "raw")
    document = Document(html, url=page["url"])
    summary_html = document.summary(html_partial=True)
    return Outcome(markdownify(summary_html, **MARKDOWNIFY_OPTIONS).strip(), "raw")


def _goose3(inputs: dict[str, str | None], page: dict) -> Outcome:
    """goose3, with a retry that rescues HTML5 `<section>` layouts.

    goose3's `post_cleanup()` keeps only a hardcoded tag whitelist (`p`, plus
    lists/headers) and deletes every other child of the top node. Modern Wikipedia
    is served as Parsoid HTML which wraps each heading's content in `<section>`, so
    goose3 located the right container and then deleted all of it -- Ada Lovelace
    collapsed from 608k chars of top node to 196 chars of whitespace, i.e. empty.
    That is an artifact of a tag whitelist, not a judgement about the content.

    `known_context_patterns` is the library's own public knob for this. It is
    applied only as a retry, because using it unconditionally regresses 16 pages
    that already work (MDN 11,715 -> 2,505); retry-on-empty has no regressions by
    construction. Empties fall 36 -> 17 corpus-wide.

    The remaining 17 empties are genuine goose3 behaviour: it is news-tuned and
    high-precision/low-recall, and even when rescued it recovers one section
    rather than the article (Ada Lovelace 11,885 vs trafilatura's 67,906).
    """
    from goose3 import Goose
    from goose3.configuration import ArticleContextPattern, Configuration

    html = inputs.get("raw")
    if not html:
        return Outcome("", "raw")

    def attempt(configuration: Configuration | None = None) -> str:
        goose = Goose(configuration) if configuration else Goose()
        try:
            article = goose.extract(raw_html=html, url=page["url"])
            return article.cleaned_text or ""
        finally:
            goose.close()

    text = attempt()
    if not text.strip():
        configuration = Configuration()
        configuration.known_context_patterns = ArticleContextPattern(tag="section")
        text = attempt(configuration)
    return Outcome(text, "raw")


def _html_text(inputs: dict[str, str | None], page: dict) -> Outcome:
    """Floor: whole visible page, no boilerplate removal whatsoever."""
    import html_text

    html = inputs.get("raw")
    if not html:
        return Outcome("", "raw")
    return Outcome(html_text.extract_text(html, guess_layout=True) or "", "raw")


def _scrapling_md(inputs: dict[str, str | None], page: dict) -> Outcome:
    """Floor: Scrapling's own `extract --md` output. No content model at all.

    Calls Scrapling's own `Convertor._extract_content` rather than reimplementing
    it. The previous hand-rolled version replicated `_strip_noise_tags` but skipped
    `_sanitize_for_ai`, so it kept CSS-hidden and `<template>` content that the
    real `scrapling extract --md` drops -- up to 35% extra on some pages. That
    misattributed bloat to a tool that would not have produced it.
    """
    from scrapling import Selector
    from scrapling.core.shell import Convertor

    html = inputs.get("raw")
    if not html:
        return Outcome("", "raw")
    selector = Selector(html, url=page["url"])
    return Outcome(
        "".join(
            Convertor._extract_content(selector, "markdown", main_content_only=True)
        ).strip(),
        "raw",
    )


def _wikiextractor(inputs: dict[str, str | None], page: dict) -> Outcome:
    """Wikiextractor-V2 over wikitext wrapped as a minimal MediaWiki dump."""
    from research.scrapers.extractors.wikiextractor_adapter import extract_wikitext

    wikitext = inputs.get("wikitext")
    if not wikitext:
        return Outcome("", "wikitext")
    return Outcome(extract_wikitext(wikitext, page), "wikitext")


def _out_of_process(tool: str):
    """Placeholder for the three challengers, which cannot run in this interpreter.

    `crawl4ai` pulls 97 packages and `docling` a document-AI stack, so neither can join
    `arena-env` without breaking resolution for the other nine entrants. Their output is
    produced by `challenger_worker.py` under `uv run --no-project --with <tool>` and
    written straight to the runs table by `run_challengers.py`. Raising here rather than
    silently returning empty output means a `run_extractors` pass that forgets the
    separate driver fails loudly instead of recording nine empty cells.
    """

    def run(inputs: dict[str, str | None], page: dict) -> Outcome:
        raise RuntimeError(
            f"{tool} runs out of process; use "
            f"`uv run python -m research.scrapers.run_challengers --tools {tool}`"
        )

    return run


REGISTRY: tuple[Entrant, ...] = (
    Entrant(
        "ours",
        ("raw", "rendered"),
        "our pipeline as the working tree holds it: trafilatura(markdown, tables, "
        "formatting, no links) + listing/document/render ladder",
        "extractor",
        _ours,
    ),
    Entrant(
        "ours@committed",
        ("raw", "rendered"),
        "our pipeline as last committed: bs4 pre-clean + "
        "trafilatura(images, links, dedup, favor_precision) + the same ladder",
        "extractor",
        _ours_committed,
    ),
    Entrant(
        "ours@raw-only",
        ("raw",),
        "our pipeline with the Playwright render fallback disabled",
        "extractor",
        _ours_raw_only,
    ),
    Entrant(
        "trafilatura",
        ("raw",),
        "stock trafilatura 2.0.0, default settings, Markdown output",
        "extractor",
        _trafilatura("raw"),
    ),
    Entrant(
        "trafilatura@rendered",
        ("rendered",),
        "stock trafilatura over the Playwright-rendered DOM",
        "extractor",
        _trafilatura("rendered"),
    ),
    Entrant(
        "resiliparse",
        ("raw",),
        "resiliparse extract_plain_text(main_content=True, preserve_formatting=True)",
        "extractor",
        _resiliparse("raw"),
    ),
    Entrant(
        "resiliparse@rendered",
        ("rendered",),
        "resiliparse main_content over the Playwright-rendered DOM",
        "extractor",
        _resiliparse("rendered"),
    ),
    Entrant(
        "justext",
        ("raw",),
        "jusText 3.0.2 with the page's own language stoplist",
        "extractor",
        _justext,
    ),
    Entrant(
        "readability-lxml",
        ("raw",),
        "readability-lxml summary HTML converted with markdownify",
        "extractor",
        _readability,
    ),
    Entrant(
        "goose3",
        ("raw",),
        "goose3 cleaned_text (high precision, low recall)",
        "extractor",
        _goose3,
    ),
    Entrant(
        "html-text",
        ("raw",),
        "FLOOR: html-text whole-page visible text, no boilerplate removal",
        "floor",
        _html_text,
    ),
    Entrant(
        "scrapling-md",
        ("raw",),
        "FLOOR: Scrapling's own markdown export (markdownify over <body>)",
        "floor",
        _scrapling_md,
    ),
    Entrant(
        "wikiextractor-v2",
        ("wikitext",),
        "Wikiextractor-V2 over MediaWiki wikitext (wiki pages only, AGPL-3.0)",
        "wiki",
        _wikiextractor,
        wiki_only=True,
    ),
    Entrant(
        "markitdown",
        ("raw",),
        "CHALLENGER: Microsoft MarkItDown HTML converter",
        "challenger",
        _out_of_process("markitdown"),
    ),
    Entrant(
        "crawl4ai",
        ("raw",),
        "CHALLENGER: crawl4ai markdown generator + PruningContentFilter (defaults)",
        "challenger",
        _out_of_process("crawl4ai"),
    ),
    Entrant(
        "docling",
        ("raw",),
        "CHALLENGER: IBM Docling HTML backend, export_to_markdown",
        "challenger",
        _out_of_process("docling"),
    ),
)

CHALLENGERS = ("markitdown", "crawl4ai", "docling")

BY_NAME = {entrant.name: entrant for entrant in REGISTRY}


def enabled(names: list[str] | None = None) -> list[Entrant]:
    if not names:
        return list(REGISTRY)
    return [BY_NAME[name] for name in names]
