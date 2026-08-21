from __future__ import annotations

import trafilatura


EXTRACTOR_NAME = "trafilatura"


def extract_markdown(html: str, url: str | None = None) -> str:
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        # Only switches that add *markdown structure* are set here. Anything that
        # changes which content trafilatura selects is left at its default: its own
        # four-stage cascade does that better than our overrides did. The measurements
        # behind each choice are in research/scrapers/RESEARCH_LOG.md, "The flag
        # audit (session 2)".
        #
        # Load-bearing rather than redundant: this is what captures the Wikipedia
        # infobox, so it is passed explicitly even though 2.0.0 defaults it on.
        include_tables=True,
        # Headings, bold and lists. Pure serialisation, no effect on selection.
        include_formatting=True,
        # Off: 2.0.0 splits a sentence before an inline link and fuses the word after
        # it (6,230 fused links across the 100-page corpus, zero with it off).
        include_links=False,
        # Not set, and deliberately: include_images lets an image replace a table
        # cell's text, favor_precision disables two of the cascade's own rescues, and
        # deduplicate drives a process-global LRU that guts repeat extractions.
    )
    cleaned = clean_markdown(markdown or "")
    if cleaned and not _looks_like_web_chrome(cleaned):
        return cleaned

    fallback = trafilatura.baseline(html)
    fallback_text = fallback[1] if fallback else ""
    fallback_markdown = clean_markdown(fallback_text)
    if fallback_markdown and not _looks_like_web_chrome(fallback_markdown):
        return fallback_markdown

    return cleaned


def markdown_needs_browser_render(markdown: str) -> bool:
    text = markdown.strip()
    if not text:
        return True

    lowered = _normalized_lower(text)
    junk_markers = (
        "enable javascript",
        "please enable javascript",
        "just a moment",
        "access denied",
        "captcha",
        "accessibility provided by",
        "](#)",
    )
    return any(marker in lowered for marker in junk_markers)


def markdown_looks_like_contact_footer(markdown: str) -> bool:
    return _looks_like_contact_footer(markdown)


# The BeautifulSoup pre-clean that used to run here is gone. It stripped
# script/style/svg/form/button (which trafilatura already does) and any element whose
# id/class/role/aria-label matched cookie/consent/gdpr/accessibility -- which cost 104
# zero-padded table cells and deleted real `## Accessibility` sections, for no change
# in median output. Evidence in research/scrapers/RESEARCH_LOG.md, "`_remove_common_junk`
# removed".


def clean_markdown(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            lines.append("")
            continue
        if "cookie" in lowered:
            continue
        lines.append(stripped)

    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = blank

    return "\n".join(collapsed).strip()


def _looks_like_web_chrome(markdown: str) -> bool:
    lowered = _normalized_lower(markdown)
    return "accessibility provided by" in lowered or "](#)" in markdown


def _looks_like_contact_footer(markdown: str) -> bool:
    if len(markdown) > 400:
        return False
    lowered = _normalized_lower(markdown)
    return "@" in lowered and ("+" in lowered or "tel" in lowered)


def _normalized_lower(text: str) -> str:
    return " ".join(text.lower().split())
