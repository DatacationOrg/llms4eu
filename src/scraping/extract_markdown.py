from __future__ import annotations

from bs4 import BeautifulSoup
import trafilatura


EXTRACTOR_NAME = "trafilatura"


def extract_markdown(html: str, url: str | None = None) -> str:
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


def _remove_common_junk(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "form", "button"]):
        tag.decompose()

    junk_terms = (
        "cookie",
        "cookies",
        "consent",
        "gdpr",
        "accessibility",
    )
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
