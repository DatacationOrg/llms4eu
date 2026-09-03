from __future__ import annotations

import trafilatura


EXTRACTOR_NAME = "trafilatura"


def extract_markdown(html: str, url: str | None = None) -> str:
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_formatting=True,
        include_links=False,
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
