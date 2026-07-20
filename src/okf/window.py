"""Windowed extraction of raw source article text.

OKF navigation answers strictly from verbatim windows of the underlying
articles. Concept summaries and extracted facts are never used as answer
evidence, so this module returns raw substrings only.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
BLANK_LINE_RE = re.compile(r"\n\s*\n")


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def _blocks(markdown: str) -> list[tuple[int, str]]:
    """Split markdown into (start_offset, block) pairs on blank lines."""
    blocks: list[tuple[int, str]] = []
    offset = 0
    for piece in BLANK_LINE_RE.split(markdown):
        start = markdown.find(piece, offset)
        if start == -1:
            start = offset
        stripped = piece.strip()
        if stripped:
            blocks.append((start, stripped))
        offset = start + len(piece)
    return blocks


def extract_windows(
    markdown: str,
    query: str,
    *,
    window_chars: int,
    max_windows: int,
) -> list[str]:
    """Return up to ``max_windows`` verbatim windows most relevant to ``query``.

    Each window is a raw contiguous slice of ``markdown`` of at most
    ``window_chars`` characters, anchored at a block whose tokens overlap the
    query. When nothing overlaps (for example across languages) the document
    head is returned so the caller still sees raw text rather than a summary.
    """
    text = markdown.strip()
    if not text or window_chars <= 0 or max_windows <= 0:
        return []

    query_terms = set(_tokens(query))
    scored: list[tuple[int, int]] = []
    for start, block in _blocks(text):
        if not query_terms:
            break
        overlap = sum(1 for token in _tokens(block) if token in query_terms)
        if overlap:
            scored.append((overlap, start))

    if not scored:
        return [text[:window_chars]]

    scored.sort(key=lambda item: (-item[0], item[1]))
    windows: list[str] = []
    used_spans: list[tuple[int, int]] = []
    for _score, start in scored:
        end = min(start + window_chars, len(text))
        if any(start < span_end and end > span_start for span_start, span_end in used_spans):
            continue
        windows.append(text[start:end])
        used_spans.append((start, end))
        if len(windows) >= max_windows:
            break
    return windows
