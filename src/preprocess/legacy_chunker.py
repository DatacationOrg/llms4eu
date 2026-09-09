"""The project's original heading-aware chunker, frozen.

The approved eval labels point at the chunk boundaries this module produces, so
its behaviour must not change. New chunking work goes through
`src.preprocess.chunkers`, which uses `langchain-text-splitters`.

Kept only because reproducing the `base` variant byte-for-byte is what keeps
those 3,476 labels, every published report and every checkpoint valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
PARAGRAPH_SEPARATOR = "\n\n"

# The historical bounds are internally consistent ratios of one size:
# 2600/1800 == 13/9 and 300/1800 == 1/6.
MAX_SIZE_NUMERATOR = 13
MAX_SIZE_DENOMINATOR = 9
MIN_SIZE_DIVISOR = 6

Measure = Callable[[str], int]
Reserve = Callable[[str], int]


def base_bounds() -> tuple[int, int, int]:
    """The configured historical bounds: (target, max, min)."""
    return (
        CONFIG["chunk_target_chars"],
        CONFIG["chunk_max_chars"],
        CONFIG["chunk_min_chars"],
    )


def derive_bounds(size: int, limit: int | None = None) -> tuple[int, int, int]:
    """Turn one chunk size into (target, max, min)."""
    target = max(int(size), 1)
    max_size = target * MAX_SIZE_NUMERATOR // MAX_SIZE_DENOMINATOR
    if limit is not None:
        max_size = min(max_size, int(limit))
        target = min(target, max_size)
    return target, max(max_size, target), max(target // MIN_SIZE_DIVISOR, 1)


def legacy_split(
    markdown: str,
    context: dict,
    size: int,
    overlap: int = 0,
    unit: str = "chars",
    provider: str | None = None,
):
    """Adapter to the `Chunker` protocol in `src.preprocess.chunkers`."""
    from src.preprocess.chunkers import TextChunk

    limit = None
    if unit == "tokens":
        from src.shared.tokenizers import provider_token_limit

        limit = provider_token_limit(provider or "")
    target, max_size, min_size = derive_bounds(size, limit)

    measure: Measure = len
    reserve: Reserve | None = None
    if unit == "tokens":
        from src.shared.tokenizers import token_counter

        measure = token_counter(provider or "")
        reserve = measure

    chunks = chunk_markdown(
        markdown,
        target,
        max_size,
        min_size,
        overlap=overlap,
        measure=measure,
        reserve=_page_reserve(reserve, context, context.get("chunk_version", "v1")),
    )
    return [
        TextChunk(
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            heading_path=chunk.heading_path or None,
        )
        for chunk in chunks
    ]


@dataclass(frozen=True)
class Paragraph:
    """One paragraph of a Markdown page, with its span in that page.

    Spans are offsets into the raw Markdown, aligned to paragraph boundaries.
    A paragraph longer than `max_size` is split, and its pieces get proportional
    sub-spans, so spans stay ordered and non-overlapping across a page.
    """

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    heading_path: str
    text: str
    start_char: int = 0
    end_char: int = 0


def chunk_markdown(
    markdown: str,
    target_chars: int | None = None,
    max_chars: int | None = None,
    min_chars: int | None = None,
    *,
    overlap: int = 0,
    measure: Measure | None = None,
    reserve: Reserve | None = None,
) -> list[Chunk]:
    """Slice a Markdown page into heading-aware chunks.

    `measure` decides what the size arguments count; it defaults to characters.
    `reserve` returns how much of the budget a chunk's representation prefix will
    consume for a given heading path, so token-sized chunks leave room for it.
    """
    target_size = target_chars or CONFIG["chunk_target_chars"]
    max_size = max_chars or CONFIG["chunk_max_chars"]
    min_size = min_chars or CONFIG["chunk_min_chars"]
    measure = measure or len
    # A character measure splits exactly, so it needs no size estimation.
    char_sized = measure is len
    join_size = measure(PARAGRAPH_SEPARATOR)

    chunks: list[Chunk] = []
    for heading_path, paragraphs in _sections(markdown):
        reserved = reserve(heading_path) if reserve else 0
        target_budget = max(target_size - reserved, 1)
        # A chunk can grow to one whole piece plus the repeated overlap tail, so
        # the piece ceiling has to leave room for that tail. Without this,
        # `max_size` stops being a ceiling as soon as overlap is enabled.
        max_budget = max(max_size - reserved - overlap, 1)
        pieces = [
            piece
            for paragraph in paragraphs
            for piece in _split_long_paragraph(
                paragraph, max_budget, measure, char_sized
            )
        ]
        chunks.extend(
            _pack(
                pieces,
                heading_path,
                target_budget,
                overlap,
                measure,
                join_size,
                char_sized,
            )
        )

    if len(chunks) == 1:
        return chunks
    return [chunk for chunk in chunks if measure(chunk.text) >= min_size]


def _page_reserve(
    reserve: Reserve | None,
    page,
    chunk_version: str,
) -> Reserve | None:
    """Budget the representation prefix that indexing will prepend.

    Measures the real `ChunkTextRepresentation` with an empty chunk text, so v1
    and v2 are both covered and a new representation cannot silently break the
    budget. Reserving it here is what keeps the *indexed* document under the
    embedder's sequence limit, not merely the chunk text.
    """
    if reserve is None:
        return None

    from src.indexing.chunk_text import PageChunk, chunk_text_representation

    representation = chunk_text_representation(chunk_version)
    marker = "x"
    marker_size = reserve(marker)

    def reserve_for(heading_path: str) -> int:
        # Measure the representation around a one-token marker and subtract it.
        # Measuring an empty chunk text instead would miss the separators the
        # representation puts between the prefix and the text, and a chunk would
        # land one token over the limit. The extra token absorbs the tokenizer
        # not being exactly compositional at that seam.
        document = representation.text_for_embedding(
            PageChunk(
                id="",
                page_id=page["id"],
                heading_path=heading_path,
                text=marker,
                title=page["title"],
                source=page["source"],
                language=page["language"],
                page_kind=page["page_kind"],
            )
        )
        return max(reserve(document) - marker_size, 0) + 1

    return reserve_for


def _pack(
    pieces: list[Paragraph],
    heading_path: str,
    target_budget: int,
    overlap: int,
    measure: Measure,
    join_size: int,
    char_sized: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[Paragraph] = []
    current_size = 0

    for piece in pieces:
        piece_size = measure(piece.text) + join_size
        if current and current_size + piece_size > target_budget:
            chunks.append(_chunk(heading_path, current))
            current = _overlap_tail(current, overlap, measure, join_size, char_sized)
            current_size = sum(measure(item.text) + join_size for item in current)
        current.append(piece)
        current_size += piece_size

    if current:
        chunks.append(_chunk(heading_path, current))
    return chunks


def _chunk(heading_path: str, pieces: list[Paragraph]) -> Chunk:
    return Chunk(
        heading_path=heading_path,
        text=PARAGRAPH_SEPARATOR.join(piece.text for piece in pieces),
        start_char=pieces[0].start,
        end_char=pieces[-1].end,
    )


def _overlap_tail(
    pieces: list[Paragraph],
    overlap: int,
    measure: Measure,
    join_size: int,
    char_sized: bool = True,
) -> list[Paragraph]:
    """Trailing text to repeat at the start of the next chunk.

    The tail must fit inside `overlap`, otherwise `max_size` stops bounding a
    chunk: a whole trailing paragraph can be many times the overlap budget. When
    the last piece is too big, a suffix of it is carried over instead.
    """
    if overlap <= 0 or not pieces:
        return []

    tail: list[Paragraph] = []
    size = 0
    for piece in reversed(pieces):
        piece_size = measure(piece.text) + join_size
        if size + piece_size <= overlap:
            tail.insert(0, piece)
            size += piece_size
            continue
        remaining = overlap - size - join_size
        fragment = _suffix(piece, remaining, measure, char_sized)
        if fragment is not None:
            tail.insert(0, fragment)
        break

    # Repeating every piece would emit the same chunk again.
    if len(tail) == len(pieces) and tail and tail[0].text == pieces[0].text:
        tail = tail[1:]
    return tail


def _suffix(
    piece: Paragraph,
    budget: int,
    measure: Measure,
    char_sized: bool,
) -> Paragraph | None:
    """Largest word-aligned suffix of `piece` fitting in `budget`."""
    text = piece.text
    if budget <= 0 or not text:
        return None

    size = measure(text)
    chars = budget if char_sized else max(int(budget * len(text) / max(size, 1)), 1)
    for _ in range(5):
        candidate = text if chars >= len(text) else text[-chars:]
        boundary = candidate.find(" ")
        if boundary > 0:
            candidate = candidate[boundary + 1 :]
        candidate = candidate.strip()
        if candidate and measure(candidate) <= budget:
            span = piece.end - piece.start
            width = max(int(span * len(candidate) / max(len(text), 1)), 1)
            return Paragraph(candidate, max(piece.end - width, piece.start), piece.end)
        chars = max(chars * 4 // 5, 1)
    return None


def _sections(markdown: str) -> list[tuple[str, list[Paragraph]]]:
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[Paragraph]]] = []
    paragraphs: list[Paragraph] = []
    current_lines: list[Paragraph] = []

    def flush_paragraph() -> None:
        if current_lines:
            text = "\n".join(line.text for line in current_lines).strip()
            if text:
                paragraphs.append(
                    Paragraph(text, current_lines[0].start, current_lines[-1].end)
                )
            current_lines.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            heading_path = " > ".join(title for _, title in heading_stack)
            sections.append((heading_path, paragraphs.copy()))
            paragraphs.clear()

    for line, start, end in _lines_with_spans(markdown):
        heading = HEADING_RE.match(line)
        if heading:
            flush_section()
            level = len(heading.group(1))
            title = _clean_heading(heading.group(2))
            heading_stack[:] = [
                (lvl, text) for lvl, text in heading_stack if lvl < level
            ]
            heading_stack.append((level, title))
            continue
        if not line:
            flush_paragraph()
            continue
        current_lines.append(Paragraph(line, start, end))

    flush_section()
    if sections:
        return sections
    return [("", [Paragraph(markdown.strip(), 0, len(markdown))])]


def _lines_with_spans(markdown: str) -> list[tuple[str, int, int]]:
    """Stripped lines with the span of their content in the raw Markdown."""
    lines: list[tuple[str, int, int]] = []
    offset = 0
    for raw in markdown.splitlines(keepends=True):
        content = raw.strip()
        start = offset + (len(raw) - len(raw.lstrip()))
        lines.append((content, start, start + len(content)))
        offset += len(raw)
    return lines


def _split_long_paragraph(
    paragraph: Paragraph,
    max_budget: int,
    measure: Measure,
    char_sized: bool = True,
) -> list[Paragraph]:
    size = measure(paragraph.text)
    if size <= max_budget:
        return [paragraph]

    if char_sized:
        return _pieces(paragraph, _split_on_chars(paragraph.text, max_budget))

    # Tokens per character varies, so estimate a character budget and tighten it
    # until every piece really fits. Dense scripts overshoot a flat estimate.
    limit = max(int(max_budget * len(paragraph.text) / size), 1)
    for _ in range(5):
        texts = _split_on_chars(paragraph.text, limit)
        if all(measure(text) <= max_budget for text in texts):
            return _pieces(paragraph, texts)
        limit = max(limit * 4 // 5, 1)
    return _pieces(paragraph, _split_on_chars(paragraph.text, limit))


def _pieces(paragraph: Paragraph, texts: list[str]) -> list[Paragraph]:
    """Subdivide a split paragraph's span proportionally to piece length.

    The pieces partition the paragraph in order, so proportional sub-spans stay
    monotonic and non-overlapping. They are approximate — line normalization
    means a piece is not a verbatim slice of the raw Markdown — but far tighter
    than giving every piece the whole paragraph's span.
    """
    if len(texts) == 1:
        return [Paragraph(texts[0], paragraph.start, paragraph.end)]

    total = sum(len(text) for text in texts) or 1
    span = paragraph.end - paragraph.start
    pieces: list[Paragraph] = []
    consumed = 0
    for text in texts:
        start = paragraph.start + span * consumed // total
        consumed += len(text)
        end = paragraph.start + span * consumed // total
        pieces.append(Paragraph(text, start, max(end, start + 1)))
    return pieces


def _split_on_chars(paragraph: str, max_chars: int) -> list[str]:
    pieces = []
    remaining = paragraph
    while len(remaining) > max_chars:
        split_at = remaining.rfind(". ", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = max_chars
        pieces.append(remaining[: split_at + 1].strip())
        remaining = remaining[split_at + 1 :].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip("# *")).strip()
