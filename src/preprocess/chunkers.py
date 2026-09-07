"""Chunking strategies, backed by `langchain-text-splitters`.

The splitting itself is library work: recursive descent through paragraph, line,
sentence and word separators until a piece fits, measured with the embedding
provider's own tokenizer. Nothing here re-implements that.

What is project-specific and therefore does live here:

- normalizing source Markdown to NFC, so the same EU text tokenizes and matches
  identically whether it arrived composed or decomposed;
- resolving each chunk's character span in the page, because `add_start_index`
  returns -1 whenever a chunk is not found verbatim;
- reserving the representation prefix (title, heading path) from the size budget,
  since the indexed document is longer than the chunk text;
- the frozen `legacy` strategy, which reproduces the chunk boundaries the
  approved eval labels were built on.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Callable, Protocol

STRATEGIES = ("recursive", "markdown", "legacy")
SIZE_UNITS = ("chars", "tokens")

# Separators tried in order. The library defaults stop at the word level; adding
# sentence punctuation first keeps a chunk boundary off the middle of a sentence.
# All of these are shared across the EU's Latin, Cyrillic and Greek scripts, and
# the trailing "" is the library's own last resort.
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# How far a chunk may exceed its target size while absorbing a short neighbour.
# Mirrors the legacy chunker's own 1,800-target/2,600-maximum ratio, and costs
# nothing real: every provider now reads its model's full context, so a tok256
# chunk reaching 384 tokens is still read whole by a model with 32,768 positions.
MERGE_TOLERANCE = 1.5


@dataclass(frozen=True)
class TextChunk:
    """One chunk with its span in the page it came from."""

    text: str
    start_char: int
    end_char: int
    heading_path: str | None = None


class Chunker(Protocol):
    name: str

    def split(self, markdown: str, context: dict) -> list[TextChunk]: ...


def normalize(markdown: str) -> str:
    """NFC-normalize page text before it is split, measured or stored.

    Slovenian č/š/ž and Greek/Cyrillic diacritics reach the scraper both
    precomposed and as combining sequences. Those forms tokenize to different
    token counts and compare unequal, so a chunk span resolved against one form
    will not be found in the other. Normalizing once, here, is what makes chunk
    spans and token budgets mean the same thing across languages.
    """
    return unicodedata.normalize("NFC", markdown)


def build_chunker(
    strategy: str,
    size: int,
    overlap: int,
    unit: str = "chars",
    provider: str | None = None,
    chunk_version: str = "v1",
    min_size: int = 0,
) -> Chunker:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    if unit not in SIZE_UNITS:
        raise ValueError(f"Unknown chunk size unit: {unit}")
    if unit == "tokens" and not provider:
        raise ValueError("Token-sized chunking needs an embedding provider")
    if overlap >= size:
        raise ValueError("Chunk overlap must stay below the chunk size")
    if min_size >= size:
        raise ValueError("Minimum chunk size must stay below the chunk size")

    if strategy == "legacy":
        return LegacyChunker(size=size, overlap=overlap, unit=unit, provider=provider)
    return LibraryChunker(
        name=strategy,
        size=size,
        overlap=overlap,
        unit=unit,
        provider=provider,
        chunk_version=chunk_version,
        split_headers=strategy == "markdown",
        min_size=min_size,
    )


@dataclass(frozen=True)
class LibraryChunker:
    """`RecursiveCharacterTextSplitter`, optionally per Markdown section."""

    name: str
    size: int
    overlap: int
    unit: str = "chars"
    provider: str | None = None
    chunk_version: str = "v1"
    split_headers: bool = False
    # Below this, a piece is folded into a neighbour rather than kept. Zero means
    # half the target size, which is what stops a chunking from being judged on
    # fragments it should never have emitted.
    min_size: int = 0

    def split(self, markdown: str, context: dict) -> list[TextChunk]:
        markdown = normalize(markdown)
        sections = (
            _markdown_sections(markdown) if self.split_headers else [(None, markdown)]
        )
        pieces: list[tuple[str, str | None]] = []
        for heading_path, text in sections:
            if not text.strip():
                continue
            splitter = self._splitter(context, heading_path)
            pieces.extend(
                (piece, heading_path)
                for piece in splitter.split_text(text)
                if piece.strip()
            )

        # Offsets are resolved against the whole page in one ordered pass. Doing
        # it per section would restart the offsets, because the header splitter
        # normalizes section text and it is no longer found verbatim.
        spans = _resolve_spans(markdown, [piece for piece, _ in pieces])
        chunks = [
            TextChunk(piece, start, end, heading_path or None)
            for (piece, heading_path), (start, end) in zip(pieces, spans)
        ]
        # Spans are resolved before merging, on pieces that really are verbatim in
        # the page. Merging first would concatenate text that no longer appears in
        # the source, and every span would fall back to a probe search.
        return _merge_undersized(
            chunks,
            measure(self.unit, self.provider),
            self.min_size or max(self.size // 2, 1),
            int(self.size * MERGE_TOLERANCE),
        )

    def _splitter(self, context: dict, heading_path: str | None):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # The indexed document is the chunk plus a title/heading prefix, so the
        # budget for the text itself is the size minus that prefix.
        reserved = _reserved_size(
            context, heading_path, self.unit, self.provider, self.chunk_version
        )
        size = max(self.size - reserved, 16)
        overlap = min(self.overlap, size - 1)

        if self.unit == "chars":
            return RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=SEPARATORS,
                keep_separator=True,
            )

        from src.shared.tokenizers import huggingface_tokenizer

        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            huggingface_tokenizer(self.provider or ""),
            chunk_size=size,
            chunk_overlap=overlap,
            separators=SEPARATORS,
            keep_separator=True,
        )


@dataclass(frozen=True)
class LegacyChunker:
    """The project's original heading-aware packing, kept for reproducibility.

    Frozen on purpose: the approved eval labels point at the chunks this
    produces, so its boundaries must not move. New work uses `recursive`.
    """

    size: int
    overlap: int
    unit: str = "chars"
    provider: str | None = None
    name: str = "legacy"

    def split(self, markdown: str, context: dict) -> list[TextChunk]:
        from src.preprocess.legacy_chunker import legacy_split

        return legacy_split(
            markdown,
            context,
            size=self.size,
            overlap=self.overlap,
            unit=self.unit,
            provider=self.provider,
        )


def _merge_undersized(
    chunks: list[TextChunk],
    size_of: Callable[[str], int],
    minimum: int,
    maximum: int,
) -> list[TextChunk]:
    """Fold short chunks into a neighbour instead of leaving or dropping them.

    A section's last piece is whatever is left over, so it is routinely a fragment
    that starts mid-sentence: 612 of tok256's 2,609 chunks came out under 300
    characters and 157 under 100. A fragment is a poor retrieval unit, it cannot
    carry a generated question, and excluding it downstream biases every
    comparison towards the cuttings that produce fewer of them.

    The legacy chunker's answer was to *drop* anything under 300 characters, which
    silently lost 0.91% of the corpus. Merging keeps every character. It costs only
    a chunk somewhat larger than its target, and no provider truncates any more.
    """
    merged: list[TextChunk] = []
    for chunk in chunks:
        if (
            merged
            and size_of(chunk.text) < minimum
            and size_of(merged[-1].text) + size_of(chunk.text) <= maximum
        ):
            merged[-1] = _join(merged[-1], chunk)
            continue
        merged.append(chunk)

    # The first chunk has no predecessor to fold into, so fold it forward.
    if len(merged) > 1 and size_of(merged[0].text) < minimum:
        if size_of(merged[0].text) + size_of(merged[1].text) <= maximum:
            merged[:2] = [_join(merged[0], merged[1])]
    return merged


def _join(first: TextChunk, second: TextChunk) -> TextChunk:
    """One chunk spanning both, keeping the earlier one's heading path."""
    return TextChunk(
        f"{first.text}\n\n{second.text}",
        min(first.start_char, second.start_char),
        max(first.end_char, second.end_char),
        first.heading_path or second.heading_path,
    )


def _markdown_sections(markdown: str) -> list[tuple[str | None, str]]:
    """Split on Markdown headings, keeping the heading path per section."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[(f"{'#' * level}", f"h{level}") for level in range(1, 7)],
        strip_headers=False,
    )
    sections = []
    for document in splitter.split_text(markdown):
        path = " > ".join(
            str(document.metadata[key])
            for key in sorted(document.metadata)
            if key.startswith("h")
        )
        sections.append((path or None, document.page_content))
    return sections or [(None, markdown)]


PROBE_CHARS = 40


def _resolve_spans(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Character span of each piece inside `text`, in order.

    The splitter's own `add_start_index` yields -1 whenever a piece is not found
    verbatim, which happens with overlap, whitespace stripping and Markdown
    header normalization. Searching forward from the previous piece's start keeps
    spans defined and non-decreasing, which is what any cross-variant comparison
    of chunk positions relies on.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        found = text.find(piece, cursor)
        if found < 0:
            probe = piece[:PROBE_CHARS].strip()
            found = text.find(probe, cursor) if probe else -1
        if found < 0:
            found = text.find(piece)
        if found < 0:
            found = cursor
        end = min(found + len(piece), len(text))
        spans.append((found, max(end, found)))
        cursor = max(found, cursor)
    return spans


def _reserved_size(
    context: dict,
    heading_path: str | None,
    unit: str,
    provider: str | None,
    chunk_version: str,
) -> int:
    from src.indexing.chunk_text import PageChunk, chunk_text_representation

    representation = chunk_text_representation(chunk_version)
    marker = "x"
    prefix = representation.text_for_embedding(
        PageChunk(
            id="",
            page_id=str(context.get("id", "")),
            heading_path=heading_path,
            text=marker,
            title=context.get("title"),
            source=context.get("source"),
            language=context.get("language"),
            page_kind=context.get("page_kind"),
        )
    )
    if unit == "chars":
        return max(len(prefix) - len(marker), 0) + 1

    from src.shared.tokenizers import token_counter

    count = token_counter(provider or "")
    return max(count(prefix) - count(marker), 0) + 1


def measure(unit: str, provider: str | None) -> Callable[[str], int]:
    """Size function matching a strategy's unit, for audits and tests."""
    if unit == "chars":
        return len

    from src.shared.tokenizers import token_counter

    return token_counter(provider or "")
