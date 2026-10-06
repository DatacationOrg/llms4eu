from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from src.db.pages import DEFAULT_CHUNK_VARIANT, chunk_variant, variant_tag
from src.db.pages import connect_pages as connect
from src.db.pages import initialize_page_artifacts_db
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(frozen=True)
class Chunk:
    heading_path: str
    text: str


def chunk_markdown(
    markdown: str,
    size: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
) -> list[Chunk]:
    """Pack each section's paragraphs into chunks of about `size` characters.

    Consecutive chunks of a section share up to `overlap` characters of trailing
    paragraphs. A chunk shorter than `min_chars` is merged into the one before
    it rather than dropped, so no page text is lost.
    """
    defaults = variant_settings(DEFAULT_CHUNK_VARIANT)
    size = size or defaults["size"]
    overlap = defaults["overlap"] if overlap is None else overlap
    min_chars = defaults["min_chars"] if min_chars is None else min_chars
    chunks: list[Chunk] = []
    for heading_path, paragraphs in _sections(markdown):
        pieces = [p for text in paragraphs for p in _split_long_paragraph(text, size)]
        for text in _pack(pieces, size, overlap):
            if chunks and len(text) < min_chars:
                previous = chunks[-1]
                if previous.heading_path != heading_path and heading_path:
                    text = f"{heading_path}\n\n{text}"
                chunks[-1] = Chunk(previous.heading_path, f"{previous.text}\n\n{text}")
            else:
                chunks.append(Chunk(heading_path, text))
    return chunks


def _pack(pieces: list[str], size: int, overlap: int) -> list[str]:
    # ponytail: overlap is whole paragraphs/sentences, so a piece longer than
    # `overlap` is never repeated; split pieces finer if overlap must be exact.
    texts: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and _length(current + [piece]) > size:
            texts.append("\n\n".join(current))
            carry: list[str] = []
            for previous in reversed(current):
                candidate = [previous, *carry]
                if _length(candidate) > overlap or _length(candidate + [piece]) > size:
                    break
                carry = candidate
            current = carry
        current.append(piece)
    if current:
        texts.append("\n\n".join(current))
    return texts


def _length(pieces: list[str]) -> int:
    return sum(len(piece) for piece in pieces) + 2 * (len(pieces) - 1)


def variant_settings(variant: str) -> dict[str, int]:
    """Size, overlap and min_chars of one named cut; `base` is the top-level config."""
    variants = {DEFAULT_CHUNK_VARIANT: {}, **CONFIG["chunk_variants"]}
    if variant not in variants:
        raise SystemExit(
            f"Unknown chunk variant {variant!r}; add it under chunk_variants in "
            "src/preprocess/config.yaml"
        )
    return {
        key.removeprefix("chunk_"): variants[variant].get(key, CONFIG[key])
        for key in ("chunk_size", "chunk_overlap", "chunk_min_chars")
    }


def rebuild_page_chunks(rechunk_all: bool = False, variant: str | None = None) -> None:
    variant = variant or chunk_variant()
    settings = variant_settings(variant)
    initialize_page_artifacts_db()
    with connect() as conn:
        if rechunk_all:
            _drop_chunks(conn, variant)
        pages = conn.execute(
            """
            select m.id, c.markdown
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            where m.page_kind != 'empty'
              and not exists (
                select 1
                from page_chunks chunks
                where chunks.page_id = m.id and chunks.variant = ?
              )
            order by m.id
            """,
            (variant,),
        ).fetchall()

        rows = [
            (
                f"{page['id']}{variant_tag(variant, ':')}:{index}",
                page["id"],
                variant,
                index,
                chunk.heading_path or None,
                chunk.text,
                len(chunk.text),
            )
            for page in pages
            for index, chunk in enumerate(chunk_markdown(page["markdown"], **settings))
        ]

        conn.executemany(
            """
            insert into page_chunks (
              id, page_id, variant, chunk_index, heading_path, text, char_count
            ) values (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    print(f"chunked {len(pages)} new pages into {len(rows)} {variant} chunks")


def _drop_chunks(conn, variant: str) -> None:
    """Delete one variant's chunks; refused while a label could not be rebuilt afterwards."""
    unanchored = conn.execute(
        """
        select count(distinct r.question_id) from eval_relevant_chunks r
        join page_chunks c on c.id = r.chunk_id and c.variant = ?
        where not exists (select 1 from eval_evidence e where e.question_id = r.question_id)
        """,
        (variant,),
    ).fetchone()[0]
    if unanchored:
        raise SystemExit(
            f"{unanchored} eval questions have no evidence quote, so rechunking would "
            "delete their labels for good. Run `just eval-evidence` first."
        )
    conn.execute("delete from page_chunks where variant = ?", (variant,))


def _sections(markdown: str) -> list[tuple[str, list[str]]]:
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    paragraphs: list[str] = []
    current_lines: list[str] = []

    def flush_paragraph() -> None:
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                paragraphs.append(text)
            current_lines.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            heading_path = " > ".join(title for _, title in heading_stack)
            sections.append((heading_path, paragraphs.copy()))
            paragraphs.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        heading = HEADING_RE.match(stripped)
        if heading:
            flush_section()
            level = len(heading.group(1))
            title = _clean_heading(heading.group(2))
            heading_stack[:] = [
                (lvl, text) for lvl, text in heading_stack if lvl < level
            ]
            heading_stack.append((level, title))
            continue
        if not stripped:
            flush_paragraph()
            continue
        current_lines.append(stripped)

    flush_section()
    return sections or [("", [markdown.strip()])]


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rechunk every page of the active variant (then `just rechunk` relabels)",
    )
    rebuild_page_chunks(parser.parse_args().rebuild)


if __name__ == "__main__":
    main()
