from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db,
)
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(frozen=True)
class Chunk:
    heading_path: str
    text: str


def chunk_markdown(
    markdown: str,
    target_chars: int | None = None,
    max_chars: int | None = None,
    min_chars: int | None = None,
) -> list[Chunk]:
    target_chars = target_chars or CONFIG["chunk_target_chars"]
    max_chars = max_chars or CONFIG["chunk_max_chars"]
    min_chars = min_chars or CONFIG["chunk_min_chars"]
    sections = _sections(markdown)
    chunks: list[Chunk] = []
    for heading_path, paragraphs in sections:
        current: list[str] = []
        current_chars = 0
        for paragraph in paragraphs:
            for piece in _split_long_paragraph(paragraph, max_chars):
                if current and current_chars + len(piece) + 2 > target_chars:
                    chunks.append(Chunk(heading_path, "\n\n".join(current)))
                    current = []
                    current_chars = 0
                current.append(piece)
                current_chars += len(piece) + 2
        if current:
            chunks.append(Chunk(heading_path, "\n\n".join(current)))

    if len(chunks) == 1:
        return chunks
    return [chunk for chunk in chunks if len(chunk.text) >= min_chars]


def rebuild_page_chunks() -> None:
    initialize_page_artifacts_db()
    with connect() as conn:
        pages = conn.execute(
            """
            select m.id, c.markdown
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            where m.page_kind != 'empty'
            order by m.id
            """
        ).fetchall()
        conn.execute("delete from page_chunks")

        rows = []
        for page in pages:
            chunks = chunk_markdown(page["markdown"])
            for index, chunk in enumerate(chunks):
                rows.append(
                    (
                        f"{page['id']}:{index}",
                        page["id"],
                        index,
                        chunk.heading_path or None,
                        chunk.text,
                        len(chunk.text),
                    )
                )

        conn.executemany(
            """
            insert into page_chunks (
              id, page_id, chunk_index, heading_path, text, char_count
            ) values (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    print(f"chunked {len(pages)} pages into {len(rows)} chunks")


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
    parser.parse_args()
    rebuild_page_chunks()


if __name__ == "__main__":
    main()
