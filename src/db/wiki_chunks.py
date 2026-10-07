"""Build `wikipages.parquet` and `chunks.parquet` from `pages.jsonl` (read only), adding the
Ling notes (`summary`, `role`) found in `notes.db`. Run again to pick up new notes.

Chunking follows aihub-core (`aihub_core/search/chunking.py`): split at the page's
headings, pack neighbouring sections up to the target size, size-split only a section too
big on its own; no overlap. The chunk keeps the shared heading trail as `breadcrumb`, and
the headings below it inline. Sizes are Qwen3-Embedding tokens of the chunk text alone.

    uv run python -m src.db.wiki_chunks [--rechunk]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from multiprocessing import Pool
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.db.dataset import ROOT, Chunk, Page, path

SIZES = (256, 512, 1024, 2048)
TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
# `# <title>` opens every page; the title is embedded separately, so it is no breadcrumb.
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
NOTES = ROOT / "notes.db"


class _Section(NamedTuple):
    text: str
    trail: tuple[str, ...]
    cost: int  # tokens of the text and every heading line it may get back


def chunk_page(page: dict) -> list[dict]:
    """The page's chunks at every size, in order."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    sections = []
    for doc in MarkdownHeaderTextSplitter(_HEADERS).split_text(page["text"]):
        trail = tuple(doc.metadata[k] for k in ("h2", "h3") if k in doc.metadata)
        cost = _count(doc.page_content) + sum(_count(h) + 2 for h in trail)
        sections.append(_Section(doc.page_content, trail, cost))
    rows = []
    for size in SIZES:
        pieces = []
        for group in _pack(sections, size):
            shared = _shared_trail(group)
            body = _render(group, shared)
            split = _count(body) > size
            for text in _sizer(size).split_text(body) if split else [body]:
                pieces.append((" > ".join(shared) or None, text))
        for n, (breadcrumb, text) in enumerate(pieces):
            rows.append(
                {
                    "id": f"{page['id']}:{size}:{n}",
                    "page_id": page["id"],
                    "size": size,
                    "n": n,
                    "title": page["title"],
                    "breadcrumb": breadcrumb,
                    "text": text,
                    "tokens": _count(text),
                }
            )
    return rows


def _pack(sections: list[_Section], size: int):
    """Neighbouring sections grouped up to `size`; one too big alone gets its own group."""
    current, used = [], 0
    for section in sections:
        if section.cost > size:
            if current:
                yield current
            yield [section]
            current, used = [], 0
            continue
        if current and used + section.cost + 1 > size:
            yield current
            current, used = [], 0
        current.append(section)
        used += section.cost + 1
    if current:
        yield current


def _shared_trail(group: list[_Section]) -> tuple[str, ...]:
    shared = group[0].trail
    for section in group[1:]:
        depth = 0
        while (
            depth < min(len(shared), len(section.trail))
            and shared[depth] == section.trail[depth]
        ):
            depth += 1
        shared = shared[:depth]
    return shared


def _render(group: list[_Section], shared: tuple[str, ...]) -> str:
    """Each section with its headings below the shared trail put back as Markdown."""
    parts = []
    for section in group:
        below = section.trail[len(shared) :]
        lines = [f"{'#' * (len(shared) + i + 2)} {h}" for i, h in enumerate(below)]
        parts.append("\n".join([*lines, section.text]))
    return "\n\n".join(parts)


_tokenizer = None
_sizers: dict[int, object] = {}


def _count(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])


def _sizer(size: int):
    if size not in _sizers:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _count("")  # loads the tokenizer
        _sizers[size] = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            _tokenizer, chunk_size=size, chunk_overlap=0
        )
    return _sizers[size]


def _notes(table: str) -> dict[str, str]:
    if not NOTES.exists():
        return {}
    with sqlite3.connect(NOTES) as db:
        return dict(db.execute(f"select id, text from {table} where text is not null"))


def _write(rows: list[dict], target) -> None:
    """Write next to the target, then swap it in: a crash never leaves half a file."""
    tmp = target.with_suffix(".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows), tmp, compression="zstd", row_group_size=50_000
    )
    os.replace(tmp, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="allow new chunk ids (embeddings become stale)",
    )
    args = parser.parse_args()

    with open(ROOT / "pages.jsonl") as fh:
        pages = [json.loads(line) for line in fh]
    summaries = _notes("summaries")
    for page in pages:
        page["summary"] = summaries.get(page["id"])

    with Pool(16) as pool:
        chunks = [
            c for page_chunks in pool.imap(chunk_page, pages, 64) for c in page_chunks
        ]
    chunks.sort(key=lambda c: c["size"])  # stable: page order and n kept within a size
    roles = _notes("roles")
    for chunk in chunks:
        chunk["role"] = roles.get(chunk["id"])

    if path(Chunk).exists() and not args.rechunk:
        old = pq.read_table(path(Chunk), columns=["id"]).column("id").to_pylist()
        if old != [c["id"] for c in chunks]:
            raise SystemExit("chunk ids changed: rerun with --rechunk, then re-embed")
    _write(pages, path(Page))
    _write(chunks, path(Chunk))
    print(
        f"{len(pages)} pages, {len(chunks)} chunks",
        {s: sum(c["size"] == s for c in chunks) for s in SIZES},
    )


if __name__ == "__main__":
    main()
