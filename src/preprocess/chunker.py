"""Cut the pages into `chunks.parquet` at every size in `SIZES` (`src/db/dataset.py`).

Follows aihub-core (`aihub_core/search/chunking.py`): split at the page's headings,
pack neighbouring sections up to the size, size-split only a section too big on its own
(at paragraph, line, then sentence ends); no overlap. A chunk keeps the heading trail
its sections share as `breadcrumb` and the headings below it inline. Wikipedia edit links
(`[Bearbeiten | Quelltext bearbeiten]`) are dropped, and with them sections that held
nothing else; a piece under MIN_TOKENS joins its neighbour. Sizes are tokens of the
chunk text alone, counted with the default embedding provider's tokenizer.

On a rechunk, a chunk whose embedded text did not change keeps its `role` and its vectors
in every `embeddings/<model>/<size>.npy`; the others are NaN until re-embedded.

    uv run python -m src.preprocess.chunker
"""

from __future__ import annotations

import argparse
import os
import re
from functools import cache
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa

from src.db.dataset import ROOT, SIZES, Chunk, Page, load, path, write
from src.db.schemas.chunk import represent
from src.shared.env import load_yaml

EMBEDDING = load_yaml(Path(__file__).parents[1] / "indexing" / "config.yaml")
MIN_TOKENS = 50
TOKENIZER = EMBEDDING["providers"][EMBEDDING["default_provider"]]  # model, revision
# `# <title>` opens every page; the title is embedded separately, so it is no breadcrumb.
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_EDIT_LINK = re.compile(r"\[[^\[\]|\n]{1,40} \| [^\[\]\n]{1,60}\]")
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class _Section(NamedTuple):
    text: str
    trail: tuple[str, ...]
    cost: int  # tokens of the text and every heading line it may get back


def chunk_page(page: dict) -> list[dict]:
    """The page's chunks at every size, in order."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    sections = []
    for doc in MarkdownHeaderTextSplitter(_HEADERS).split_text(page["text"]):
        text = _EDIT_LINK.sub("", doc.page_content).strip()
        if not text:
            continue
        trail = tuple(doc.metadata[k] for k in ("h2", "h3") if k in doc.metadata)
        cost = _count(text) + sum(_count(h) + 2 for h in trail)
        sections.append(_Section(text, trail, cost))
    rows = []
    for size in SIZES:
        pieces: list[list] = []  # [breadcrumb, text, tokens]
        for group in _pack(sections, size):
            shared = _shared_trail(group)
            body = _render(group, shared)
            split = _count(body) > size
            for text in _sizer(size).split_text(body) if split else [body]:
                pieces.append([" > ".join(shared) or None, text, _count(text)])
        for n, (breadcrumb, text, tokens) in enumerate(_merge_tiny(pieces, size)):
            rows.append(
                {
                    "id": f"{page['id']}:{size}:{n}",
                    "page_id": page["id"],
                    "size": size,
                    "n": n,
                    "title": page["title"],
                    "breadcrumb": breadcrumb,
                    "text": text,
                    "tokens": tokens,
                }
            )
    return rows


def _merge_tiny(pieces: list[list], size: int) -> list[list]:
    """A piece under MIN_TOKENS joins the one before it (the first one, the next),
    as long as the two stay within size + MIN_TOKENS."""
    out: list[list] = []
    for piece in pieces:
        tiny = piece[2] < MIN_TOKENS or (out and out[-1][2] < MIN_TOKENS)
        if out and tiny and out[-1][2] + piece[2] + 2 <= size + MIN_TOKENS:  # 2: "\n\n"
            out[-1][1] += "\n\n" + piece[1]
            out[-1][2] = _count(out[-1][1])
        else:
            out.append(piece)
    return out


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


@cache
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        TOKENIZER["model_name"], revision=TOKENIZER["revision"]
    )


def _count(text: str) -> int:
    return len(_tokenizer()(text, add_special_tokens=False)["input_ids"])


@cache
def _sizer(size: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _tokenizer(),
        chunk_size=size,
        chunk_overlap=0,
        separators=_SEPARATORS,
        keep_separator="end",
    )


def _key(chunk: dict) -> tuple:
    return chunk["page_id"], represent(
        chunk["title"], chunk["breadcrumb"], chunk["text"]
    )


def _carry_over(chunks: list[dict]) -> list[tuple]:
    """Keep the role and vectors of every chunk whose embedded text did not change.
    Returns the (written, target) vector files to swap in with the new chunks."""
    if not path(Chunk).exists():
        return []
    columns = ["page_id", "size", "title", "breadcrumb", "text", "role"]
    old = load(Chunk, columns).to_pylist()
    swaps = []
    for size in SIZES:
        before = [c for c in old if c["size"] == size]
        row = {_key(c): i for i, c in enumerate(before)}
        after = [c for c in chunks if c["size"] == size]
        source = np.array([row.get(_key(c), -1) for c in after])
        kept = source >= 0
        for chunk, i in zip(after, source):
            chunk["role"] = before[i]["role"] if i >= 0 else None
        for target in sorted((ROOT / "embeddings").glob(f"*/{size}.npy")):
            vectors = np.load(target, mmap_mode="r")
            new = np.full((len(after), vectors.shape[1]), np.nan, np.float16)
            if len(vectors) == len(before):  # else from another cut: re-embed all
                new[kept] = vectors[source[kept]]
            written = target.with_name(target.name + ".tmp")
            with open(written, "wb") as fh:
                np.save(fh, new)
            swaps.append((written, target))
        print(f"{size}: {kept.sum()} of {len(after)} chunks unchanged", flush=True)
    return swaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    workers = parser.parse_args().workers
    if not path(Page).exists():
        raise SystemExit("no wikipages.parquet yet: run python -m src.preprocess.pages")
    pages = load(Page, ["id", "title", "text"]).to_pylist()
    with Pool(workers) as pool:
        chunks = [c for cs in pool.imap(chunk_page, pages, chunksize=64) for c in cs]
    chunks.sort(key=lambda c: c["size"])  # stable: page order and n kept within a size
    for chunk in chunks:
        chunk["role"] = None
    swaps = _carry_over(chunks)
    # ponytail: chunks and vectors are swapped one rename after another, not atomically;
    # after a crash between them, a rerun finds the row counts differ and re-embeds.
    write(Chunk, pa.Table.from_pylist(chunks))
    for written, target in swaps:
        os.replace(written, target)
    print(
        f"{len(chunks)} chunks", {s: sum(c["size"] == s for c in chunks) for s in SIZES}
    )


if __name__ == "__main__":
    main()
