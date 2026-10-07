"""The wiki places dataset: load it. One pydantic model per Parquet file in `schemas/`,
each listing its file's columns. `Page` is what a system indexes (`Chunk` the same pages
cut at four sizes); the other files are questions about the pages, by page id
(`svwiki/Q123`). Test on `ok` rows (`answer_ok` in `Rag`), tune on `split == "dev"`.

    hard = load(Rag, ["id", "question"], answer_ok=True, kind="challenge").to_pandas()
    for q in read(Unanswerable, ok=True): print(q.question, q.why)
    texts = pages(hard.id)
    ids, vectors = embeddings("qwen3-embedding-0.6b", 512)  # rows of load(Chunk, size=512)
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.db.schemas.base import Row
from src.db.schemas.chunk import Chunk
from src.db.schemas.compare import Compare
from src.db.schemas.meta import Meta
from src.db.schemas.page import Page
from src.db.schemas.rag import Rag
from src.db.schemas.tables import Tables
from src.db.schemas.unanswerable import Unanswerable

__all__ = [
    "Chunk",
    "Compare",
    "Meta",
    "Page",
    "Rag",
    "Tables",
    "Unanswerable",
    "embeddings",
    "load",
    "pages",
    "read",
]

ROOT = Path(os.getenv("DATASET_DIR", "/data/llms4eu/wiki"))  # or a copy of it


def path(model: type[Row]) -> Path:
    return ROOT / model.file


def load(model: type[Row], columns: list[str] | None = None, **where):
    """The file as a pyarrow Table (`.to_pandas()`), only `columns`, rows matching `where`."""
    filters = [(k, "==", v) for k, v in where.items()] or None
    return pq.read_table(path(model), columns=columns, filters=filters)


def read(model: type[Row], **where) -> Iterator[Row]:
    """The matching rows as validated models, JSON columns parsed."""
    for batch in load(model, **where).to_batches():
        yield from map(model.model_validate, batch.to_pylist())


def pages(ids: Iterable[str]) -> dict[str, dict]:
    """Page id -> page (title, text, metadata)."""
    table = pq.read_table(path(Page), filters=[("id", "in", list(set(ids)))])
    return {p["id"]: p for p in table.to_pylist()}


def embeddings(model: str, size: int) -> tuple[list[str], np.ndarray]:
    """Chunk ids and their unit vectors (float16, memory-mapped), row i = chunk ids[i].
    Rows not embedded yet are NaN."""
    ids = load(Chunk, ["id"], size=size).column("id").to_pylist()
    vectors = np.load(ROOT / "embeddings" / model / f"{size}.npy", mmap_mode="r")
    assert len(ids) == len(vectors), "embeddings are from another chunking"
    return ids, vectors
