"""Load the wiki places QA test set: questions about the pages in `pages.jsonl`.

The release is a folder of zstd Parquet files (`wiki_qa_<name>.parquet`, guide in
its `README.md`), by default `/data/llms4eu/wiki/qa`; set `WIKI_QA_DIR` to read a
copy elsewhere. `load` reads only the asked columns and pushes equality filters
into the Parquet scan; `rows` also decodes the JSON-string columns; `pages`
fetches article texts by id from the `pages.jsonl` beside the folder.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(os.getenv("WIKI_QA_DIR", "/data/llms4eu/wiki/qa"))

# Columns stored as JSON strings: their keys vary from row to row.
JSON_COLUMNS = {
    "rag": {"labels", "criteria", "page_tags", "variants"},
    "unanswerable": {"check"},
    "compare": {"check"},
    "meta": {"check"},
    "tables": {"answer", "answer_final", "sonnet"},
    "clean": {"item", "labels", "original"},
    "challenge": {"item", "labels"},
}


def load(name: str, columns: list[str] | None = None, where: dict | None = None):
    """One file as a pyarrow Table (`.to_pandas()` for a DataFrame), e.g.
    load("rag", ["id", "question"], {"answer_ok": True, "split": "test"})."""
    filters = [(k, "==", v) for k, v in (where or {}).items()] or None
    return pq.read_table(
        ROOT / f"wiki_qa_{name}.parquet", columns=columns, filters=filters
    )


def rows(
    name: str, columns: list[str] | None = None, where: dict | None = None
) -> Iterator[dict]:
    """Rows of `load` as dicts, JSON-string columns decoded."""
    decode = JSON_COLUMNS[name]
    for batch in load(name, columns, where).to_batches():
        for row in batch.to_pylist():
            yield {
                k: json.loads(v) if k in decode and v is not None else v
                for k, v in row.items()
            }


def pages(ids: Iterable[str]) -> dict[str, dict]:
    """Page id -> page row (title, text, metadata) for the given ids; one pass over pages.jsonl."""
    want = set(ids)
    with open(ROOT.parent / "pages.jsonl") as fh:
        return {p["id"]: p for p in map(json.loads, fh) if p["id"] in want}
