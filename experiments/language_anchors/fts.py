"""A lean BM25 over the chunks: DuckDB full-text search, index on disk, memory capped.

The pipeline's BM25 (`src/retrieval/retrievers/sparse.py`) builds its index in memory (~4 GB burst); on the shared
server systemd-oomd kept killing that build under memory pressure. This index is built once into
out/bm25-<size>.duckdb (~0.7 GB) and queried at ~0.1 s per query with ~0.7 GB of memory. It is a different BM25
implementation (DuckDB's tokenizer and parameters), so compare query versions within one engine, not across engines.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import duckdb

from src.db.dataset import ROOT

OUT = Path(__file__).parent / "out"
# keep letters of every script and digits; everything else separates words (like the pipeline's `[^\W_]+`)
IGNORE = r"(\\.|[^a-z\\p{L}0-9])+"


@dataclass(frozen=True)
class Hit:
    id: str
    score: float


class FtsRetriever:
    def __init__(self, size: int, memory_limit: str = "3GB") -> None:
        path = OUT / f"bm25-{size}.duckdb"
        if not path.exists():
            OUT.mkdir(exist_ok=True)
            build = duckdb.connect(str(path))
            build.sql(f"set memory_limit='{memory_limit}'")
            build.sql(
                f"create table chunks as select id, text from '{ROOT / 'chunks.parquet'}' where size = {int(size)}"  # nosec B608 - constant path and an int
            )
            build.sql(
                "pragma create_fts_index('chunks', 'id', 'text', stemmer='none', stopwords='none', "
                f"ignore='{IGNORE}', strip_accents=0, lower=1)"
            )
            build.close()
        self.con = duckdb.connect(str(path), read_only=True)
        # gentle on the shared server: one thread per query; parallelism comes from a few queries at once
        self.con.sql(f"set memory_limit='{memory_limit}'; set threads=1")

    def retrieve_batch(
        self, queries: list[str], limit: int, workers: int = 4
    ) -> dict[int, list[Hit]]:
        """Queries in parallel: one read-only cursor per thread (DuckDB releases the GIL while it scans)."""
        sql = (
            "select id, s from (select id, fts_main_chunks.match_bm25(id, ?) as s from chunks) "
            f"where s is not null order by s desc limit {int(limit)}"  # nosec B608 - an int; the query is a bound parameter
        )
        local = threading.local()

        def search(query: str) -> list[Hit]:
            if not hasattr(local, "cursor"):
                local.cursor = self.con.cursor()
            return [Hit(*row) for row in local.cursor.execute(sql, [query]).fetchall()]

        with ThreadPoolExecutor(workers) as pool:
            return dict(enumerate(pool.map(search, queries)))
