"""Build `wikipages.parquet` from the scraped `pages.jsonl` (read only).

A rerun keeps the `summary` already in the Parquet file, by page id.

    uv run python -m src.preprocess.pages
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.json as pj

from src.db.dataset import ROOT, Page, load, path, write


def main() -> None:
    table = pj.read_json(ROOT / "pages.jsonl")
    summaries = {}
    if path(Page).exists():
        old = load(Page, ["id", "summary"])
        summaries = dict(
            zip(old.column("id").to_pylist(), old.column("summary").to_pylist())
        )
    summary = [summaries.get(i) for i in table.column("id").to_pylist()]
    write(Page, table.append_column("summary", pa.array(summary, pa.string())))
    print(
        f"{table.num_rows} pages, {sum(s is not None for s in summary)} with a summary"
    )


if __name__ == "__main__":
    main()
