"""Copy the page database to a scratch location for destructive experiments.

Chunk-variant work rewrites `page_chunks`, and `eval_relevant_chunks` cascades
on delete from it, so a mistake against `data/db/pages.db` destroys the approved
labels. Experiments point `PAGES_DB_PATH` at the copy this makes instead.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.shared.env import ROOT

DEFAULT_SOURCE = "data/db/pages.db"
DEFAULT_TARGET = ".local/db/pages-variants.db"


def snapshot_pages_db(source: str, target: str) -> Path:
    source_path = _resolve(source)
    target_path = _resolve(target)
    if not source_path.exists():
        raise RuntimeError(f"Source database does not exist: {source_path}")
    if target_path == source_path:
        raise RuntimeError("Refusing to snapshot a database onto itself")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # The backup API copies a consistent snapshot even while the source is open,
    # which a file copy does not guarantee. It only ever reads the source.
    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as source_conn:
        with sqlite3.connect(target_path) as target_conn:
            source_conn.backup(target_conn)
            target_conn.execute("vacuum")
    return target_path


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    parser.add_argument("target", nargs="?", default=DEFAULT_TARGET)
    args = parser.parse_args()
    target = snapshot_pages_db(args.source, args.target)
    print(f"snapshot written to {target}")
    print(f"run experiments with PAGES_DB_PATH={target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
