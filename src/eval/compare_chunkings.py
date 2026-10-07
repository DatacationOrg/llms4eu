"""Score the same retrieval methods on every chunk size, in one table.

Each size is evaluated in turn on its own chunks and stored vectors (embed them first
with `python -m src.indexing`). `hit@k` and `recall@k` count whole chunks, and a chunk
twice as long is about twice as likely to hold any given answer, so the table also
shows each size's chunk count and the characters a reader gets back at k. Every size's
eval is checkpointed, so a stopped sweep resumes where it was.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from src.db.dataset import SIZES, Chunk, load
from src.eval.evaluate import CONFIG, REPORTS_DIR, EvalRun, run_eval
from src.eval.metrics import plain_table
from src.shared.env import load_local_env


def compare(
    sizes: list[int],
    methods: list[str],
    limit: int | None,
    category: str | None,
    output: Path,
) -> None:
    rows = []
    for size in sizes:
        # Every stage reads the size from the environment, as a shell would set it.
        os.environ["CHUNK_SIZE"] = str(size)
        run = run_eval(methods, limit=limit, category=category, checkpoint=True)
        rows.extend(size_rows(str(size), run, _char_counts(size)))

    report = "\n".join(
        [
            f"# Chunk sizes, {date.today():%Y-%m-%d}",
            "",
            f"Methods: {', '.join(methods)}",
            f"Limit: {limit if limit is not None else 'all'}",
            f"Category: {category or 'all'}",
            "",
            plain_table(_HEADERS, rows),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved report: {output}")


_SCORE_NAMES = [
    "hit@1",
    f"hit@{CONFIG['category_hit_k']}",
    f"recall@{max(CONFIG['metric_ks'])}",
    f"mrr@{CONFIG['mrr_k']}",
]
_HEADERS = [
    "size",
    "method",
    "chunks",
    "mean chars",
    "questions",
    *_SCORE_NAMES,
    f"chars@{CONFIG['category_hit_k']}",
    "ms/query",
]


def size_rows(size: str, run: EvalRun, char_counts: dict[str, int]) -> list[list[str]]:
    """One table row per method: the size's shape, its scores, and its cost."""
    k = CONFIG["category_hit_k"]
    chunks = len(char_counts)
    mean_chars = sum(char_counts.values()) / chunks if chunks else 0
    return [
        [
            size,
            method,
            str(chunks),
            f"{mean_chars:.0f}",
            str(len(run.questions)),
            *(f"{run.scores[method][name]:.3f}" for name in _SCORE_NAMES),
            f"{chars_at_k(run.rankings[method], char_counts, k):.0f}",
            f"{run.timings[method]['ms_per_query']:.1f}",
        ]
        for method in run.methods
    ]


def chars_at_k(
    rankings: dict[str, list[str]], char_counts: dict[str, int], k: int
) -> float:
    """Mean characters in a question's top-k results: what a reader is handed."""
    if not rankings:
        return 0.0
    totals = [
        sum(char_counts.get(chunk_id, 0) for chunk_id in ranked[:k])
        for ranked in rankings.values()
    ]
    return sum(totals) / len(totals)


def _char_counts(size: int) -> dict[str, int]:
    chunks = load(Chunk, ["id", "text"], size=size).to_pydict()
    return {i: len(text) for i, text in zip(chunks["id"], chunks["text"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--methods", default=",".join(CONFIG["default_methods"]))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / f"chunk_sizes_{date.today():%Y-%m-%d}.md",
    )
    args = parser.parse_args()
    load_local_env()
    compare(
        args.sizes,
        [name.strip() for name in args.methods.split(",") if name.strip()],
        args.limit,
        args.category,
        args.output,
    )


if __name__ == "__main__":
    main()
