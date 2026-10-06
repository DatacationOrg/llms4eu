"""Score the same retrieval methods on every chunk variant, in one table.

Each variant is chunked, labelled from the evidence quotes, indexed for the
providers the methods need, and evaluated in turn. `hit@k` and `recall@k` count
whole chunks, and a chunk twice as long is about twice as likely to hold any
given answer, so the table also shows each variant's chunk count and the
characters a reader gets back at k. Every variant's eval is checkpointed, so a
stopped sweep resumes where it was.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from src.db.pages import connect_pages as connect
from src.eval.evaluate import CONFIG, EvalRun, REPORTS_DIR, run_eval
from src.eval.evidence import relabel
from src.eval.metrics import plain_table
from src.indexing.store import rebuild_chunk_collection
from src.preprocess.chunker import rebuild_page_chunks, variant_settings
from src.retrieval.methods import missing_retriever_indexes
from src.shared.env import load_local_env


def compare(
    variants: list[str],
    methods: list[str],
    limit: int | None,
    category: str | None,
    output: Path,
) -> None:
    rows = []
    for variant in variants:
        variant_settings(variant)  # fail on a typo before any work
        # Every stage reads the variant from the environment, as a shell would set it.
        os.environ["CHUNK_VARIANT"] = variant
        rebuild_page_chunks()
        relabel()
        for provider in sorted(set(missing_retriever_indexes(methods).values())):
            rebuild_chunk_collection(provider)
        run = run_eval(methods, limit=limit, category=category, checkpoint=True)
        rows.extend(variant_rows(variant, run, _char_counts(variant)))

    report = "\n".join(
        [
            f"# Chunk variants, {date.today():%Y-%m-%d}",
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
    "variant",
    "method",
    "chunks",
    "mean chars",
    "questions",
    *_SCORE_NAMES,
    f"chars@{CONFIG['category_hit_k']}",
    "ms/query",
]


def variant_rows(
    variant: str, run: EvalRun, char_counts: dict[str, int]
) -> list[list[str]]:
    """One table row per method: the variant's shape, its scores, and its cost."""
    k = CONFIG["category_hit_k"]
    chunks = len(char_counts)
    mean_chars = sum(char_counts.values()) / chunks if chunks else 0
    return [
        [
            variant,
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


def _char_counts(variant: str) -> dict[str, int]:
    with connect() as conn:
        return {
            row["id"]: row["char_count"]
            for row in conn.execute(
                "select id, char_count from page_chunks where variant = ?", (variant,)
            )
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True, help="comma-separated names")
    parser.add_argument("--methods", default=",".join(CONFIG["default_methods"]))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / f"chunk_variants_{date.today():%Y-%m-%d}.md",
    )
    args = parser.parse_args()
    load_local_env()
    compare(
        [name.strip() for name in args.variants.split(",") if name.strip()],
        [name.strip() for name in args.methods.split(",") if name.strip()],
        args.limit,
        args.category,
        args.output,
    )


if __name__ == "__main__":
    main()
