"""Step 2c: fuse BM25 on the full question with BM25 on its anchors. Does it rescue translated questions without
hurting same-language ones?

    nice -n 10 uv run python -m experiments.language_anchors.fusion --per-lang 100

For the original and the translated question alike (the system does not know which one it gets): BM25 on the full
question and on its realistic anchors (numbers + capitalised words of that question, `realistic_anchors`), page
rankings fused with reciprocal rank fusion (RRF, k = 60: score = sum of 1 / (k + rank)). A question without anchors
keeps its full ranking. hit@10 on the gold page, 95% Wilson intervals. DuckDB BM25 (fts.py), CPU.
"""

from __future__ import annotations

import argparse

import pandas as pd

from experiments.language_anchors.fts import FtsRetriever
from experiments.language_anchors.translated_topk import (
    DEPTH,
    lang_group,
    realistic_anchors,
    sample,
    top_pages,
    wilson,
)
from src.db.dataset import chunk_size

RRF_K = 60


def rrf(*rankings: list[str]) -> list[str]:
    """Pages ordered by reciprocal rank fusion over the given page rankings."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, page in enumerate(ranking, 1):
            score[page] = score.get(page, 0.0) + 1 / (RRF_K + rank)
    return sorted(score, key=lambda p: -score[p])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=int, default=300, help="questions per kind (easy, hard)"
    )
    ap.add_argument(
        "--per-lang",
        type=int,
        default=0,
        help="stratified: questions per kind and language (overrides --n)",
    )
    args = ap.parse_args()

    df = sample(args.n, args.per_lang)
    bm25 = FtsRetriever(chunk_size())
    rows = []
    for version in ("question", "question_x"):
        full_texts = df[version].tolist()
        anchor_texts = [realistic_anchors(t) for t in full_texts]
        full = bm25.retrieve_batch(full_texts, DEPTH)
        anchor = bm25.retrieve_batch([a or "-" for a in anchor_texts], DEPTH)
        for i, r in enumerate(df.itertuples()):
            full_pages = top_pages([c.id for c in full[i]], DEPTH)
            anchor_pages = (
                top_pages([c.id for c in anchor[i]], DEPTH) if anchor_texts[i] else []
            )
            fused = rrf(full_pages, anchor_pages) if anchor_pages else full_pages
            rows.append(
                {
                    "kind": r.kind,
                    "version": "original" if version == "question" else "translated",
                    "page_group": lang_group(r.lang),
                    "query_group": lang_group(
                        r.lang if version == "question" else r.x_lang
                    ),
                    "full": r.id in full_pages[:10],
                    "anchors": r.id in anchor_pages[:10],
                    "fused": r.id in fused[:10],
                }
            )
    t = pd.DataFrame(rows)
    hit = {m: (m, wilson) for m in ("full", "anchors", "fused")}
    print(
        f"\nBM25 (duckdb), {chunk_size()}-token chunks, {(df.kind == 'corpus').sum()} easy + {(df.kind == 'challenge').sum()} hard; hit@10 [95% interval]\n"
    )
    print(t.groupby(["kind", "version"]).agg(n=("full", "size"), **hit).to_markdown())
    print("\nBy query language group (the language the question is asked in):\n")
    print(
        t.groupby(["kind", "version", "query_group"])
        .agg(n=("full", "size"), **hit)
        .to_markdown()
    )


if __name__ == "__main__":
    main()
