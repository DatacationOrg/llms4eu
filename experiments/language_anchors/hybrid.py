"""Fusion inside the hybrid: dense + BM25 + BM25 on the anchors. Does the combination beat each part, in the same
language and across languages?

    nice -n 10 uv run python -m experiments.language_anchors.hybrid --per-lang 100

Same stratified sample as fusion.py. For the original and the translated question: dense (Qwen3-Embedding-0.6B on
the CHUNK_SIZE chunks, GPU for the query embeddings), BM25 on the full question, and BM25 on its realistic anchors
(DuckDB, cached). Page rankings combined with reciprocal rank fusion (k = 60), so every combination uses the same
rule; the pipeline's own hybrid uses weighted score fusion (0.7 dense, 0.3 BM25) instead. hit@10, 95% intervals.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from experiments.language_anchors.fts import FtsRetriever
from experiments.language_anchors.fusion import rrf
from experiments.language_anchors.translated_topk import (
    DEPTH,
    lang_group,
    realistic_anchors,
    sample,
    top_pages,
    wilson,
)
from src.db.dataset import chunk_size
from src.retrieval.retrievers.vector import VectorChunkRetriever

COMBINATIONS = {
    "dense": ("dense",),
    "bm25": ("bm25",),
    "dense+bm25": ("dense", "bm25"),
    "bm25+anchors": ("bm25", "anchors"),
    "dense+anchors": ("dense", "anchors"),
    "dense+bm25+anchors": ("dense", "bm25", "anchors"),
}


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
    # the query-embedding cache lives under LLMS4EU_DATA, which is read-only on the shared disk: keep ours local
    os.environ.setdefault(
        "LLMS4EU_DATA", str(Path(__file__).parent / "out" / "artifacts")
    )

    df = sample(args.n, args.per_lang)
    bm25 = FtsRetriever(chunk_size())
    dense = VectorChunkRetriever(name="qwen", provider="qwen")
    rows = []
    for version in ("question", "question_x"):
        full = df[version].tolist()
        anchor_texts = [realistic_anchors(t) or "-" for t in full]
        found = {
            "dense": dense.retrieve_batch(full, DEPTH),
            "bm25": bm25.retrieve_batch(full, DEPTH),
            "anchors": bm25.retrieve_batch(anchor_texts, DEPTH),
        }
        for i, r in enumerate(df.itertuples()):
            ranking = {
                k: top_pages([c.id for c in v[i]], DEPTH) for k, v in found.items()
            }
            if anchor_texts[i] == "-":
                ranking["anchors"] = []
            row = {
                "kind": r.kind,
                "version": "original" if version == "question" else "translated",
                "query_group": lang_group(
                    r.lang if version == "question" else r.x_lang
                ),
            }
            for name, parts in COMBINATIONS.items():
                lists = [ranking[p] for p in parts if ranking[p]]
                row[name] = r.id in (rrf(*lists) if lists else [])[:10]
            rows.append(row)
        print(f"done: {version}", flush=True)
    t = pd.DataFrame(rows)
    hit = {name: (name, wilson) for name in COMBINATIONS}
    print(
        f"\n{chunk_size()}-token chunks; {(df.kind == 'corpus').sum()} easy + {(df.kind == 'challenge').sum()} hard; hit@10 [95% interval]; RRF\n"
    )
    print(t.groupby(["kind", "version"]).agg(n=("dense", "size"), **hit).to_markdown())
    print("\nBy query language group:\n")
    print(
        t.groupby(["kind", "version", "query_group"])
        .agg(n=("dense", "size"), **hit)
        .to_markdown()
    )


if __name__ == "__main__":
    main()
