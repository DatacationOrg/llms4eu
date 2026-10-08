"""Step 3: remove the anchors from the full question. Does BM25 still find the page without its numbers and names?

    uv run python -m experiments.language_anchors.removal --n 300

Same sample as step 2 (dev, answer_ok, x_ok; n easy + n hard). Anchors are the tokens a question shares with its
translation, by kind (shared_tokens.py); a variant deletes them from the original and from the translated question:
`-numbers`, `-names`, `-both`. Deleting words breaks sentences, which does not matter for BM25 (bag of words); dense
needs placeholders instead (later). A variant is scored only on questions that lose something. Questions may now fit
several places, so hits are counted on the gold page and on gold + `relevant` (where judged). DuckDB BM25 (fts.py).

Step 4, dense: `--engine dense --placeholder "…"` (Qwen3-Embedding-0.6B on the same chunks; removed words become a
placeholder so the sentence stays intact; GPU for the query embeddings). Run BM25 with the same placeholder to compare.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.language_anchors.fts import FtsRetriever
from experiments.language_anchors.shared_tokens import WORD, anchors
from experiments.language_anchors.translated_topk import (
    DEPTH,
    lang_group,
    sample,
    top_pages,
    wilson,
)
from src.db.dataset import chunk_size
from src.retrieval.retrievers.vector import VectorChunkRetriever

VARIANTS = {
    "full": (),
    "-numbers": ("number",),
    "-names": ("name",),
    "-both": ("number", "name"),
}


def without(text: str, tokens: set[str], placeholder: str = "") -> str:
    """The text with every word whose casefold is in `tokens` replaced by `placeholder` (deleted by default).
    A placeholder keeps the sentence intact for dense retrieval; for BM25 it changes nothing (not a word)."""
    return re.sub(
        r"\s+",
        " ",
        WORD.sub(
            lambda m: placeholder if m.group().casefold() in tokens else m.group(), text
        ),
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=int, default=300, help="questions per kind (easy, hard)"
    )
    # step 4: dense (Qwen3-Embedding-0.6B on the same chunks); removed words become "…" so sentences stay intact
    ap.add_argument(
        "--per-lang",
        type=int,
        default=0,
        help="stratified: questions per kind and language (overrides --n)",
    )
    ap.add_argument(
        "--kind",
        choices=["challenge", "corpus"],
        help="only hard (challenge) or easy (corpus) questions",
    )
    ap.add_argument("--engine", choices=["duckdb", "dense"], default="duckdb")
    ap.add_argument("--placeholder", default="", help='e.g. "…"; default: delete')
    args = ap.parse_args()
    # the query-embedding cache lives under LLMS4EU_DATA, which is read-only on the shared disk: keep ours local
    os.environ.setdefault(
        "LLMS4EU_DATA", str(Path(__file__).parent / "out" / "artifacts")
    )

    df = sample(args.n, args.per_lang)
    if args.kind:
        df = df[df.kind == args.kind].reset_index(drop=True)
    shared = [anchors(q, x) for q, x in zip(df.question, df.question_x, strict=True)]
    retriever = (
        FtsRetriever(chunk_size())
        if args.engine == "duckdb"
        else VectorChunkRetriever(name="qwen", provider="qwen")
    )
    rows = []
    for version in ("question", "question_x"):
        for variant, kinds in VARIANTS.items():
            removed = [{t for k in kinds for t in a[k]} for a in shared]
            texts = [
                without(t, r, args.placeholder) if r else t
                for t, r in zip(df[version], removed, strict=True)
            ]
            found = retriever.retrieve_batch(texts, DEPTH)
            for i, r in enumerate(df.itertuples()):
                if variant != "full" and not removed[i]:
                    continue  # nothing of this kind to remove
                top10 = top_pages([c.id for c in found[i]], 10)
                fitting = {r.id} | (
                    set(r.relevant)
                    if isinstance(r.relevant, (list, np.ndarray))
                    else set()
                )
                rows.append(
                    {
                        "qid": i,
                        "kind": r.kind,
                        "lang_group": lang_group(r.lang),
                        "version": "original"
                        if version == "question"
                        else "translated",
                        "variant": variant,
                        "hit10_gold": r.id in top10,
                        "hit10_any_fitting": bool(fitting & set(top10)),
                    }
                )
    t = pd.DataFrame(rows)
    print(
        f"\n{args.engine} (placeholder {args.placeholder!r}), {chunk_size()}-token chunks; {args.n} easy + {args.n} hard; variants only on questions that lose something\n"
    )
    # paired: each variant next to the full question on exactly the same questions
    full = t[t.variant == "full"].set_index(["qid", "version"])["hit10_gold"]
    t["full_same_questions"] = [
        full[(q, v)] for q, v in zip(t.qid, t.version, strict=True)
    ]
    table = t.groupby(["kind", "version", "variant"], sort=False).agg(
        n=("hit10_gold", "size"),
        full_same_questions=("full_same_questions", "mean"),
        hit10_gold=("hit10_gold", "mean"),
        hit10_any_fitting=("hit10_any_fitting", "mean"),
    )
    print(table.round(3).to_markdown())
    # the key comparison per page-language group, with 95% intervals: full vs anchors removed, on the same questions
    print(
        "\nOriginal language, by kind and page-language group: full vs without, hit@10 [95% interval]:\n"
    )
    orig = t[t.version == "original"]
    print(
        orig.groupby(["kind", "lang_group", "variant"], sort=False)
        .agg(
            n=("hit10_gold", "size"),
            full_same_questions=("full_same_questions", wilson),
            without=("hit10_gold", wilson),
        )
        .to_markdown()
    )
    print("\nTranslated, by kind:\n")
    tr = t[t.version == "translated"]
    print(
        tr.groupby(["kind", "variant"], sort=False)
        .agg(
            n=("hit10_gold", "size"),
            full_same_questions=("full_same_questions", wilson),
            without=("hit10_gold", wilson),
        )
        .to_markdown()
    )
    print("\nExamples (hard, original, -both):\n")
    for r, a in list(zip(df.itertuples(), shared, strict=True))[:300]:
        if r.kind == "challenge" and (a["number"] or a["name"]):
            print(
                f"- {r.question}\n  -> {without(r.question, set(a['number']) | set(a['name']), args.placeholder)}"
            )
            break


if __name__ == "__main__":
    main()
