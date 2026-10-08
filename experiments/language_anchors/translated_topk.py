"""Step 2: why translated hard questions fail. Pipeline BM25 (512-token chunks) on three queries per question:
the original, the translation, and the anchors alone (tokens shared by both, see shared_tokens.py).

    uv run python -m experiments.language_anchors.translated_topk --n 300

Sample: dev, answer_ok, x_ok rows, the first n easy and n hard in a fixed hash order. For each query: the gold
page's rank, and the language of the top-10 pages (a page id starts with its wiki, `svwiki/...`). Expected share
of a language = its share of the corpus. Prints markdown tables; read-only, CPU.
"""

from __future__ import annotations

import argparse

import duckdb
import pandas as pd

from experiments.language_anchors.fts import FtsRetriever
from experiments.language_anchors.shared_tokens import WORD, anchors
from src.db.dataset import ROOT, chunk_size
from src.retrieval.retrievers.sparse import SparseRetriever

SEED = 7
DEPTH = 100


def lang_of(page: str) -> str:
    return page.split("wiki/", 1)[0]


def top_pages(chunk_ids: list[str], k: int) -> list[str]:
    pages: list[str] = []
    for c in chunk_ids:
        page = c.rsplit(":", 2)[0]
        if page not in pages:
            pages.append(page)
        if len(pages) == k:
            break
    return pages


def realistic_anchors(question: str) -> str:
    """What a system could take from the query alone: numbers and capitalised words that are not the first word.
    (German capitalises every noun, so German queries keep more ordinary words.)"""
    words = WORD.findall(question)
    return " ".join(
        w
        for i, w in enumerate(words)
        if any(c.isdigit() for c in w) or (i > 0 and w[:1].isupper())
    )


def sample(n: int, per_lang: int = 0) -> pd.DataFrame:
    """Dev, answer_ok, x_ok rows in a fixed hash order: the first n per kind, or (per_lang > 0) the first per_lang
    per kind and language, so Swedish does not dominate and every language is represented."""
    rag = ROOT / "qa" / "wiki_qa_rag.parquet"
    partition, limit = ("kind, lang", int(per_lang)) if per_lang else ("kind", int(n))
    return duckdb.sql(
        f"""select id, kind, lang, x_lang, question, question_x, relevant from '{rag}'
        where answer_ok and x_ok and split = 'dev'
        qualify row_number() over (partition by {partition} order by hash(id || n || {SEED})) <= {limit}"""  # nosec B608 - constants and ints
    ).df()


def lang_group(lang: str) -> str:
    """Groups that behave differently for BM25: Swedish (lake register), German (nouns capitalised), other script."""
    return {"sv": "sv", "de": "de", "el": "el/bg", "bg": "el/bg"}.get(
        lang, "other Latin"
    )


def wilson(hits: pd.Series) -> str:
    """Share of True with its 95% Wilson interval, e.g. '44% [37-51]'."""
    n, p, z = len(hits), float(hits.mean()), 1.96
    if n == 0:
        return "-"
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{p:.0%} [{max(centre - half, 0):.0%}-{min(centre + half, 1):.0%}]"


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
    # duckdb: the lean on-disk BM25 (fts.py); the pipeline's in-memory build gets killed under memory pressure
    ap.add_argument("--engine", choices=["pipeline", "duckdb"], default="pipeline")
    args = ap.parse_args()

    df = sample(args.n, args.per_lang)
    shared = [anchors(q, x) for q, x in zip(df.question, df.question_x, strict=True)]
    df["anchors_only"] = [" ".join(t for ts in a.values() for t in ts) for a in shared]
    # the anchors split by kind: is the specificity in the numbers or in the names?
    df["numbers_only"] = [" ".join(a["number"]) for a in shared]
    df["names_only"] = [" ".join(a["name"]) for a in shared]
    # step 2b: the realistic version, from the translated question alone (no knowledge of the original)
    df["realistic_x"] = df.question_x.map(realistic_anchors)
    corpus_share = (
        duckdb.sql(
            f"select in_language lang, count(*) / sum(count(*)) over () as corpus_share from '{ROOT / 'wikipages.parquet'}' group by 1"  # nosec B608 - constant path
        )
        .df()
        .set_index("lang")["corpus_share"]
    )
    bm25 = (
        SparseRetriever() if args.engine == "pipeline" else FtsRetriever(chunk_size())
    )
    rows = []
    for query in (
        "question",
        "question_x",
        "anchors_only",
        "numbers_only",
        "names_only",
        "realistic_x",
    ):
        texts = df[query].tolist()
        found = bm25.retrieve_batch([t or "-" for t in texts], DEPTH)
        for i, r in enumerate(df.itertuples()):
            pages = top_pages([c.id for c in found[i]], DEPTH) if texts[i] else []
            top10 = pages[:10]
            qlang = {"question": r.lang, "question_x": r.x_lang}.get(query)
            rows.append(
                {
                    "kind": r.kind,
                    "query": query,
                    "has_anchors": bool(r.anchors_only),
                    "has_query": bool(texts[i]),
                    "hit10": r.id in top10,
                    "found_top100": r.id in pages,
                    "top10_query_lang": sum(lang_of(p) == qlang for p in top10)
                    / max(len(top10), 1)
                    if qlang
                    else None,
                    "top10_page_lang": sum(lang_of(p) == r.lang for p in top10)
                    / max(len(top10), 1),
                    "expected_query_lang": corpus_share.get(qlang, 0.0)
                    if qlang
                    else None,
                    "query_lang_differs": qlang is not None and qlang != r.lang,
                    "x_lang": r.x_lang,
                    "lang_group": lang_group(r.lang),
                }
            )
    t = pd.DataFrame(rows)
    print(
        f"\nBM25 ({args.engine}), {chunk_size()}-token chunks, {(df.kind == 'corpus').sum()} easy + {(df.kind == 'challenge').sum()} hard questions\n"
    )
    print(
        t.groupby(["kind", "query"])[
            [
                "hit10",
                "found_top100",
                "top10_query_lang",
                "expected_query_lang",
                "top10_page_lang",
            ]
        ]
        .mean()
        .round(3)
        .to_markdown()
    )
    print("\nAnchor queries, only questions that have that kind of anchor:\n")
    anchor_queries = t[
        t["query"].isin(["anchors_only", "numbers_only", "names_only", "realistic_x"])
        & t.has_query
    ]
    print(
        anchor_queries.groupby(["kind", "query"])
        .agg(
            n=("hit10", "size"),
            hit10=("hit10", "mean"),
            found_top100=("found_top100", "mean"),
            top10_page_lang=("top10_page_lang", "mean"),
        )
        .round(3)
        .to_markdown()
    )
    print("\nStep 2b by translation language group (hard questions):\n")
    hard = t[
        (t.kind == "challenge")
        & t["query"].isin(["question_x", "anchors_only", "realistic_x"])
    ].copy()
    hard["x_group"] = hard.x_lang.map(
        lambda x: (
            "de (nouns capitalised)"
            if x == "de"
            else "el/bg (other script)"
            if x in ("el", "bg")
            else "other Latin"
        )
    )
    print(
        hard.pivot_table(
            index="x_group", columns="query", values="hit10", aggfunc="mean"
        )
        .round(3)
        .to_markdown()
    )
    print("\nExamples (hard): translated question -> realistic query\n")
    for r in df[df.kind == "challenge"].head(5).itertuples():
        print(
            f"- [{r.x_lang}] {r.question_x}\n  -> {r.realistic_x}\n  oracle anchors: {r.anchors_only}"
        )
    print("\nHard questions by page-language group, hit@10 with 95% interval:\n")
    hard = t[
        (t.kind == "challenge")
        & t["query"].isin(["question", "question_x", "anchors_only", "realistic_x"])
    ]
    print(
        hard.groupby(["lang_group", "query"])["hit10"]
        .agg(n="size", hit10=wilson)
        .unstack("query")
        .to_markdown()
    )


if __name__ == "__main__":
    main()
