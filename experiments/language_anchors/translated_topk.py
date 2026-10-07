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

from experiments.language_anchors.shared_tokens import anchors
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


def sample(n: int) -> pd.DataFrame:
    rag = ROOT / "qa" / "wiki_qa_rag.parquet"
    return duckdb.sql(
        f"""select id, kind, lang, x_lang, question, question_x from '{rag}'
        where answer_ok and x_ok and split = 'dev'
        qualify row_number() over (partition by kind order by hash(id || n || {SEED})) <= {int(n)}"""  # nosec B608 - constants and an int
    ).df()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=int, default=300, help="questions per kind (easy, hard)"
    )
    args = ap.parse_args()

    df = sample(args.n)
    df["anchors_only"] = [
        " ".join(t for ts in anchors(q, x).values() for t in ts)
        for q, x in zip(df.question, df.question_x, strict=True)
    ]
    corpus_share = (
        duckdb.sql(
            f"select in_language lang, count(*) / sum(count(*)) over () as corpus_share from '{ROOT / 'wikipages.parquet'}' group by 1"
        )  # nosec B608 - constant path
        .df()
        .set_index("lang")["corpus_share"]
    )
    bm25 = SparseRetriever()
    rows = []
    for query in ("question", "question_x", "anchors_only"):
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
                }
            )
    t = pd.DataFrame(rows)
    print(
        f"\nPipeline BM25, {chunk_size()}-token chunks, {args.n} easy + {args.n} hard questions\n"
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
    print("\nAnchors only, questions that have anchors:\n")
    print(
        t[(t["query"] == "anchors_only") & t.has_anchors]
        .groupby("kind")[["hit10", "found_top100", "top10_page_lang"]]
        .mean()
        .round(3)
        .to_markdown()
    )


if __name__ == "__main__":
    main()
