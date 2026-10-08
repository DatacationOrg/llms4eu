"""Oracle location filter: how much could knowing where a question is about recover, especially across languages?

    nice -n 10 uv run python -m experiments.location.oracle_filter --per-lang 100

Same stratified sample as language_anchors (dev, answer_ok, x_ok; up to 100 per kind and language). For the
original and the translated question: BM25 (DuckDB, fts.py) on the full question and on its realistic anchors, and
both fused (RRF), each searched
- without a filter;
- within the gold page's Wikipedia language edition ("svwiki/..."), a cheap stand-in for its country (each place's
  article is in its country's language; exceptions: Belgium, Finland's Swedish pages, Ireland);
- within 25 km of the gold page (any kind of place).
Oracle: the filter uses the gold page itself, so this is an upper bound for a location resolver. hit@10 on the gold
page, 95% Wilson intervals. Unfiltered rankings come from and go to the BM25 cache.
"""

from __future__ import annotations

import argparse

import duckdb
import numpy as np
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
from experiments.location.region_density import haversine_km
from src.db.dataset import ROOT, chunk_size

RADIUS_KM = 25


def within(gold: list[str], radius: float) -> list[list[str]]:
    """Per gold page, the ids of all pages within `radius` km of it (itself included)."""
    pages = duckdb.sql(
        f"select id, latitude, longitude from '{ROOT / 'wikipages.parquet'}' where latitude is not null"  # nosec B608 - constant path
    ).df()
    ids, lats, lons = (
        pages.id.to_numpy(),
        pages.latitude.to_numpy(),
        pages.longitude.to_numpy(),
    )
    where = dict(zip(pages.id, zip(lats, lons, strict=True), strict=True))
    out = []
    for g in gold:
        lat, lon = where[g]
        out.append(list(ids[haversine_km(lat, lon, lats, lons) <= radius]))
    return out


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
    near = within(df.id.tolist(), RADIUS_KM)
    edition = [g.split("/", 1)[0] + "/" for g in df.id]
    print(
        f"pages within {RADIUS_KM} km of the gold page: median {np.median([len(n) for n in near]):.0f}",
        flush=True,
    )
    bm25 = FtsRetriever(chunk_size())
    filters = {
        "none": {},
        "language edition": {"prefixes": edition},
        f"{RADIUS_KM} km": {"allowed": near},
    }
    rows = []
    for version in ("question", "question_x"):
        texts = {"full": df[version].tolist()}
        texts["anchors"] = [realistic_anchors(t) or "-" for t in texts["full"]]
        for fname, kw in filters.items():
            ranked = {q: bm25.retrieve_batch(texts[q], DEPTH, **kw) for q in texts}
            for i, r in enumerate(df.itertuples()):
                full = top_pages([c.id for c in ranked["full"][i]], DEPTH)
                anchors = (
                    top_pages([c.id for c in ranked["anchors"][i]], DEPTH)
                    if texts["anchors"][i] != "-"
                    else []
                )
                fused = rrf(full, anchors) if anchors else full
                rows.append(
                    {
                        "kind": r.kind,
                        "version": "original"
                        if version == "question"
                        else "translated",
                        "filter": fname,
                        "page_group": lang_group(r.lang),
                        "full": r.id in full[:10],
                        "anchors": r.id in anchors[:10],
                        "fused": r.id in fused[:10],
                    }
                )
            print(f"done: {version}, {fname}", flush=True)
    t = pd.DataFrame(rows)
    hit = {m: (m, wilson) for m in ("full", "anchors", "fused")}
    print(
        f"\nBM25 (duckdb), {chunk_size()}-token chunks; hit@10 [95% interval]; oracle filters\n"
    )
    print(
        t.groupby(["kind", "version", "filter"], sort=False)
        .agg(n=("full", "size"), **hit)
        .to_markdown()
    )
    print("\nHard questions by page-language group:\n")
    hard = t[t.kind == "challenge"]
    print(
        hard.groupby(["version", "filter", "page_group"], sort=False)
        .agg(n=("full", "size"), **hit)
        .to_markdown()
    )


if __name__ == "__main__":
    main()
