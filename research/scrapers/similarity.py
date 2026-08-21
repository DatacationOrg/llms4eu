"""How much do the tools actually differ, tool-to-tool?

The A/B arena answers "which is better". It cannot answer "is there anything to
choose between them at all", and that question comes first: where two tools
return the same thing there is nothing to vote on, and a vote spent there is a
vote not spent somewhere it matters.

Two measurements, deliberately kept apart, because they disagree and the
disagreement is the finding:

* **Word-level cosine** over lowercased word tokens. Blind to markdown syntax,
  whitespace and punctuation, so it answers *did these two select the same
  content*. The fetch layer scores ~100% here.
* **Byte-exact identity** of the markdown. Sensitive to everything, including
  the Scrapling defect where a dropped whitespace-only text node fuses a
  citation marker into the next word. The same fetch-layer pairs score 44-54%.

Cosine alone would have said the four fetchers are interchangeable, which is
comfortable and wrong. Byte-identity alone would have said they are all
different, which is true and useless. Both, side by side, say the real thing:
same words, different bytes, and the bytes are a bug in one of them.

Means are reported on the page set both tools handled *and* on the set every
tool handled, because coverage differs by up to 18 pages (justext returns
nothing on 18) and a mean over a tool's own easy subset flatters it.

    uv run python -m research.scrapers.similarity [out.json]
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from . import store

WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

FETCHERS = ("ours-httpx", "scrapy", "scrapling", "scrapling-stealthy")

# Ordered so the two clusters the matrix reveals sit in adjacent blocks:
# content-model extractors, then the partial over-extractor, then the
# whole-page dumps. The order is a presentation choice; the numbers are not.
EXTRACTORS = (
    "ours",
    # The settings on the branch today, as a separate column: the deck compares tool
    # against tool everywhere else, and this is the one place where a *settings*
    # difference can be read on the same scale -- if `ours@committed` does not sit
    # inside the content-model block, then our own configuration moved us out of the
    # family we chose the tool from.
    "ours@committed",
    "trafilatura",
    "resiliparse",
    "justext",
    "html-text",
    "goose3",
    "readability-lxml",
    "crawl4ai",
    "scrapling-md",
    "markitdown",
    "docling",
)


def tokens(text: str) -> Counter:
    return Counter(WORD.findall(text.lower()))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _summarise(texts: dict[str, dict[str, str]], names: tuple[str, ...]) -> dict:
    """Char means and both similarity matrices for one layer's outputs."""
    vectors = {n: {p: tokens(t) for p, t in texts[n].items()} for n in names}
    common = set.intersection(*(set(texts[n]) for n in names)) if names else set()

    chars = {}
    for n in names:
        own = [len(t) for t in texts[n].values()]
        chars[n] = {
            "pages": len(own),
            "mean_own": round(sum(own) / len(own)) if own else None,
            "mean_common": (
                round(sum(len(texts[n][p]) for p in common) / len(common))
                if common
                else None
            ),
        }

    cos: dict[str, dict] = {}
    exact: dict[str, dict] = {}
    for a, b in itertools.combinations_with_replacement(names, 2):
        both = sorted(texts[a].keys() & texts[b].keys())
        if not both:
            cos[f"{a}|{b}"] = exact[f"{a}|{b}"] = None
            continue
        sims = [cosine(vectors[a][p], vectors[b][p]) for p in both]
        same = sum(1 for p in both if texts[a][p] == texts[b][p])
        cos[f"{a}|{b}"] = {"pct": round(100 * sum(sims) / len(sims), 1), "n": len(both)}
        exact[f"{a}|{b}"] = {"pct": round(100 * same / len(both), 1), "n": len(both)}

    return {
        "names": list(names),
        "common_pages": len(common),
        "chars": chars,
        "cosine": cos,
        "exact": exact,
    }


def fetch_layer() -> dict:
    """One converter over four fetchers, so any difference is the fetcher's.

    Reads the `fetchcmp-*.html` bodies off disk rather than the `snapshots`
    table: the table's sha256 column was only ever filled for ~45 of the 100
    pages, so a DB-driven version of this silently measures a thin, unstated
    slice. Dynamic pages -- the ones where two identical httpx fetches already
    disagree -- are excluded, otherwise page churn is scored as fetcher
    divergence.
    """
    import trafilatura

    report = json.loads(store.fetch_layer_json().read_text())
    by_url = {p["url"]: p["id"] for p in store.load_pages(include_dropped=True)}
    dynamic = {by_url[u] for u in report["dynamic_urls"] if u in by_url}
    every = {p["id"] for p in store.load_pages(include_dropped=True)}

    texts: dict[str, dict[str, str]] = {n: {} for n in FETCHERS}
    for page_id in sorted(every - dynamic):
        for name in FETCHERS:
            path = store.data_dir() / "snapshots" / page_id / f"fetchcmp-{name}.html"
            if not path.exists():
                continue
            markdown = trafilatura.extract(
                path.read_text(encoding="utf-8", errors="replace"),
                output_format="markdown",
                include_tables=True,
                include_links=False,
                favor_recall=True,
            )
            if markdown:
                texts[name][page_id] = markdown

    out = _summarise(texts, FETCHERS)
    out["excluded_dynamic"] = len(dynamic)
    return out


def extractor_layer() -> dict:
    """The recorded arena runs, raw variant, live corpus only."""
    live = {p["id"] for p in store.load_pages()}
    texts = {
        name: {p: t for p in live if (t := store.run_output(p, name))}
        for name in EXTRACTORS
    }
    return _summarise(texts, EXTRACTORS)


def main() -> None:
    data = {"fetchers": fetch_layer(), "extractors": extractor_layer()}
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if destination:
        destination.write_text(json.dumps(data, indent=2), encoding="utf-8")

    for layer in ("fetchers", "extractors"):
        block = data[layer]
        print(f"\n=== {layer} === (all handled {block['common_pages']} pages)")
        print(f"  {'tool':22}{'pages':>7}{'avg own':>10}{'avg common':>12}")
        for name, row in block["chars"].items():
            print(
                f"  {name:22}{row['pages']:>7}{row['mean_own']:>10}"
                f"{row['mean_common']:>12}"
            )
        print(f"  {'pair':46}{'cosine':>8}{'exact':>8}")
        for key, value in block["cosine"].items():
            a, b = key.split("|")
            if a == b or value is None:
                continue
            print(f"  {a:22}{b:22}  {value['pct']:>6.1f}{block['exact'][key]['pct']:>8.1f}")


if __name__ == "__main__":
    main()
