"""Not used by the pipeline: how the wiki pages (`pages.jsonl`) were collected.

EU nature and heritage places on Wikipedia, each in its own country's language:

1. `collect` asks Wikidata for every place of the configured types and countries
   (one SPARQL query per type and country, each answer saved in `wikidata/`) and
   writes one seed row per place, its article in the local language, to
   `urls.jsonl`. Places without one are left out.
2. Fetch them with the scraper:
   `uv run python -m src.scraping.fetch_pages $DATASET_DIR/urls.jsonl --out $DATASET_DIR/fetched.jsonl`
3. `status` shows how far the fetch got; `export` writes the fetched pages with
   their metadata to `pages.jsonl`, which `src.preprocess.chunker` turns into Parquet.

All files live in the dataset folder (`DATASET_DIR`, default `/data/llms4eu/wiki`).

    uv run python -m datagen.corpus.wiki_places collect|status|export
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.db.dataset import ROOT
from src.shared import wikidata
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

POINT = re.compile(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)")
PLACE_QUERY = """
SELECT ?item ?coord ?sitelinks ?article ?title ?lang WHERE {
  VALUES ?wiki { %s }
  ?item wdt:P31 wd:%s; wdt:P17 wd:%s; wdt:P625 ?coord; wikibase:sitelinks ?sitelinks.
  ?article schema:about ?item; schema:isPartOf ?wiki;
           schema:name ?title; schema:inLanguage ?lang.
}
"""


def add_rows(
    rows: dict[str, dict],
    bindings: list[dict],
    category: str,
    country: str,
    country_languages: list[str],
) -> None:
    """One row per place, in its country's language; a place of several types
    keeps them all."""
    # ponytail: a bilingual country (BE, FI, LU, …) takes its first listed
    # language that has an article; compare article lengths if that skews.
    for b in bindings:
        qid, lang = _qid(b["item"]), b["lang"]["value"]
        row = rows.get(qid)
        if row:
            if category not in row["categories"]:
                row["categories"].append(category)
            if row["country"] != country or country_languages.index(
                lang
            ) >= country_languages.index(row["language"]):
                continue
        elif lang not in country_languages:
            continue
        point = POINT.search(b["coord"]["value"])
        if not point:  # an "unknown value" coordinate
            continue
        lon, lat = point.groups()
        rows[qid] = {
            "source": f"wikipedia_{lang}",
            "url": b["article"]["value"],
            "language": lang,
            "qid": qid,
            "title": b["title"]["value"],
            "country": country,
            "country_languages": country_languages,
            "latitude": float(lat),
            "longitude": float(lon),
            "sitelinks": int(b["sitelinks"]["value"]),
            "categories": row["categories"] if row else [category],
        }


def collect() -> None:
    rows: dict[str, dict] = {}
    skipped = []
    for country_qid, country in CONFIG["countries"].items():
        iso, languages = country["iso"], country["languages"]
        wikis = " ".join(f"<https://{lang}.wikipedia.org/>" for lang in languages)
        for category, type_qid in CONFIG["types"].items():
            bindings = _cached_query(
                ROOT / "wikidata" / f"{iso}_{category}.json",
                PLACE_QUERY % (wikis, type_qid, country_qid),
            )
            if bindings is None:
                skipped.append(f"{category}/{iso}")
            else:
                add_rows(rows, bindings, category, iso, languages)
            print(f"{iso} {category}: {len(rows)} places so far", flush=True)

    # Shuffled, so the fetch workers spread over all wikis instead of queueing
    # on one wiki's per-domain delay.
    shuffled = list(rows.values())
    random.Random(0).shuffle(shuffled)  # nosec B311 - spreads fetch order, not security
    path = ROOT / "urls.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for row in shuffled:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} urls to {path}; skipped: {skipped or 'none'}")


def fetched() -> dict[str, dict]:
    """The scraper's last row per URL (a rerun appends)."""
    with (ROOT / "fetched.jsonl").open(encoding="utf-8") as fh:
        return {row["url"]: row for row in map(json.loads, fh)}


def ok(page: dict) -> bool:
    return page["error"] is None and bool(page["text"])


def status() -> None:
    total = sum(1 for _ in (ROOT / "urls.jsonl").open(encoding="utf-8"))
    pages = fetched().values()
    print(f"{len(pages)} of {total} urls fetched ({len(pages) / total:.1%})\n")
    done = Counter(page["source"] for page in pages)
    good = Counter(page["source"] for page in pages if ok(page))
    for source, count in done.most_common():
        print(f"{source:15} ok {good[source]:6}  failed {count - good[source]:5}")


def export() -> None:
    with (ROOT / "urls.jsonl").open(encoding="utf-8") as fh:
        seeds = {row["url"]: row for row in map(json.loads, fh)}
    path = ROOT / "pages.jsonl"
    written = 0
    with path.open("w", encoding="utf-8") as out:
        for url, page in fetched().items():
            if url in seeds and ok(page):
                out.write(
                    json.dumps(page_row(seeds[url], page), ensure_ascii=False) + "\n"
                )
                written += 1
    print(f"wrote {written} pages to {path}")


def page_row(seed: dict, page) -> dict:
    """The ilsp/llms4eu_synthetic_qa field names, plus place metadata."""
    lang, text = seed["language"], page["text"]
    return {
        "id": f"{lang}wiki/{seed['qid']}",
        "wikiname": f"{lang}wiki",
        "wikidata_id": seed["qid"],
        "title": seed["title"],
        "url": seed["url"],
        "final_url": page["final_url"],
        "in_language": lang,
        "language": page["language"],
        "language_ok": language_ok(lang, page["language"], page["final_url"]),
        "country": seed["country"],
        "country_languages": seed["country_languages"],
        "latitude": seed["latitude"],
        "longitude": seed["longitude"],
        "sitelinks": seed["sitelinks"],
        "categories": seed["categories"],
        "machine_generated": any(m in text for m in CONFIG["machine_markers"]),
        "char_count": len(text),
        "word_count": len(text.split()),
        "fetched_at": page["fetched_at"],
        "text": text,
    }


def language_ok(target: str, detected: str | None, final_url: str | None) -> bool:
    """The page declares the target language and was served from its wiki."""
    host = urlparse(final_url or "").netloc
    return detected == target and host == f"{target}.wikipedia.org"


def _cached_query(path: Path, query: str) -> list[dict] | None:
    """Each answer is saved as it arrives, so a re-run only asks what is missing."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    bindings = _sparql_with_retry(query)
    if bindings is not None:
        path.write_text(json.dumps(bindings), encoding="utf-8")
    return bindings


def _sparql_with_retry(query: str) -> list[dict] | None:
    for attempt in range(2):
        time.sleep(1)
        try:
            return wikidata.sparql(query)
        except (httpx.HTTPError, ValueError) as exc:
            print(f"query failed ({type(exc).__name__}), attempt {attempt + 1}")
            time.sleep(10)
    return None


def _qid(binding: dict) -> str:
    return binding["value"].rsplit("/", 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "status", "export"])
    command = parser.parse_args().command
    {"collect": collect, "status": status, "export": export}[command]()


if __name__ == "__main__":
    main()
