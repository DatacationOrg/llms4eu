"""EU nature and heritage places on Wikipedia, each in its own country's language.

`collect` asks Wikidata for every place of the configured types and countries
and writes one seed row per place: its article in the local language
(`urls.jsonl`, the input of `fetch_pages`). Places without one are left out.
`status` shows how far a background fetch got; `export` writes the fetched
pages with their metadata to `pages.jsonl`. All paths live under
`LLMS4EU_DATA`, which the just recipes point at `/data/llms4eu/wiki`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.db.pages import connect_pages
from src.scraping.page_store import initialize_raw_pages_db
from src.shared import wikidata
from src.shared.env import data_path, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))["wiki_places"]

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
                data_path("wikidata", f"{iso}_{category}.json"),
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
    random.Random(0).shuffle(shuffled)
    path = data_path("urls.jsonl")
    with path.open("w", encoding="utf-8") as out:
        for row in shuffled:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} urls to {path}; skipped: {skipped or 'none'}")


def status() -> None:
    total = sum(1 for _ in data_path("urls.jsonl").open(encoding="utf-8"))
    initialize_raw_pages_db()
    with connect_pages() as conn:
        rows = conn.execute(
            "select source, count(*) as done,"
            " sum(error is null and markdown_chars > 0) as ok"
            " from page_metadata group by source order by done desc"
        ).fetchall()
    done = sum(row["done"] for row in rows)
    print(f"{done} of {total} urls fetched ({done / total:.1%})\n")
    for row in rows:
        failed = row["done"] - row["ok"]
        print(f"{row['source']:15} ok {row['ok']:6}  failed {failed:5}")
    log = data_path("fetch.log")
    if log.exists():
        print("\n" + "".join(deque(log.open(encoding="utf-8"), maxlen=3)), end="")


def export() -> None:
    seeds = {}
    for line in data_path("urls.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        seeds[row["url"]] = row

    path = data_path("pages.jsonl")
    written = 0
    with connect_pages() as conn, path.open("w", encoding="utf-8") as out:
        pages = conn.execute(
            "select m.url, m.final_url, m.language, m.fetched_at, c.markdown"
            " from page_metadata m join page_markdown_content c on c.page_id = m.id"
            " where m.error is null"
        )
        for page in pages:
            seed = seeds.get(page["url"])
            if seed:
                out.write(json.dumps(page_row(seed, page), ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} pages to {path}")


def page_row(seed: dict, page) -> dict:
    """The ilsp/llms4eu_synthetic_qa field names, plus place metadata."""
    lang, text = seed["language"], page["markdown"]
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
