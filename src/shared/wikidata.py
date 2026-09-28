"""The few Wikidata lookups geo and the wiki places list need."""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "llms4eu-tourism-rag (Datacation research)"}
HUMAN = "Q5"


def _get(url: str, timeout: float = 30, **params: object) -> dict:
    for _ in range(5):
        response = httpx.get(
            url, params={**params, "format": "json"}, headers=HEADERS, timeout=timeout
        )
        if response.status_code != 429:
            break
        retry_after = response.headers.get("retry-after", "")
        time.sleep(int(retry_after) if retry_after.isdigit() else 10)
    response.raise_for_status()
    return response.json()


def qid_for_title(language: str, title: str) -> str | None:
    """Wikidata item of a Wikipedia article, following wiki redirects."""
    # ponytail: one request per title; batch 50 titles per call if corpora grow.
    pages = _get(
        f"https://{language}.wikipedia.org/w/api.php",
        action="query",
        titles=title,
        redirects=1,
        prop="pageprops",
        ppprop="wikibase_item",
    )["query"].get("pages", {})
    return next(
        (p["pageprops"]["wikibase_item"] for p in pages.values() if "pageprops" in p),
        None,
    )


def coordinates(qids: list[str]) -> dict[str, tuple[float, float]]:
    """Coordinates (P625) of each item that has them and is not a person."""
    found: dict[str, tuple[float, float]] = {}
    for qid, entity in _entities(qids, "claims"):
        claims = entity.get("claims", {})
        kinds = {_value(c).get("id") for c in claims.get("P31", [])}
        points = [_value(c) for c in claims.get("P625", [])]
        # ponytail: "has coordinates and is not a person" stands in for a
        # place-class list; a language item with coordinates slips through.
        if points and points[0] and HUMAN not in kinds:
            found[qid] = (points[0]["latitude"], points[0]["longitude"])
    return found


def sparql(query: str) -> list[dict]:
    """Result bindings of a query; the service gives up after 60 s."""
    return _get(SPARQL, timeout=70, query=query)["results"]["bindings"]


def search(name: str, language: str) -> list[tuple[str, str]]:
    """(qid, label) of the items whose label or alias matches `name`."""
    hits = _get(
        WIKIDATA_API,
        action="wbsearchentities",
        search=name,
        language=language,
        uselang=language,
        type="item",
        limit=5,
    ).get("search", [])
    return [(hit["id"], hit.get("label", "")) for hit in hits]


def _entities(qids: list[str], props: str) -> Iterator[tuple[str, dict]]:
    """(qid, entity) for each item, fetched 50 per request."""
    unique = list(dict.fromkeys(qids))
    for start in range(0, len(unique), 50):
        entities = _get(
            WIKIDATA_API,
            action="wbgetentities",
            ids="|".join(unique[start : start + 50]),
            props=props,
        )["entities"]
        yield from entities.items()
        time.sleep(0.5)


def _value(claim: dict) -> dict:
    return claim["mainsnak"].get("datavalue", {}).get("value") or {}
