"""The few Wikidata lookups geo needs: title to item, item to coordinates, search."""

from __future__ import annotations

from typing import Any
import httpx

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "llms4eu-tourism-rag (Datacation research)"}
HUMAN = "Q5"


def _get(url: str, **params: str | int) -> dict[str, Any]:
    response = httpx.get(
        url, params={**params, "format": "json"}, headers=HEADERS, timeout=30
    )
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
    unique = list(dict.fromkeys(qids))
    for start in range(0, len(unique), 50):
        entities = _get(
            WIKIDATA_API,
            action="wbgetentities",
            ids="|".join(unique[start : start + 50]),
            props="claims",
        )["entities"]
        for qid, entity in entities.items():
            claims = entity.get("claims", {})
            kinds = {_value(c).get("id") for c in claims.get("P31", [])}
            points = [_value(c) for c in claims.get("P625", [])]
            # ponytail: "has coordinates and is not a person" stands in for a
            # place-class list; a language item with coordinates slips through.
            if points and points[0] and HUMAN not in kinds:
                found[qid] = (points[0]["latitude"], points[0]["longitude"])
    return found


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


def _value(claim: dict[str, Any]) -> dict[str, Any]:
    return claim["mainsnak"].get("datavalue", {}).get("value") or {}
