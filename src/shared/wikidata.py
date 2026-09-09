"""Wikidata as the entity anchor for page locations.

A Wikipedia page maps to one Wikidata item, and that item says what the page
is (P31), where it is (P625), and which administrative and NUTS units hold it
(P131, P300, P605). That is bulk, deterministic geotagging for every Wikipedia
source, and the class tells a person or a concept apart from a place without a
model in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.shared.geocode import USER_AGENT, Coordinates

__all__ = ["WikidataClient", "WikidataEntity"]

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Classes that settle granularity without looking at coordinates.
COUNTRY_CLASSES = frozenset({"Q6256", "Q3624078"})  # country, sovereign state
REGION_CLASSES = frozenset(
    {
        "Q82794",  # geographic region
        "Q1620908",  # historical region
        "Q1474320",  # statistical region of Slovenia
        "Q56061",  # administrative territorial entity
        "Q3965305",  # hill range (spans regions, not a point)
    }
)
MUNICIPALITY_CLASSES = frozenset(
    {
        "Q328584",  # municipality of Slovenia
        "Q6960199",  # city municipality of Slovenia
        "Q15284",  # municipality
        "Q2039348",  # municipality of Austria
        "Q484170",  # commune of France
    }
)
HUMAN_CLASS = "Q5"
# Things Wikidata gives coordinates and a country, which are still not places a
# visitor can go: languages (a language item points at its main country) and
# states that no longer exist.
NON_PLACE_CLASSES = frozenset(
    {
        "Q34770",  # language
        "Q1288568",  # modern language
        "Q33742",  # natural language
        "Q3024240",  # historical country
        "Q48349",  # empire
        "Q17272482",  # time zone
    }
)


@dataclass(frozen=True)
class WikidataEntity:
    qid: str
    label: str | None
    coordinates: Coordinates | None
    instance_of: tuple[str, ...]
    country_qid: str | None
    admin_parent_qid: str | None
    nuts_codes: tuple[str, ...]
    iso_3166_2: str | None

    @property
    def is_human(self) -> bool:
        return HUMAN_CLASS in self.instance_of

    @property
    def granularity(self) -> str | None:
        """point | municipality | region | country, or None for a non-place."""
        classes = set(self.instance_of)
        if self.is_human or classes & NON_PLACE_CLASSES:
            return None
        if classes & COUNTRY_CLASSES:
            return "country"
        # A place sits in a country or an administrative unit. A language or a
        # vanished empire can carry coordinates on Wikidata and yet is not
        # somewhere a tourist can go.
        if self.country_qid is None and self.admin_parent_qid is None:
            return None
        if self.nuts_codes:
            # A country item lists its own two-letter code (often beside its
            # NUTS-1 grouping, e.g. SI and SI0); a region never does.
            shortest = min(self.nuts_codes, key=len)
            return "country" if len(shortest) <= 2 else "region"
        if classes & REGION_CLASSES:
            return "region"
        if classes & MUNICIPALITY_CLASSES:
            return "municipality"
        if self.coordinates is not None:
            return "point"
        return None


@dataclass
class WikidataClient:
    user_agent: str = USER_AGENT
    timeout_seconds: float = 30.0
    batch_size: int = 50

    def qids_for_wikipedia_titles(
        self, titles: list[str], language: str = "sl"
    ) -> dict[str, str]:
        """Title -> QID through the Wikipedia API, following redirects.

        `wbgetentities&sites=` does not follow a wiki redirect ('Posavska
        regija' -> 'Posavska statistična regija' came back missing), so the
        wiki itself resolves the title first.
        """
        found: dict[str, str] = {}
        for start in range(0, len(titles), self.batch_size):
            batch = titles[start : start + self.batch_size]
            data = self._get(
                f"https://{language}.wikipedia.org/w/api.php",
                {
                    "action": "query",
                    "titles": "|".join(batch),
                    "redirects": 1,
                    "prop": "pageprops",
                    "ppprop": "wikibase_item",
                    "format": "json",
                },
            ).get("query", {})
            resolved = {}
            for entry in data.get("normalized", []) + data.get("redirects", []):
                resolved[entry["to"]] = entry["from"]
            qid_by_final_title = {
                page.get("title"): (page.get("pageprops") or {}).get("wikibase_item")
                for page in data.get("pages", {}).values()
            }
            for final_title, qid in qid_by_final_title.items():
                if not qid:
                    continue
                original = final_title
                # Walk normalisation and redirect chains back to the asked title.
                while original in resolved and original not in batch:
                    original = resolved[original]
                found[original] = qid
                found.setdefault(final_title, qid)
        return found

    def entities(self, qids: list[str]) -> dict[str, WikidataEntity]:
        out: dict[str, WikidataEntity] = {}
        unique = list(dict.fromkeys(qid for qid in qids if qid))
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            data = self._get(
                WIKIDATA_API,
                {
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "claims|labels",
                    "languages": "en|sl",
                    "format": "json",
                },
            )
            for qid, raw in data.get("entities", {}).items():
                if "missing" in raw:
                    continue
                out[qid] = _entity(qid, raw)
        return out

    def search(self, text: str, language: str = "sl", limit: int = 3) -> list[str]:
        data = self._get(
            WIKIDATA_API,
            {
                "action": "wbsearchentities",
                "search": text,
                "language": language,
                "uselang": language,
                "type": "item",
                "limit": limit,
                "format": "json",
            },
        )
        return [hit["id"] for hit in data.get("search", [])]

    def _get(self, url: str, params: dict) -> dict:
        response = httpx.get(
            url,
            params=params,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


def _entity(qid: str, raw: dict) -> WikidataEntity:
    claims = raw.get("claims", {})
    labels = raw.get("labels", {})
    label = (labels.get("en") or labels.get("sl") or {}).get("value")
    coordinates = None
    for value in _values(claims, "P625"):
        if isinstance(value, dict) and "latitude" in value:
            coordinates = Coordinates(
                float(value["latitude"]), float(value["longitude"])
            )
            break
    return WikidataEntity(
        qid=qid,
        label=label,
        coordinates=coordinates,
        instance_of=tuple(_item_ids(claims, "P31")),
        country_qid=next(iter(_item_ids(claims, "P17")), None),
        admin_parent_qid=next(iter(_item_ids(claims, "P131")), None),
        nuts_codes=tuple(str(v) for v in _values(claims, "P605") if isinstance(v, str)),
        iso_3166_2=next(
            (str(v) for v in _values(claims, "P300") if isinstance(v, str)), None
        ),
    )


def _values(claims: dict, prop: str) -> list:
    return [
        claim["mainsnak"]["datavalue"]["value"]
        for claim in claims.get(prop, [])
        if "datavalue" in claim.get("mainsnak", {})
    ]


def _item_ids(claims: dict, prop: str) -> list[str]:
    return [
        value["id"]
        for value in _values(claims, prop)
        if isinstance(value, dict) and "id" in value
    ]
