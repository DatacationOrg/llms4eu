"""Geography as a soft re-rank: near pages move up, unknown pages stay put.

The stage retrieves `limit * overfetch` chunks unfiltered; each score becomes
`(1 - w) * text + w * exp(-km / decay)`, text min-max normalised over the list.
A page with no location scores 1.0 on geography: the 2026-09-08 run lost 15 of
72 scoped questions to a hard filter, every one a gold page with no location.
The model only names the place; Wikidata supplies the coordinates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field

from src.db.pages import connect_pages as connect
from src.retrieval.base import RankedChunk, Retriever
from src.shared import wikidata
from src.shared.env import load_yaml
from src.shared.llm import LocalOllamaStructuredLlm
from src.shared.prompts import render

CONFIG = load_yaml(Path(__file__).parents[1] / "config.yaml")


class Place(NamedTuple):
    latitude: float
    longitude: float
    decay_km: float


class QueryPlace(BaseModel):
    place: str | None = Field(None, description="Base form, or null.")
    language: str = Field(description="ISO 639-1 code of the question.")
    granularity: Literal["site", "town", "region", "country"] | None = None


@dataclass(frozen=True)
class GeoRetriever:
    name: str
    stage: Retriever
    place_of: Callable[[str], Place | None]
    page_points: Callable[[], dict[str, tuple[float, float]]]
    weight: float
    overfetch: int

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        return self.retrieve_batch([query], limit)[0]

    def retrieve_batch(
        self, queries: list[str], limit: int
    ) -> dict[int, list[RankedChunk]]:
        places = [self.place_of(query) for query in queries]
        plain = [i for i, place in enumerate(places) if place is None]
        scoped = [i for i, place in enumerate(places) if place is not None]
        results: dict[int, list[RankedChunk]] = {}
        if plain:
            ranked = self.stage.retrieve_batch([queries[i] for i in plain], limit)
            results.update({i: ranked.get(n, []) for n, i in enumerate(plain)})
        if scoped:
            ranked = self.stage.retrieve_batch(
                [queries[i] for i in scoped], limit * self.overfetch
            )
            for n, i in enumerate(scoped):
                chunks = fuse(
                    ranked.get(n, []), places[i], self.page_points(), self.weight
                )
                results[i] = chunks[:limit]
        return results


def fuse(
    chunks: list[RankedChunk],
    place: Place,
    points: dict[str, tuple[float, float]],
    weight: float,
) -> list[RankedChunk]:
    if not chunks:
        return chunks
    low = min(chunk.score for chunk in chunks)
    span = max(chunk.score for chunk in chunks) - low or 1.0

    def geo(chunk: RankedChunk) -> float:
        point = points.get(chunk.id.rsplit(":", 1)[0])
        if point is None:
            return 1.0
        return math.exp(-haversine_km(place[:2], point) / place.decay_km)

    fused = [
        replace(c, score=(1 - weight) * (c.score - low) / span + weight * geo(c))
        for c in chunks
    ]
    return sorted(fused, key=lambda chunk: chunk.score, reverse=True)


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 12742 * math.asin(math.sqrt(h))


@cache
def page_points() -> dict[str, tuple[float, float]]:
    with connect() as conn:
        rows = conn.execute("select page_id, latitude, longitude from page_locations")
        return {row["page_id"]: (row["latitude"], row["longitude"]) for row in rows}


def place_of(query: str) -> Place | None:
    """The query's place, cached; a failed lookup is retried next run, not cached."""
    key = " ".join(query.lower().split())
    with connect() as conn:
        row = conn.execute(
            "select latitude, longitude, decay_km from geo_query_places where query = ?",
            (key,),
        ).fetchone()
    if row is not None:
        return None if row["latitude"] is None else Place(*row)
    try:
        place = _resolve(query)
    except Exception as exc:  # a gazetteer or model outage must not end a run
        print(f"geo lookup failed, unscoped: {exc}", flush=True)
        return None
    with connect() as conn:
        conn.execute(
            "insert or replace into geo_query_places values (?, ?, ?, ?)",
            (key, *(place or (None, None, None))),
        )
    return place


def _resolve(query: str) -> Place | None:
    llm = LocalOllamaStructuredLlm(
        CONFIG["geo_model"], reasoning=False, method=CONFIG["geo_structured_method"]
    )
    named = llm.structured_output(render("geo_place", question=query), QueryPlace)
    # ponytail: regions are points with a wide decay and countries are skipped;
    # add a country-code match when a multi-country eval shows it matters.
    decay = CONFIG["geo_decay_km"].get(named.granularity)
    if not named.place or decay is None:
        return None
    words = _words(named.place)
    for qid, label in wikidata.search(named.place, named.language):
        if words & _words(label):
            point = wikidata.coordinates([qid]).get(qid)
            if point:
                return Place(*point, decay)
    return None


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))
