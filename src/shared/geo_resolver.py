"""Turn a question into a `GeoScope`: a model names the place, a gazetteer places it.

Two halves on purpose. `Gazetteer` is deterministic and needs no model: corpus
page locations first (a name the corpus already knows resolves without a
network call), then NUTS region names, then Wikidata, then Nominatim. The LLM
step in `LlmGazetteerResolver` only decides *which* place a question is
anchored to and how wide the question is; it never produces coordinates. Both
are cached per normalised query in `geo_scope_cache`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from src.db.pages import cached_geo_scope, primary_locations, store_geo_scope
from src.shared.geo_scope import GeoScope
from src.shared.geocode import Coordinates, GeocodeProvider, name_matches
from src.shared.llm import StructuredLlm
from src.shared.nuts import normalize_place_name, nuts_index, nuts_parents
from src.shared.wikidata import WikidataClient

__all__ = [
    "QUERY_LOCATION_PROMPT",
    "ExtractedQueryLocation",
    "Gazetteer",
    "LazyResolver",
    "LlmGazetteerResolver",
    "ScopeResolver",
    "normalize_query",
    "scope_for_point",
]

Granularity = Literal["point", "municipality", "region", "country", "none"]


class ExtractedQueryLocation(BaseModel):
    """What place, if any, a question is anchored to."""

    place: str | None = Field(
        default=None,
        description=(
            "The one real-world place the question is asking about or around: a "
            "town, castle, river, region or country, written so a gazetteer finds "
            "it (e.g. 'Brestanica, Slovenia', 'Posavska', 'Štajerska'). Null when "
            "the question has no geographic anchor, including questions whose "
            "answer is a place ('Where was X born?')."
        ),
    )
    granularity: Granularity = Field(
        default="none",
        description=(
            "How wide the place is: point (a building, site or village), "
            "municipality (a town and its surroundings), region, country, or none."
        ),
    )
    radius_km: float | None = Field(
        default=None,
        description=(
            "Only when the question says near/around/close to a place: the radius "
            "in kilometres that fits the wording. Otherwise null."
        ),
    )
    reason: str = Field(default="", description="One short sentence.")


class ScopeResolver(Protocol):
    def resolve(self, query: str) -> GeoScope | None: ...


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").casefold()).strip()


def scope_for_point(
    coordinates: Coordinates,
    *,
    label: str | None,
    radius_km: float | None,
    country_code: str | None = None,
    include_null: bool = False,
) -> GeoScope:
    """A scope around a point, with its NUTS codes read off the boundaries."""
    nuts3 = nuts_index().locate(coordinates.latitude, coordinates.longitude)
    parents = nuts_parents(nuts3) if nuts3 else {}
    return GeoScope(
        country_code=parents.get("country_code") or country_code,
        nuts2=parents.get("nuts2"),
        nuts3=nuts3,
        latitude=coordinates.latitude,
        longitude=coordinates.longitude,
        radius_km=radius_km,
        label=label,
        include_null=include_null,
    )


def _scope_for_code(code: str, *, label: str | None, include_null: bool) -> GeoScope:
    parents = nuts_parents(code)
    region = nuts_index().regions.get(code)
    return GeoScope(
        country_code=parents["country_code"],
        nuts2=parents["nuts2"],
        nuts3=parents["nuts3"],
        label=label or (region.name if region else code),
        include_null=include_null,
    )


@dataclass
class Gazetteer:
    """Place name -> scope, cheapest source first, every hit verified by name."""

    geocoder: GeocodeProvider | None = None
    wikidata: WikidataClient | None = None
    default_radius_km: float = 25.0
    country_hint: str | None = None
    include_null: bool = False
    _corpus: dict[str, GeoScope] | None = field(default=None, init=False, repr=False)

    def lookup(
        self,
        place: str,
        granularity: str = "point",
        radius_km: float | None = None,
    ) -> GeoScope | None:
        place = (place or "").strip()
        if not place:
            return None
        radius = self._radius(granularity, radius_km)

        # Region or country wording: NUTS names are the authority on codes.
        if granularity in {"region", "country"}:
            code = self._nuts_code(place, granularity)
            if code:
                return _scope_for_code(
                    code, label=place, include_null=self.include_null
                )

        corpus = self._corpus_scopes().get(normalize_place_name(place))
        if corpus is not None:
            return self._with_radius(corpus, radius)

        point = self._wikidata_point(place) or self._nominatim_point(place)
        if point is None:
            code = self._nuts_code(place, "region")
            return (
                _scope_for_code(code, label=place, include_null=self.include_null)
                if code
                else None
            )
        coordinates, country = point
        scope = scope_for_point(
            coordinates,
            label=place,
            radius_km=radius,
            country_code=country,
            include_null=self.include_null,
        )
        if granularity == "region":
            # A region that matched no NUTS name is wider than the NUTS-3 unit
            # its point happens to fall in; keep the NUTS-2 it implies.
            return replace(scope, nuts3=None, radius_km=None)
        if granularity == "country":
            # A country outside the NUTS index (or hidden by the country hint)
            # resolved to its centroid; the scope is the whole country, not the
            # NUTS-3 unit the centroid falls in.
            return replace(scope, nuts3=None, nuts2=None, radius_km=None)
        return scope

    def _radius(self, granularity: str, radius_km: float | None) -> float | None:
        if radius_km:
            return float(radius_km)
        if granularity in {"point", "municipality"}:
            return self.default_radius_km
        return None

    def _with_radius(self, scope: GeoScope, radius: float | None) -> GeoScope:
        if scope.coordinates is None:
            return scope
        return GeoScope(
            **{**scope.__dict__, "radius_km": radius, "include_null": self.include_null}
        )

    def _nuts_code(self, place: str, granularity: str) -> str | None:
        codes = nuts_index().find_by_name(place, self.country_hint)
        if not codes:
            return None
        if granularity == "country":
            countries = [code for code in codes if len(code) == 2]
            return countries[0] if countries else None
        return codes[0]

    def _corpus_scopes(self) -> dict[str, GeoScope]:
        if self._corpus is None:
            scopes = {}
            try:
                locations = primary_locations()
            except Exception:  # no pages database in this process
                locations = {}
            for location in locations.values():
                if location.coordinates is None or not location.name:
                    continue
                scopes[normalize_place_name(location.name)] = GeoScope(
                    country_code=location.country_code,
                    nuts2=location.nuts2,
                    nuts3=location.nuts3,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    label=location.name,
                )
            self._corpus = scopes
        return self._corpus

    def _wikidata_point(self, place: str) -> tuple[Coordinates, str | None] | None:
        if self.wikidata is None:
            return None
        try:
            for language in ("sl", "en"):
                qids = self.wikidata.search(place, language=language)
                if not qids:
                    continue
                for entity in self.wikidata.entities(qids).values():
                    if entity.is_human or entity.coordinates is None:
                        continue
                    if entity.label and not name_matches(place, entity.label):
                        continue
                    return entity.coordinates, None
        except Exception as exc:  # network: fall through to the next source
            print(f"wikidata lookup failed for {place!r}: {exc}", flush=True)
        return None

    def _nominatim_point(self, place: str) -> tuple[Coordinates, str | None] | None:
        if self.geocoder is None:
            return None
        hit = self.geocoder.geocode(place)
        if hit is None or not name_matches(place, hit.display_name):
            return None
        return hit.coordinates, hit.country_code


QUERY_LOCATION_PROMPT = (
    "You extract the geographic scope of a question for a tourism search over "
    "European places, so results can be narrowed to one area.\n\n"
    "Return `place` only when the question is anchored to a specific place: it "
    "asks what is in, near or around it, or about that place itself. A question "
    "whose *answer* is a place, or that is about a person, an event or a concept, "
    "has no anchor: return place null and granularity none. Never guess a place "
    "that the question does not name or clearly imply.\n"
    "Write the place so a gazetteer finds it, with the country when it helps. "
    "Write a region or a country by its official local name (e.g. 'Posavska', "
    "'Vzhodna Slovenija', 'Štajerska', 'Slovenija'), not a translation.\n"
    "Set radius_km only for near/around wording.\n\n"
    "question: {query}"
)


@dataclass
class LlmGazetteerResolver:
    llm: StructuredLlm
    gazetteer: Gazetteer
    retries: int = 3
    use_cache: bool = True

    def resolve(self, query: str) -> GeoScope | None:
        key = normalize_query(query)
        if not key:
            return None
        if self.use_cache:
            cached = _read_cache(key)
            if cached is not None:
                return cached or None
        scope, settled = self._resolve_uncached(query)
        if self.use_cache and settled:
            _write_cache(key, scope)
        return scope

    def _resolve_uncached(self, query: str) -> tuple[GeoScope | None, bool]:
        """(scope, settled): only a settled answer may be cached.

        A failed extraction or a place the gazetteer could not find is not
        settled: Nominatim answering 429 or the LLM timing out must not turn
        into a permanent "no scope" for that question.
        """
        try:
            extracted = self.llm.structured_output(
                QUERY_LOCATION_PROMPT.format(query=query),
                ExtractedQueryLocation,
                retries=self.retries,
            )
        except RuntimeError as exc:
            print(f"geo resolver: extraction failed, no scope: {exc}", flush=True)
            return None, False
        if not extracted.place or extracted.granularity == "none":
            return None, True
        scope = self.gazetteer.lookup(
            extracted.place, extracted.granularity, extracted.radius_km
        )
        if scope is None:
            print(
                f"geo resolver: {extracted.place!r} not found by the gazetteer, "
                "no scope (not cached)",
                flush=True,
            )
        return scope, scope is not None


@dataclass
class LazyResolver:
    """Builds the real resolver on first use, so catalog listing needs no LLM env."""

    factory: object  # Callable[[], ScopeResolver]
    _inner: ScopeResolver | None = field(default=None, init=False, repr=False)

    def resolve(self, query: str) -> GeoScope | None:
        if self._inner is None:
            self._inner = self.factory()
        return self._inner.resolve(query)


def _read_cache(key: str) -> GeoScope | str | None:
    """None for a miss, "" for a cached 'no scope', else the scope."""
    try:
        raw = cached_geo_scope(key)
    except Exception:
        return None
    if raw is None:
        return None
    return "" if raw == "null" else GeoScope.from_json(raw)


def _write_cache(key: str, scope: GeoScope | None) -> None:
    try:
        store_geo_scope(key, "null" if scope is None else scope.to_json())
    except Exception as exc:
        print(f"geo resolver: cache write failed: {exc}", flush=True)
