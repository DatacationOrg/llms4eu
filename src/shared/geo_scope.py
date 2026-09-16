"""The geographic scope a question resolves to, and how it narrows retrieval.

A scope is codes plus an optional point. The codes filter (Chroma `where`, a
BM25 page set); the point boosts by distance. Widening drops the most specific
criterion first, so a wrong geocode can narrow a search but never empty it:
radius -> NUTS-3 -> NUTS-2 -> country -> no filter at all.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace

from src.shared.geocode import Coordinates

__all__ = ["GeoScope"]

_KM_PER_DEGREE_LAT = 111.32


@dataclass(frozen=True)
class GeoScope:
    country_code: str | None = None
    nuts2: str | None = None
    nuts3: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    radius_km: float | None = None
    # What was resolved, for logs and prompts: "Rajhenburg Castle (SI036)".
    label: str | None = None
    # Filtered path only: whether pages with no location survive the filter.
    include_null: bool = False

    @property
    def coordinates(self) -> Coordinates | None:
        if self.latitude is None or self.longitude is None:
            return None
        return Coordinates(self.latitude, self.longitude)

    @property
    def filters(self) -> bool:
        """Whether this scope narrows anything (a bare point only boosts)."""
        return bool(
            (self.radius_km and self.coordinates)
            or self.nuts3
            or self.nuts2
            or self.country_code
        )

    @property
    def level(self) -> str:
        if self.radius_km and self.coordinates:
            return "radius"
        if self.nuts3:
            return "nuts3"
        if self.nuts2:
            return "nuts2"
        if self.country_code:
            return "country"
        return "none"

    def widen(self) -> GeoScope | None:
        """One step less specific; None once nothing is left to drop."""
        level = self.level
        if level == "radius":
            return replace(self, radius_km=None)
        if level == "nuts3":
            return replace(self, nuts3=None)
        if level == "nuts2":
            return replace(self, nuts2=None)
        if level == "country":
            return replace(self, country_code=None)
        return None

    def bounding_box(self) -> tuple[float, float, float, float] | None:
        """(lat_min, lat_max, lon_min, lon_max) around the point and radius."""
        if not (self.radius_km and self.coordinates):
            return None
        d_lat = self.radius_km / _KM_PER_DEGREE_LAT
        cos_lat = max(math.cos(math.radians(self.latitude)), 1e-6)
        d_lon = self.radius_km / (_KM_PER_DEGREE_LAT * cos_lat)
        return (
            self.latitude - d_lat,
            self.latitude + d_lat,
            self.longitude - d_lon,
            self.longitude + d_lon,
        )

    def chroma_where(self) -> dict | None:
        """The metadata filter for this scope's most specific level."""
        level = self.level
        if level == "radius":
            lat_min, lat_max, lon_min, lon_max = self.bounding_box()
            where: dict = {
                "$and": [
                    {"latitude": {"$gte": lat_min}},
                    {"latitude": {"$lte": lat_max}},
                    {"longitude": {"$gte": lon_min}},
                    {"longitude": {"$lte": lon_max}},
                ]
            }
        elif level == "nuts3":
            where = {"nuts3": {"$eq": self.nuts3}}
        elif level == "nuts2":
            where = {"nuts2": {"$eq": self.nuts2}}
        elif level == "country":
            where = {"country_code": {"$eq": self.country_code}}
        else:
            return None
        if self.include_null:
            # Unlocated chunks carry "" codes (Chroma has no null), so they can
            # be let through explicitly.
            return {"$or": [where, {"country_code": {"$eq": ""}}]}
        return where

    def to_legacy_filter(self) -> dict[str, str]:
        """The `{nuts2_region, country_code}` dict `src.rag.retry` widens."""
        legacy = {}
        if self.nuts2:
            legacy["nuts2_region"] = self.nuts2
        if self.country_code:
            legacy["country_code"] = self.country_code
        return legacy

    @classmethod
    def from_legacy_filter(cls, legacy: dict[str, str] | None) -> GeoScope | None:
        if not legacy:
            return None
        return cls(
            country_code=legacy.get("country_code"),
            nuts2=legacy.get("nuts2_region"),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> GeoScope:
        return cls(**json.loads(text))

    def describe(self) -> str:
        parts = [self.label] if self.label else []
        codes = [code for code in (self.nuts3, self.nuts2, self.country_code) if code]
        if codes:
            parts.append("/".join(codes))
        if self.radius_km and self.coordinates:
            parts.append(f"within {self.radius_km:g} km")
        return " ".join(parts) or "no scope"
