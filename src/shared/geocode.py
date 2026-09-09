"""Coordinates, distance, and the OpenStreetMap Nominatim gazetteer.

Ported from the unmerged `feature/geo-aware-place-retrieval` branch. A model
never produces coordinates here: it names a place, and the gazetteer resolves
and thereby verifies it (Hu et al. 2024, toponym resolution with lightweight
LLMs and geo-knowledge).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import EllipsisType
from typing import NamedTuple, Protocol

import httpx

from src.shared.nuts import normalize_place_name

__all__ = [
    "Coordinates",
    "GeocodeHit",
    "GeocodeProvider",
    "NominatimGeocoder",
    "distance_multiplier",
    "haversine_km",
    "name_matches",
]

USER_AGENT = "llms4eu-tourism-rag (g.dekleuver@datacation.nl)"


class Coordinates(NamedTuple):
    latitude: float
    longitude: float


@dataclass(frozen=True)
class GeocodeHit:
    coordinates: Coordinates
    display_name: str
    kind: str  # Nominatim class/type, e.g. "boundary/administrative"
    country_code: str | None
    importance: float


class GeocodeProvider(Protocol):
    def geocode(self, query: str) -> GeocodeHit | None: ...


@dataclass
class NominatimGeocoder:
    """One request per second, per the Nominatim usage policy."""

    user_agent: str = USER_AGENT
    base_url: str = "https://nominatim.openstreetmap.org/search"
    min_interval_seconds: float = 1.0
    # ISO 3166-1 alpha-2 hint, e.g. "si". Biases ambiguous names to the corpus.
    country_codes: str | None = None
    timeout_seconds: float = 10.0
    _last_request_at: float = field(default=0.0, init=False, repr=False)

    def geocode(
        self, query: str, *, country_codes: str | None | EllipsisType = ...
    ) -> GeocodeHit | None:
        """One lookup. `country_codes=None` lifts the instance's hint for this call."""
        query = (query or "").strip()
        if not query:
            return None
        self._throttle()
        params = {"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1}
        hint = self.country_codes if country_codes is ... else country_codes
        if hint:
            params["countrycodes"] = hint
        try:
            response = httpx.get(
                self.base_url,
                params=params,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout_seconds,
            )
        except httpx.RequestError:
            return None
        if response.status_code != 200:
            return None
        results = response.json()
        if not results:
            return None
        hit = results[0]
        return GeocodeHit(
            coordinates=Coordinates(float(hit["lat"]), float(hit["lon"])),
            display_name=hit.get("display_name", ""),
            kind=f"{hit.get('category', hit.get('class', ''))}/{hit.get('type', '')}",
            country_code=(hit.get("address") or {}).get("country_code", "").upper()
            or None,
            importance=float(hit.get("importance", 0.0)),
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()


def name_matches(requested: str, display_name: str) -> bool:
    """Whether a gazetteer hit plausibly is the place the model named.

    A model that says 'Brestanica Castle' and a hit whose display name is a
    street in Ljubljana share no token; the hit is refused. Tokens under three
    characters are ignored so 'St' or 'na' cannot vouch for a match.
    """
    wanted = {
        token for token in normalize_place_name(requested).split() if len(token) >= 3
    }
    found = set(normalize_place_name(display_name).split())
    return bool(wanted) and bool(wanted & found)


def haversine_km(a: Coordinates, b: Coordinates) -> float:
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = (
        math.radians(value)
        for value in (a.latitude, a.longitude, b.latitude, b.longitude)
    )
    haversine = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(haversine))


def distance_multiplier(distance_km: float, *, weight: float, decay_km: float) -> float:
    """Score multiplier in [1 - weight, 1]: 1 at the place, decaying with distance.

    A soft boost, not a filter: distance can nudge a ranking but never override
    a strong text match, and a chunk with no coordinates keeps multiplier 1.
    """
    return (1 - weight) + weight * math.exp(-distance_km / decay_km)
