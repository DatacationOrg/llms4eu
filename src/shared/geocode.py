from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import httpx
from pydantic import BaseModel, Field

from src.shared.llm import StructuredLlm

__all__ = [
    "Coordinates",
    "ExtractedLocation",
    "GeocodeProvider",
    "NominatimGeocoder",
    "extract_location_query",
    "locate_text",
]

_LOCATION_PROMPT = (
    "Identify the single specific real-world place this text is about.\n"
    "Reply with a short, geocodable name such as 'Brestanica Castle, Slovenia', "
    "or null if the text does not name or clearly imply one specific place.\n\n"
    "text:\n{text}"
)


class Coordinates(NamedTuple):
    latitude: float
    longitude: float


class ExtractedLocation(BaseModel):
    location_query: str | None = Field(
        default=None,
        description=(
            "A short, geocodable place name mentioned or clearly implied by "
            "the text, or null if no single specific place is identifiable."
        ),
    )


class GeocodeProvider(Protocol):
    def geocode(self, query: str) -> Coordinates | None: ...


@dataclass
class NominatimGeocoder:
    user_agent: str = "llms4eu-tourism-rag"
    base_url: str = "https://nominatim.openstreetmap.org/search"
    min_interval_seconds: float = 1.0
    _last_request_at: float = field(default=0.0, init=False, repr=False)

    def geocode(self, query: str) -> Coordinates | None:
        self._throttle()
        try:
            response = httpx.get(
                self.base_url,
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": self.user_agent},
                timeout=10.0,
            )
        except httpx.RequestError:
            return None
        if response.status_code != 200:
            return None
        results = response.json()
        if not results:
            return None
        return Coordinates(float(results[0]["lat"]), float(results[0]["lon"]))

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()


def extract_location_query(llm: StructuredLlm, text: str) -> str | None:
    result = llm.structured_output(
        _LOCATION_PROMPT.format(text=text), ExtractedLocation
    )
    query = (result.location_query or "").strip()
    return query or None


def locate_text(
    llm: StructuredLlm, geocoder: GeocodeProvider, text: str
) -> Coordinates | None:
    query = extract_location_query(llm, text)
    if query is None:
        return None
    return geocoder.geocode(query)
