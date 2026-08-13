from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.shared.geocode import Coordinates

if TYPE_CHECKING:
    from src.rag.search import ScoredPlace

__all__ = ["apply_geo_boost", "haversine_km"]


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


def apply_geo_boost(
    places: list[ScoredPlace],
    question_coords: Coordinates | None,
    *,
    weight: float,
    decay_km: float,
) -> list[ScoredPlace]:
    """Rescale each place's score by distance from the question's location.

    A place without coordinates is left unboosted rather than penalized,
    since missing geocoding data should not push an otherwise relevant place
    out of the answer.
    """
    if question_coords is None:
        return places

    boosted = [
        place.model_copy(
            update={
                "score": place.score * _boost(place, question_coords, weight, decay_km)
            }
        )
        for place in places
    ]
    return sorted(boosted, key=lambda place: place.score, reverse=True)


def _boost(
    place: ScoredPlace,
    question_coords: Coordinates,
    weight: float,
    decay_km: float,
) -> float:
    if place.latitude is None or place.longitude is None:
        return 1.0
    distance_km = haversine_km(
        question_coords, Coordinates(place.latitude, place.longitude)
    )
    return (1 - weight) + weight * math.exp(-distance_km / decay_km)
