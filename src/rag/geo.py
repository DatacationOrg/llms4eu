from __future__ import annotations

from typing import TYPE_CHECKING

from src.shared.geo_boost import distance_multiplier, haversine_km
from src.shared.geocode import Coordinates

if TYPE_CHECKING:
    from src.rag.search import ScoredPlace

__all__ = ["apply_geo_boost", "haversine_km"]


def apply_geo_boost(
    places: list[ScoredPlace],
    question_coords: Coordinates | None,
    *,
    weight: float,
    decay_km: float,
) -> list[ScoredPlace]:
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
    return distance_multiplier(distance_km, weight=weight, decay_km=decay_km)
