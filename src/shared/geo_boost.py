from __future__ import annotations

import math

from src.shared.geocode import Coordinates

__all__ = ["distance_multiplier", "haversine_km"]


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
    return (1 - weight) + weight * math.exp(-distance_km / decay_km)
