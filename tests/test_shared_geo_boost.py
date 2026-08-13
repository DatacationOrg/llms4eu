from src.shared.geo_boost import distance_multiplier, haversine_km
from src.shared.geocode import Coordinates


def test_haversine_km_matches_known_distance():
    ljubljana = Coordinates(46.0569, 14.5058)
    zagreb = Coordinates(45.8150, 15.9819)
    assert 110 < haversine_km(ljubljana, zagreb) < 125


def test_distance_multiplier_is_one_at_zero_distance():
    assert distance_multiplier(0, weight=0.25, decay_km=50) == 1.0


def test_distance_multiplier_floors_near_one_minus_weight_far_away():
    far = distance_multiplier(100_000, weight=0.25, decay_km=50)
    assert 0.75 <= far < 0.751
