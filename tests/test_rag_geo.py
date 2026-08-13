from src.rag.geo import apply_geo_boost, haversine_km
from src.rag.search import ScoredPlace
from src.shared.geocode import Coordinates


def _place(id_, score, latitude=None, longitude=None):
    return ScoredPlace(
        id=id_,
        place_description="d",
        summary="s",
        latitude=latitude,
        longitude=longitude,
        score=score,
    )


def test_haversine_km_is_zero_for_identical_points():
    point = Coordinates(46.0, 15.0)
    assert haversine_km(point, point) == 0.0


def test_haversine_km_matches_known_distance():
    ljubljana = Coordinates(46.0569, 14.5058)
    zagreb = Coordinates(45.8150, 15.9819)
    assert 110 < haversine_km(ljubljana, zagreb) < 125


def test_apply_geo_boost_returns_places_unchanged_without_question_coords():
    places = [_place("a", 0.9), _place("b", 0.8)]
    assert apply_geo_boost(places, None, weight=0.25, decay_km=50) == places


def test_apply_geo_boost_favors_the_closer_place_over_a_higher_semantic_score():
    question_coords = Coordinates(46.0569, 14.5058)  # Ljubljana
    near = _place("near", score=0.70, latitude=46.05, longitude=14.51)
    far = _place("far", score=0.75, latitude=40.71, longitude=-74.01)  # New York

    ranked = apply_geo_boost([far, near], question_coords, weight=0.5, decay_km=50)

    assert [place.id for place in ranked] == ["near", "far"]


def test_apply_geo_boost_leaves_places_without_coordinates_unboosted():
    question_coords = Coordinates(46.0569, 14.5058)
    unlocated = _place("unlocated", score=0.6)

    ranked = apply_geo_boost([unlocated], question_coords, weight=0.5, decay_km=50)

    assert ranked[0].score == 0.6


def test_apply_geo_boost_never_drops_below_the_weight_floor():
    # A world away, the decay term underflows to 0: score settles at the
    # (1 - weight) floor instead of being driven all the way to zero.
    question_coords = Coordinates(46.0569, 14.5058)
    far = _place("far", score=0.9, latitude=40.71, longitude=-74.01)

    ranked = apply_geo_boost([far], question_coords, weight=0.5, decay_km=50)

    assert ranked[0].score >= 0.9 * 0.5
    assert ranked[0].score > 0


def test_apply_geo_boost_is_stronger_when_nearby_than_moderately_far():
    question_coords = Coordinates(46.0569, 14.5058)  # Ljubljana
    nearby = _place("nearby", score=0.9, latitude=46.05, longitude=14.51)
    moderately_far = _place(
        "moderately_far", score=0.9, latitude=45.8150, longitude=15.9819
    )  # Zagreb, ~117 km away

    ranked = {
        place.id: place.score
        for place in apply_geo_boost(
            [nearby, moderately_far], question_coords, weight=0.5, decay_km=50
        )
    }

    assert ranked["nearby"] > ranked["moderately_far"]
