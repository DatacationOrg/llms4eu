"""Geo primitives: distance, boost, scope widening and filters, NUTS lookup."""

import json

import pytest

from src.shared import nuts as nuts_module
from src.shared.geo_scope import GeoScope
from src.shared.geocode import (
    Coordinates,
    distance_multiplier,
    haversine_km,
    name_matches,
)
from src.shared.nuts import normalize_place_name, nuts_parents
from src.shared.wikidata import WikidataEntity


def test_haversine_km_matches_known_distance():
    ljubljana = Coordinates(46.0569, 14.5058)
    zagreb = Coordinates(45.8150, 15.9819)
    assert 110 < haversine_km(ljubljana, zagreb) < 125


def test_distance_multiplier_is_one_at_zero_and_floors_far_away():
    assert distance_multiplier(0, weight=0.25, decay_km=50) == 1.0
    far = distance_multiplier(100_000, weight=0.25, decay_km=50)
    assert 0.75 <= far < 0.751


def test_scope_widens_radius_then_nuts3_then_nuts2_then_country():
    scope = GeoScope(
        country_code="SI",
        nuts2="SI03",
        nuts3="SI036",
        latitude=45.99,
        longitude=15.47,
        radius_km=25,
    )
    levels = []
    current = scope
    while current is not None:
        levels.append(current.level)
        current = current.widen()
    assert levels == ["radius", "nuts3", "nuts2", "country", "none"]
    # Widened to nothing, the point is still there for the distance boost.
    assert scope.widen().widen().widen().widen().coordinates == scope.coordinates
    assert scope.widen().widen().widen().widen().filters is False


def test_chroma_where_uses_the_most_specific_level():
    scope = GeoScope(country_code="SI", nuts2="SI03", nuts3="SI036")
    assert scope.chroma_where() == {"nuts3": {"$eq": "SI036"}}
    assert scope.widen().chroma_where() == {"nuts2": {"$eq": "SI03"}}
    assert GeoScope(country_code="SI", include_null=True).chroma_where() == {
        "$or": [{"country_code": {"$eq": "SI"}}, {"country_code": {"$eq": ""}}]
    }
    assert GeoScope(latitude=1.0, longitude=1.0).chroma_where() is None


def test_radius_where_is_a_bounding_box():
    scope = GeoScope(latitude=46.0, longitude=15.0, radius_km=111.32)
    where = scope.chroma_where()["$and"]
    assert where[0] == {"latitude": {"$gte": pytest.approx(45.0)}}
    assert where[1] == {"latitude": {"$lte": pytest.approx(47.0)}}
    lat_min, lat_max, lon_min, lon_max = scope.bounding_box()
    assert lon_max - lon_min > lat_max - lat_min  # degrees of longitude shrink north


def test_legacy_filter_round_trip_matches_wider_geo_filter_order():
    scope = GeoScope(country_code="SI", nuts2="SI03", nuts3="SI036")
    legacy = scope.to_legacy_filter()
    assert legacy == {"nuts2_region": "SI03", "country_code": "SI"}
    restored = GeoScope.from_legacy_filter(legacy)
    assert restored.widen().to_legacy_filter() == {"country_code": "SI"}
    assert restored.widen().widen().to_legacy_filter() == {}


def test_scope_json_round_trip():
    scope = GeoScope(country_code="SI", nuts3="SI036", label="Posavska", radius_km=None)
    assert GeoScope.from_json(scope.to_json()) == scope
    assert json.loads(scope.to_json())["label"] == "Posavska"


def test_normalize_place_name_folds_diacritics_and_filler():
    assert normalize_place_name("Posavska statistična regija") == "posavska"
    assert normalize_place_name("Štajerska") == "stajerska"
    assert (
        normalize_place_name("Grad Rajhenburg, Brestanica")
        == "grad rajhenburg brestanica"
    )


def test_name_matches_needs_a_shared_token_of_three_letters():
    assert name_matches(
        "Brestanica Castle", "Grad Rajhenburg, Brestanica, Krško, Slovenija"
    )
    assert not name_matches(
        "Brestanica Castle", "Slovenska cesta, Ljubljana, Slovenija"
    )
    assert not name_matches("na", "Ljubljana")


def test_nuts_parents():
    assert nuts_parents("SI036") == {
        "country_code": "SI",
        "nuts1": "SI0",
        "nuts2": "SI03",
        "nuts3": "SI036",
    }
    assert nuts_parents("SI")["nuts2"] is None


def test_nuts_index_locates_points_and_names(tiny_nuts):
    assert tiny_nuts.locate(45.99, 15.47) == "SI036"
    assert tiny_nuts.locate(45.99, 15.47, level=2) == "SI03"
    assert tiny_nuts.locate(45.99, 16.0) == "SI037"
    assert tiny_nuts.locate(48.2, 16.4) is None
    assert tiny_nuts.find_by_name("Posavska statistična regija") == ["SI036"]
    assert tiny_nuts.find_by_name("vzhodna slovenija") == ["SI03"]
    assert tiny_nuts.find_by_name("Slovenija")[0] == "SI"
    assert tiny_nuts.find_by_name("Jugovzhodna") == ["SI037"]
    # Inflected forms match on word stems, as a last tier.
    assert tiny_nuts.find_by_name("Posavje") == ["SI036"]
    assert tiny_nuts.find_by_name("Posavsko") == ["SI036"]
    assert tiny_nuts.find_by_name("Gorenjska") == []
    assert tiny_nuts.name_of("SI036") == "Posavska"


def test_nuts_index_missing_file_says_how_to_fetch(tmp_path):
    with pytest.raises(RuntimeError, match="fetch-nuts"):
        nuts_module.nuts_index(tmp_path / "missing.geojson")


def _entity(**overrides) -> WikidataEntity:
    base = dict(
        qid="Q1",
        label="x",
        coordinates=Coordinates(46.0, 15.0),
        instance_of=(),
        country_qid="Q215",
        admin_parent_qid=None,
        nuts_codes=(),
        iso_3166_2=None,
    )
    return WikidataEntity(**{**base, **overrides})


def test_wikidata_granularity_rules():
    assert _entity(instance_of=("Q5",)).granularity is None  # a person
    assert _entity(coordinates=None, instance_of=("Q811102",)).granularity is None
    assert _entity(instance_of=("Q6256",)).granularity == "country"
    assert _entity(nuts_codes=("SI", "SI0")).granularity == "country"
    assert (
        _entity(nuts_codes=("SI036",), instance_of=("Q1474320",)).granularity
        == "region"
    )
    assert _entity(instance_of=("Q82794",)).granularity == "region"
    assert _entity(instance_of=("Q6960199",)).granularity == "municipality"
    assert _entity(instance_of=("Q23413",)).granularity == "point"
