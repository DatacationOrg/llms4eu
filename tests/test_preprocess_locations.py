"""Location rows: the granularity and code rules, and the Wikipedia title parse."""

from src.preprocess import locations
from src.preprocess.locations import _location_from_entity, _wikipedia_title
from src.shared.geocode import Coordinates
from src.shared.wikidata import WikidataEntity


def _entity(**overrides):
    base = dict(
        qid="Q1",
        label="Grad Rajhenburg",
        coordinates=Coordinates(45.989, 15.466),
        instance_of=("Q23413",),
        country_qid="Q215",
        admin_parent_qid="Q3484655",
        nuts_codes=(),
        iso_3166_2=None,
    )
    return WikidataEntity(**{**base, **overrides})


def test_wikipedia_title_parses_language_and_unquotes():
    assert _wikipedia_title("https://sl.wikipedia.org/wiki/Grad_Rajhenburg") == (
        "sl",
        "Grad Rajhenburg",
    )
    assert _wikipedia_title("https://sl.wikipedia.org/wiki/Ob%C4%8Dina_Kr%C5%A1ko") == (
        "sl",
        "Občina Krško",
    )
    assert _wikipedia_title("https://www.brestanica.com/cerkve/") == ("", None)


def test_point_gets_the_full_hierarchy_from_its_coordinates(tiny_nuts, monkeypatch):
    monkeypatch.setattr(locations, "nuts_index", lambda: tiny_nuts)

    row = _location_from_entity("p", _entity(), "wikidata", 1.0)

    assert (row.country_code, row.nuts2, row.nuts3, row.nuts3_name) == (
        "SI",
        "SI03",
        "SI036",
        "Posavska",
    )
    assert row.granularity == "point" and row.role == "primary"
    assert row.location_key == "Q1" and row.method == "wikidata"


def test_region_keeps_nuts3_only_when_it_is_that_region(tiny_nuts, monkeypatch):
    monkeypatch.setattr(locations, "nuts_index", lambda: tiny_nuts)
    posavska = _entity(
        label="Posavska", instance_of=("Q1474320",), nuts_codes=("SI036",)
    )
    stajerska = _entity(label="Štajerska", instance_of=("Q82794",))

    assert _location_from_entity("p", posavska, "wikidata", 1.0).nuts3 == "SI036"
    wide = _location_from_entity("p", stajerska, "wikidata", 1.0)
    # Spans several NUTS-3 units: its centroid's NUTS-2 stands, NUTS-3 does not.
    assert (wide.granularity, wide.nuts2, wide.nuts3) == ("region", "SI03", None)


def test_country_keeps_only_its_code(tiny_nuts, monkeypatch):
    monkeypatch.setattr(locations, "nuts_index", lambda: tiny_nuts)
    slovenia = _entity(
        label="Slovenia",
        instance_of=("Q6256",),
        nuts_codes=("SI", "SI0"),
        coordinates=Coordinates(46.0, 15.0),
    )
    row = _location_from_entity("p", slovenia, "wikidata", 1.0)
    assert (row.granularity, row.country_code, row.nuts2, row.nuts3) == (
        "country",
        "SI",
        None,
        None,
    )


def test_non_places_get_no_row(tiny_nuts, monkeypatch):
    monkeypatch.setattr(locations, "nuts_index", lambda: tiny_nuts)
    person = _entity(instance_of=("Q5",), coordinates=None)
    language = _entity(instance_of=("Q34770",), country_qid=None, admin_parent_qid=None)
    assert _location_from_entity("p", person, "wikidata", 1.0) is None
    assert _location_from_entity("p", language, "wikidata", 1.0) is None


def test_languages_and_vanished_states_are_not_places(tiny_nuts, monkeypatch):
    """Wikidata gives German (Q188) coordinates and a country; still not a place."""
    monkeypatch.setattr(locations, "nuts_index", lambda: tiny_nuts)
    german = _entity(instance_of=("Q34770", "Q1288568"), country_qid="Q183")
    empire = _entity(instance_of=("Q3024240", "Q48349"))
    assert _location_from_entity("p", german, "wikidata", 1.0) is None
    assert _location_from_entity("p", empire, "wikidata", 1.0) is None
