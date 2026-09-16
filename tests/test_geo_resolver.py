"""Gazetteer chain and the resolver's cache, with every network source stubbed."""

from src.shared import geo_resolver
from src.shared.geo_resolver import (
    ExtractedQueryLocation,
    Gazetteer,
    LlmGazetteerResolver,
    normalize_query,
)
from src.shared.geo_scope import GeoScope
from src.shared.geocode import Coordinates, GeocodeHit


class StubGeocoder:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def geocode(self, query):
        self.queries.append(query)
        return self.hits.get(query)


def _hit(lat, lon, display):
    return GeocodeHit(Coordinates(lat, lon), display, "place/village", "SI", 0.5)


def test_region_names_resolve_to_nuts_codes_without_any_lookup(tiny_nuts, monkeypatch):
    monkeypatch.setattr(geo_resolver, "nuts_index", lambda: tiny_nuts)
    geocoder = StubGeocoder({})
    gazetteer = Gazetteer(geocoder=geocoder)

    scope = gazetteer.lookup("Posavska statistična regija", "region")

    assert scope == GeoScope(
        country_code="SI",
        nuts2="SI03",
        nuts3="SI036",
        label="Posavska statistična regija",
    )
    assert gazetteer.lookup("Slovenija", "country").level == "country"
    assert geocoder.queries == []


def test_point_lookup_geocodes_verifies_the_name_and_reads_codes_off_the_map(
    tiny_nuts, monkeypatch
):
    monkeypatch.setattr(geo_resolver, "nuts_index", lambda: tiny_nuts)
    monkeypatch.setattr(geo_resolver, "primary_locations", lambda: {})
    geocoder = StubGeocoder(
        {
            "Brestanica": _hit(45.996, 15.477, "Brestanica, Krško, Slovenija"),
            "Nowhere Castle": _hit(46.0, 15.0, "Slovenska cesta, Ljubljana"),
        }
    )
    gazetteer = Gazetteer(geocoder=geocoder, default_radius_km=25)

    scope = gazetteer.lookup("Brestanica", "point")
    assert (scope.nuts3, scope.nuts2, scope.country_code) == ("SI036", "SI03", "SI")
    assert scope.radius_km == 25 and scope.level == "radius"
    assert gazetteer.lookup("Brestanica", "point", radius_km=5).radius_km == 5
    # A hit whose name shares nothing with the request is refused, not trusted.
    assert gazetteer.lookup("Nowhere Castle", "point") is None


def test_region_with_no_nuts_name_keeps_only_nuts2_of_its_point(tiny_nuts, monkeypatch):
    monkeypatch.setattr(geo_resolver, "nuts_index", lambda: tiny_nuts)
    monkeypatch.setattr(geo_resolver, "primary_locations", lambda: {})
    geocoder = StubGeocoder(
        {"eastern Slovenia": _hit(46.0, 16.0, "Eastern Slovenia, Slovenija")}
    )

    scope = Gazetteer(geocoder=geocoder).lookup("eastern Slovenia", "region")

    assert (scope.nuts2, scope.nuts3, scope.radius_km) == ("SI03", None, None)
    assert scope.level == "nuts2"


def test_corpus_locations_are_the_first_gazetteer(tiny_nuts, monkeypatch):
    from src.db.pages import PageLocation

    monkeypatch.setattr(geo_resolver, "nuts_index", lambda: tiny_nuts)
    castle = PageLocation(
        "p",
        "primary",
        "Q1",
        "Grad Rajhenburg",
        "Q1",
        45.989,
        15.466,
        "point",
        "SI",
        "SI03",
        "SI036",
        "Posavska",
        None,
        1.0,
        "wikidata",
    )
    monkeypatch.setattr(geo_resolver, "primary_locations", lambda: {"p": castle})
    geocoder = StubGeocoder({})

    scope = Gazetteer(geocoder=geocoder).lookup("grad rajhenburg", "point")

    assert scope.nuts3 == "SI036" and scope.coordinates == castle.coordinates
    assert scope.radius_km == 25
    assert geocoder.queries == []


class ScriptedLlm:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def structured_output(self, prompt, output_schema, *, retries=3):
        self.calls += 1
        return self.answers.pop(0)


def test_resolver_caches_hits_and_misses_by_normalised_query(tiny_nuts, monkeypatch):
    monkeypatch.setattr(geo_resolver, "nuts_index", lambda: tiny_nuts)
    cache = {}
    monkeypatch.setattr(geo_resolver, "cached_geo_scope", cache.get)
    monkeypatch.setattr(geo_resolver, "store_geo_scope", cache.__setitem__)
    llm = ScriptedLlm(
        [
            ExtractedQueryLocation(place="Posavska", granularity="region"),
            ExtractedQueryLocation(place=None, granularity="none"),
        ]
    )
    resolver = LlmGazetteerResolver(llm=llm, gazetteer=Gazetteer())

    first = resolver.resolve("Kaj videti v Posavju?")
    again = resolver.resolve("  kaj videti v POSAVJU? ")
    assert first.nuts3 == "SI036" and again == first
    assert resolver.resolve("Kje se je rodil Milan Grlj?") is None
    assert resolver.resolve("kje se je rodil milan grlj?") is None
    assert llm.calls == 2
    assert cache[normalize_query("Kje se je rodil Milan Grlj?")] == "null"


def test_resolver_extraction_failure_means_no_scope(tiny_nuts, monkeypatch, capsys):
    class Failing:
        def structured_output(self, prompt, output_schema, *, retries=3):
            raise RuntimeError("structured call failed")

    resolver = LlmGazetteerResolver(
        llm=Failing(), gazetteer=Gazetteer(), use_cache=False
    )
    assert resolver.resolve("anything") is None
    assert "no scope" in capsys.readouterr().out


def test_country_outside_the_nuts_names_is_a_country_scope_not_its_centroid_unit(
    tiny_nuts, monkeypatch
):
    monkeypatch.setattr(geo_resolver, "nuts_index", lambda: tiny_nuts)
    monkeypatch.setattr(geo_resolver, "primary_locations", lambda: {})
    # "Avstrija" matches no NUTS name under the si hint; Nominatim returns a
    # centroid that (in the tiny index) falls inside SI036.
    geocoder = StubGeocoder({"Avstrija": _hit(46.0, 15.0, "Avstrija")})

    scope = Gazetteer(geocoder=geocoder, country_hint="si").lookup(
        "Avstrija", "country"
    )

    assert scope.level == "country"
    assert (scope.nuts3, scope.nuts2, scope.radius_km) == (None, None, None)


def test_resolver_does_not_cache_failures_or_gazetteer_misses(
    tiny_nuts, monkeypatch, capsys
):
    monkeypatch.setattr(geo_resolver, "nuts_index", lambda: tiny_nuts)
    monkeypatch.setattr(geo_resolver, "primary_locations", lambda: {})
    cache = {}
    monkeypatch.setattr(geo_resolver, "cached_geo_scope", cache.get)
    monkeypatch.setattr(geo_resolver, "store_geo_scope", cache.__setitem__)

    class Flaky:
        def __init__(self):
            self.calls = 0

        def structured_output(self, prompt, output_schema, *, retries=3):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("timeout")
            return ExtractedQueryLocation(place="Atlantis", granularity="point")

    llm = Flaky()
    resolver = LlmGazetteerResolver(
        llm=llm, gazetteer=Gazetteer(geocoder=StubGeocoder({}))
    )

    assert resolver.resolve("near Atlantis") is None  # LLM failed
    assert resolver.resolve("near Atlantis") is None  # gazetteer found nothing
    assert cache == {}  # neither answer is settled, so neither is remembered
    assert llm.calls == 2
    assert "not cached" in capsys.readouterr().out
