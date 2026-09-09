"""The gazetteer-miss classifier: no network, fake geocoders."""

from __future__ import annotations

from src.preprocess.rejections import (
    Rejection,
    classify_rejections,
    deinflect,
    format_rejections,
    head_segment,
    inflected_candidates,
    parse_rejections_table,
    retry_candidates,
)
from src.shared.geocode import Coordinates, GeocodeHit


def _hit(name: str) -> GeocodeHit:
    return GeocodeHit(
        coordinates=Coordinates(46.0, 15.0),
        display_name=f"{name}, Slovenija",
        kind="place/town",
        country_code="SI",
        importance=0.5,
    )


def test_retry_candidates_strip_type_words_and_qualifiers():
    assert retry_candidates("Grad Sevnica") == ["Sevnica"]
    assert retry_candidates("Grad Rajhenburg, Brestanica, Slovenija") == [
        "Grad Rajhenburg, Brestanica",
        "Rajhenburg, Brestanica",
        "Rajhenburg",
        "Brestanica",
    ]
    assert "Brestanici" in retry_candidates("v Brestanici")
    assert retry_candidates("Sevnica") == []


def test_retry_candidates_drop_parentheses():
    assert retry_candidates("Trg (naselje)") == ["Trg"]


def test_deinflect_produces_slovenian_nominatives():
    assert "Brestanica" in deinflect("Brestanici")
    assert "Celje" in deinflect("Celju")
    assert "Krško" in deinflect("Krškem")
    assert "Maribor" in deinflect("Mariboru")
    assert "Ljubljana" in deinflect("Ljubljani")
    assert deinflect("Rim") == []


def test_inflected_candidates_change_only_the_head_noun():
    # Only the last word is deinflected: a multi-word oblique form like "Novem
    # mestu" keeps its inflected adjective. Known limitation, documented here.
    assert "Novem mesto" in inflected_candidates("Novem mestu")
    assert "Novo mesto" not in inflected_candidates("Novem mestu")
    assert inflected_candidates("v Sevnici")[0] == "Sevnica"
    assert inflected_candidates("Kostanjevica na Krki")[0] == "Kostanjevica na Krka"


def test_classification_prefers_the_cheapest_fix():
    hinted = {"Sevnica": _hit("Sevnica"), "Brestanica": _hit("Brestanica")}
    unhinted = {"Trst": _hit("Trst")}
    rejections = [
        Rejection("Grad Sevnica", "nominatim: no hit", "p1", "mentioned"),
        Rejection("v Brestanici", "nominatim: no hit", "p1", "mentioned"),
        Rejection("Brestanici", "nominatim: no hit", "p2", "mentioned"),
        Rejection("Trst", "nominatim: no hit", "p3", "mentioned"),
        Rejection("Emona", "nominatim: no hit", "p4", "primary"),
        Rejection("Nikjer", "nominatim: no hit", "p5", "mentioned"),
    ]
    rows = classify_rejections(
        rejections,
        geocode_hinted=hinted.get,
        geocode_unhinted=unhinted.get,
        wikidata_lookup=lambda n: (
            ("Emona", Coordinates(46.05, 14.5)) if n == "Emona" else None
        ),
        known_names=["Brestanica"],
    )
    by_name = {row.name: row for row in rows}
    assert by_name["Grad Sevnica"].category == "qualifier"
    assert by_name["Grad Sevnica"].resolved_as == "Sevnica"
    # "v Brestanici": qualifier stripping (drop the preposition) still gives an
    # oblique form, which the fake gazetteer does not know; deinflection then
    # matches the corpus name without any lookup.
    assert by_name["v Brestanici"].category == "inflected"
    assert by_name["v Brestanici"].resolved_as == "brestanica"
    assert by_name["Trst"].category == "outside_hint"
    assert by_name["Emona"].category == "wikidata"
    assert by_name["Nikjer"].category == "unresolved"
    # Categories sort in fix order; unresolved last.
    assert [row.category for row in rows][-1] == "unresolved"


def test_distinct_names_are_grouped_across_pages():
    rejections = [
        Rejection("Nikjer", "nominatim: no hit", "p1", "mentioned"),
        Rejection("nikjer", "nominatim: no hit", "p2", "primary"),
    ]
    rows = classify_rejections(
        rejections, geocode_hinted=None, geocode_unhinted=None, wikidata_lookup=None
    )
    assert len(rows) == 1
    assert rows[0].pages == 2
    assert rows[0].roles == ("mentioned", "primary")
    text = format_rejections(rows)
    assert "distinct rejected names: 1 (over 2 page mentions)" in text
    assert "unresolved=1" in text


def test_no_geocoders_means_everything_unresolved_without_error():
    rows = classify_rejections(
        [Rejection("Grad Sevnica", "nominatim: no hit", "p", "mentioned")],
        geocode_hinted=None,
        geocode_unhinted=None,
        wikidata_lookup=None,
    )
    assert rows[0].category == "unresolved"


def test_head_segment_is_the_place_not_the_qualifier():
    assert head_segment("Gradec, Austria") == "Gradec"
    assert head_segment("Grad Sevnica") == "Sevnica"
    assert head_segment("v Brestanici") == "Brestanici"
    assert head_segment("Dunaj (Vienna), Austria") == "Dunaj"


def test_a_qualifier_hit_must_name_the_head():
    # "Austria" alone resolves (to a hotel in Ljubljana, say) but does not name
    # Gradec, so the row is not a qualifier fix. Unhinted, Gradec resolves.
    hinted = {"Austria": _hit("Austria Trend Hotel Ljubljana")}
    unhinted = {"Gradec, Austria": _hit("Gradec, Steiermark, Österreich")}
    rows = classify_rejections(
        [Rejection("Gradec, Austria", "nominatim: no hit", "p", "mentioned")],
        geocode_hinted=hinted.get,
        geocode_unhinted=unhinted.get,
        wikidata_lookup=None,
    )
    assert rows[0].category == "outside_hint"


def test_corpus_names_are_found_before_any_lookup():
    calls = []

    def geocode(query):
        calls.append(query)
        return None

    rows = classify_rejections(
        [Rejection("Rajhenburg", "nominatim: name mismatch", "p", "primary")],
        geocode_hinted=geocode,
        geocode_unhinted=geocode,
        wikidata_lookup=None,
        known_names=["Grad Rajhenburg", "Brestanica"],
    )
    assert rows[0].category == "corpus"
    assert rows[0].resolved_as == "grad rajhenburg"
    assert calls == []


def test_table_round_trip_keeps_names_reasons_and_page_counts():
    rows = classify_rejections(
        [
            Rejection("Nikjer", "nominatim: no hit", "p1", "mentioned"),
            Rejection("Nikjer", "nominatim: no hit", "p2", "primary"),
            Rejection("Tam", "nominatim: name mismatch", "p3", "mentioned"),
        ],
        geocode_hinted=None,
        geocode_unhinted=None,
        wikidata_lookup=None,
    )
    parsed = parse_rejections_table(format_rejections(rows))
    assert sorted((r.name, r.reason) for r in parsed) == [
        ("Nikjer", "nominatim: no hit"),
        ("Nikjer", "nominatim: no hit"),
        ("Tam", "nominatim: name mismatch"),
    ]
    assert len({r.page_id for r in parsed}) == 3


def test_one_shared_token_is_not_a_corpus_match():
    rows = classify_rejections(
        [
            Rejection(
                "Stara vas, Videm ob Savi", "nominatim: no hit", "p", "mentioned"
            ),
            Rejection("Rajhenburg", "nominatim: name mismatch", "q", "primary"),
            Rejection("Sevnica Castle, Sevnica", "nominatim: no hit", "r", "mentioned"),
        ],
        geocode_hinted=None,
        geocode_unhinted=None,
        wikidata_lookup=None,
        known_names=["Stara Zagora", "Grad Rajhenburg", "Sevnica"],
    )
    by_name = {row.name: row for row in rows}
    assert by_name["Stara vas, Videm ob Savi"].category == "unresolved"
    assert by_name["Rajhenburg"].category == "corpus"  # the whole proper name
    assert by_name["Sevnica Castle, Sevnica"].category == "corpus"  # exact head
