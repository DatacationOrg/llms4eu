from src.data_prep.wiki_places import add_rows, language_ok


def binding(url, lang, item="Q1", coord="Point(15.5 46.0)"):
    return {
        "item": {"value": f"http://www.wikidata.org/entity/{item}"},
        "coord": {"value": coord},
        "sitelinks": {"value": "4"},
        "article": {"value": url},
        "title": {"value": "Grad"},
        "lang": {"value": lang},
    }


def test_add_rows_keeps_one_local_article_per_place():
    rows = {}
    langs = ["nl", "fr", "de"]
    add_rows(
        rows, [binding("https://en.wikipedia.org/wiki/E", "en")], "castle", "BE", langs
    )
    add_rows(
        rows, [binding("https://fr.wikipedia.org/wiki/C", "fr")], "castle", "BE", langs
    )
    add_rows(
        rows, [binding("https://nl.wikipedia.org/wiki/K", "nl")], "garden", "BE", langs
    )
    unknown = binding("https://nl.wikipedia.org/wiki/X", "nl", item="Q2", coord="t1")
    add_rows(rows, [unknown], "castle", "BE", langs)

    assert list(rows) == ["Q1"]
    assert rows["Q1"]["url"] == "https://nl.wikipedia.org/wiki/K"
    assert rows["Q1"]["categories"] == ["castle", "garden"]
    assert (rows["Q1"]["latitude"], rows["Q1"]["longitude"]) == (46.0, 15.5)


def test_language_ok_needs_declared_language_and_host():
    assert language_ok("sl", "sl", "https://sl.wikipedia.org/wiki/Grad")
    assert not language_ok("sl", "en", "https://sl.wikipedia.org/wiki/Grad")
    assert not language_ok("sl", "sl", "https://en.wikipedia.org/wiki/Castle")
    assert not language_ok("sl", None, None)
