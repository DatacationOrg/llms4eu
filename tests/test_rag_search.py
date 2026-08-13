from src.db.initialize import initialize_db
from src.preprocess.index import rebuild_vector_index
from src.rag.search import search_places, search_places_with_model
from src.shared.geocode import Coordinates


def test_search_returns_forest_walk_for_matching_query(monkeypatch, tmp_path):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "places.db"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))

    monkeypatch.setattr(
        "src.preprocess.index.load_embedder",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "src.preprocess.index.embed_texts",
        lambda _model, texts: [
            [1.0, 0.0] if "forest walk" in text.lower() else [0.0, 1.0]
            for text in texts
        ],
    )
    monkeypatch.setattr("src.rag.search.load_embedder", lambda _: object())
    monkeypatch.setattr(
        "src.rag.search.embed_texts",
        lambda _model, texts: [[1.0, 0.0]],
    )

    initialize_db()
    rebuild_vector_index()

    places = search_places("quiet forest walk near water", limit=1)

    assert len(places) == 1
    assert "forest" in places[0].summary.lower()


def test_search_reorders_by_distance_when_geo_is_enabled(monkeypatch, tmp_path):
    import src.rag.search as search_module
    from src.db.places import load_places, update_place_location

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "places.db"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))

    monkeypatch.setattr("src.preprocess.index.load_embedder", lambda _: object())
    monkeypatch.setattr(
        "src.preprocess.index.embed_texts",
        lambda _model, texts: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr("src.rag.search.load_embedder", lambda _: object())
    monkeypatch.setattr(
        "src.rag.search.embed_texts", lambda _model, texts: [[1.0, 0.0]]
    )

    initialize_db()
    rebuild_vector_index()

    places = load_places()
    ljubljana = Coordinates(46.0569, 14.5058)
    update_place_location(places[0].id, *ljubljana)  # near the question
    update_place_location(places[1].id, 40.7128, -74.0060)  # New York, far away

    monkeypatch.setitem(search_module.GEO_CONFIG, "enabled", True)
    monkeypatch.setitem(search_module.GEO_CONFIG, "weight", 0.5)
    monkeypatch.setitem(search_module.GEO_CONFIG, "decay_km", 50)
    monkeypatch.setattr(
        "src.rag.search.locate_text",
        lambda llm, geocoder, query: ljubljana,
    )

    results = search_places_with_model(
        "a place to visit", embedding_model="fake", limit=len(places)
    )
    ranked_ids = [place.id for place in results]

    assert ranked_ids.index(places[0].id) < ranked_ids.index(places[1].id)
