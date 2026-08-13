from src.db.initialize import initialize_db
from src.preprocess.index import rebuild_vector_index
from src.rag.search import search_places


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
