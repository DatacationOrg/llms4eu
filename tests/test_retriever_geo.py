from src.retrieval.base import RankedChunk
from src.retrieval.retrievers.geo import GeoBoostRetriever
from src.shared.geocode import Coordinates


class _FakeRetriever:
    name = "fake"

    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, limit):
        return self.chunks[:limit]


def _retriever(chunks, coords_by_chunk):
    return GeoBoostRetriever(
        name="fake_geo",
        base_retriever=_FakeRetriever(chunks),
        llm=object(),
        geocoder=object(),
        chunk_coordinates=lambda ids: {
            k: v for k, v in coords_by_chunk.items() if k in ids
        },
        weight=0.5,
        decay_km=50,
    )


def test_leaves_chunks_unchanged_when_query_names_no_place(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.retrievers.geo.locate_text", lambda llm, geocoder, query: None
    )
    chunks = [RankedChunk(id="a", score=0.9, text="t")]

    assert _retriever(chunks, {}).retrieve("no place here", 10) == chunks


def test_favors_the_closer_chunk_over_a_higher_base_score(monkeypatch):
    ljubljana = Coordinates(46.0569, 14.5058)
    new_york = Coordinates(40.7128, -74.0060)
    monkeypatch.setattr(
        "src.retrieval.retrievers.geo.locate_text",
        lambda llm, geocoder, query: ljubljana,
    )
    near = RankedChunk(id="near", score=0.70, text="t")
    far = RankedChunk(id="far", score=0.75, text="t")
    retriever = _retriever([far, near], {"near": ljubljana, "far": new_york})

    ranked = retriever.retrieve("castles near Ljubljana", 10)

    assert [chunk.id for chunk in ranked] == ["near", "far"]


def test_leaves_chunks_without_coordinates_unboosted(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.retrievers.geo.locate_text",
        lambda llm, geocoder, query: Coordinates(46.0569, 14.5058),
    )
    chunk = RankedChunk(id="unlocated", score=0.6, text="t")

    ranked = _retriever([chunk], {}).retrieve("a place", 10)

    assert ranked[0].score == 0.6
