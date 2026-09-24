from src.retrieval.base import RankedChunk
from src.retrieval.retrievers.geo import GeoRetriever, Place

BRESTANICA = Place(45.99, 15.47, 10.0)
POINTS = {"near": (45.98, 15.48), "far": (46.42, 15.87)}  # 1 km and 57 km away


class Stage:
    name = "stage"

    def retrieve_batch(self, queries, limit):
        chunks = [
            RankedChunk("far:0", 0.9, ""),
            RankedChunk("unlocated:0", 0.85, ""),
            RankedChunk("near:0", 0.8, ""),
            RankedChunk("filler:0", 0.0, ""),  # the over-fetched tail
        ]
        return {i: chunks[:limit] for i in range(len(queries))}


def retriever(place):
    return GeoRetriever("geo", Stage(), lambda _: place, lambda: POINTS, 0.3, 4)


def test_near_page_overtakes_far_one_and_unlocated_stays_neutral():
    ids = [chunk.id for chunk in retriever(BRESTANICA).retrieve("q", 3)]

    assert ids == ["unlocated:0", "near:0", "far:0"]  # far: 0.70, near: 0.88


def test_query_without_place_keeps_the_stage_order():
    ids = [chunk.id for chunk in retriever(None).retrieve("q", 2)]

    assert ids == ["far:0", "unlocated:0"]
