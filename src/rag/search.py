import argparse
from pathlib import Path

from src.db.places import load_places_by_id
from src.shared.embed import embed_texts, load_embedder
from src.shared.env import load_local_env, load_yaml
from src.shared.schema import Place
from src.vector_store.places import search_place_vectors


__all__ = ["ScoredPlace", "search_places", "search"]

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
SEARCH_CONFIG = CONFIG["search"]


class ScoredPlace(Place):
    score: float


def search_places(query: str, limit: int | None = None) -> list[ScoredPlace]:
    load_local_env()
    model = load_embedder(SEARCH_CONFIG["embedding_model"])
    vector = embed_texts(model, [query])[0]

    hits = search_place_vectors(
        SEARCH_CONFIG["collection_name"],
        vector,
        limit or SEARCH_CONFIG["default_limit"],
    )
    ids = [hit.id for hit in hits]

    places = load_places_by_id(ids)
    return [
        ScoredPlace(score=hit.score, **places[hit.id].model_dump())
        for hit in hits
        if hit.id in places
    ]


def search() -> None:
    """CLI entrypoint for inspecting retrieval without an LLM answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=SEARCH_CONFIG["default_limit"])
    args = parser.parse_args()

    for place in search_places(args.query, args.limit):
        print(f"{place.score:.3f} {place.id}")
        print(place.summary)
        print(place.place_description)
        print()


if __name__ == "__main__":
    search()
