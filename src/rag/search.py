import argparse
from pathlib import Path

from src.db.places import load_places_by_id
from src.rag.geo import apply_geo_boost
from src.shared.embed import embed_texts, load_embedder
from src.shared.env import load_local_env, load_yaml
from src.shared.geocode import NominatimGeocoder, locate_text
from src.shared.llm import LocalOllamaStructuredLlm
from src.shared.schema import Place
from src.vector_store.places import search_place_vectors


__all__ = ["ScoredPlace", "search_places", "search_places_with_model", "search"]

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
SEARCH_CONFIG = CONFIG["search"]
GEO_CONFIG = CONFIG["geo"]


class ScoredPlace(Place):
    score: float


def search_places(query: str, limit: int | None = None) -> list[ScoredPlace]:
    return search_places_with_model(
        query=query,
        embedding_model=SEARCH_CONFIG["embedding_model"],
        limit=limit,
        collection_name=SEARCH_CONFIG["collection_name"],
    )


def search_places_with_model(
    query: str,
    embedding_model: str,
    limit: int | None = None,
    collection_name: str | None = None,
) -> list[ScoredPlace]:
    load_local_env()
    model = load_embedder(embedding_model)
    vector = embed_texts(model, [query])[0]

    hits = search_place_vectors(
        collection_name or SEARCH_CONFIG["collection_name"],
        vector,
        limit or SEARCH_CONFIG["default_limit"],
    )
    ids = [hit.id for hit in hits]

    places = load_places_by_id(ids)
    scored = [
        ScoredPlace(score=hit.score, **places[hit.id].model_dump())
        for hit in hits
        if hit.id in places
    ]
    return _apply_geo_ranking(query, scored)


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



def _apply_geo_ranking(query: str, places: list[ScoredPlace]) -> list[ScoredPlace]:
    if not GEO_CONFIG["enabled"] or not places:
        return places

    coords = locate_text(
        LocalOllamaStructuredLlm(GEO_CONFIG["model"], method="function_calling"),
        NominatimGeocoder(),
        query,
    )
    return apply_geo_boost(
        places,
        coords,
        weight=GEO_CONFIG["weight"],
        decay_km=GEO_CONFIG["decay_km"],
    )


if __name__ == "__main__":
    search()
