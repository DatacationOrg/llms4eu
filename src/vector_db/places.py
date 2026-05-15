from dataclasses import dataclass

import chromadb

from src.shared.env import chroma_path
from src.shared.schema import Place


__all__ = [
    "SearchHit",
    "recreate_places_collection",
    "upsert_place_vectors",
    "search_place_vectors",
]


@dataclass
class SearchHit:
    id: str
    score: float


def recreate_places_collection(collection: str) -> None:
    client = _client()
    for name in [item.name for item in client.list_collections()]:
        if name == collection:
            client.delete_collection(collection)
            break
    client.create_collection(collection, metadata={"hnsw:space": "cosine"})


def upsert_place_vectors(
    collection: str,
    places: list[Place],
    vectors: list[list[float]],
) -> None:
    _collection(collection).upsert(
        ids=[place.id for place in places],
        embeddings=vectors,
        metadatas=[{"id": place.id} for place in places],
    )


def search_place_vectors(
    collection: str,
    vector: list[float],
    limit: int,
) -> list[SearchHit]:
    result = _collection(collection).query(
        query_embeddings=[vector],
        n_results=limit,
        include=["metadatas", "distances"],
    )
    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        SearchHit(
            id=(metadata or {}).get("id", item_id),
            score=1 - distance,
        )
        for item_id, metadata, distance in zip(ids, metadatas, distances)
    ]


# ---------- PRIVATE FUNCTIONS ----------


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(chroma_path()))


def _collection(collection: str):
    return _client().get_or_create_collection(
        collection,
        metadata={"hnsw:space": "cosine"},
    )
