from pathlib import Path

from src.db.places import load_places
from src.shared.embed import embed_texts, load_embedder
from src.shared.env import load_local_env, load_yaml
from src.vector_store.places import recreate_places_collection, upsert_place_vectors


__all__ = ["rebuild_vector_index"]

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def rebuild_vector_index() -> None:
    load_local_env()

    places = load_places()

    model = load_embedder(CONFIG["embedding_model"])
    vectors = embed_texts(
        model,
        [p.embedding_text for p in places],
    )

    recreate_places_collection(CONFIG["collection_name"])
    upsert_place_vectors(CONFIG["collection_name"], places, vectors)

    print(f"indexed {len(places)} places")


if __name__ == "__main__":
    rebuild_vector_index()
