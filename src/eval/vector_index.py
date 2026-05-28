from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import chromadb

from src.eval.db import connect, initialize_eval_db
from src.shared.embed import embed_texts, load_embedder
from src.shared.env import chroma_path, load_local_env, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    score: float
    text: str


def rebuild_chunk_vector_index() -> None:
    load_local_env()
    initialize_eval_db()
    chunks = load_chunks()

    client = _client()
    collection_name = CONFIG["collection_name"]
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    model = load_embedder(CONFIG["embedding_model"])
    _configure_embedder(model)
    print(f"embedding {len(chunks)} chunks with {CONFIG['embedding_model']}")
    vectors = embed_texts(
        model,
        [_embedding_text(chunk) for chunk in chunks],
        batch_size=8,
        show_progress_bar=True,
    )

    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=vectors,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "id": chunk["id"],
                "page_id": chunk["page_id"],
                "heading_path": chunk["heading_path"] or "",
                "title": chunk["title"] or "",
                "summary": chunk["summary"] or "",
            }
            for chunk in chunks
        ],
    )
    print(f"indexed {len(chunks)} chunks")


def load_chunks() -> list[dict]:
    with connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                """
                select c.id, c.page_id, c.heading_path, c.text, c.summary, m.title
                from page_chunks c
                join page_metadata m on m.id = c.page_id
                order by c.id
                """
            )
        ]


def search_chunk_vectors(query: str, limit: int) -> list[ScoredChunk]:
    load_local_env()
    vector = embed_texts(_embedder(), [query], prompt_name=_query_prompt_name())[0]
    result = _collection().query(
        query_embeddings=[vector],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        ScoredChunk(
            id=(metadata or {}).get("id", item_id),
            score=1 - distance,
            text=document or "",
        )
        for item_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        )
    ]


def _embedding_text(chunk: dict) -> str:
    parts = [
        chunk.get("title") or "",
        chunk.get("heading_path") or "",
        chunk.get("summary") or "",
        chunk["text"],
    ]
    return "\n".join(part for part in parts if part)


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(chroma_path()))


@cache
def _embedder():
    model = load_embedder(CONFIG["embedding_model"])
    _configure_embedder(model)
    return model


def _configure_embedder(model) -> None:
    max_seq_length = CONFIG.get("embedding_max_seq_length")
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)


def _query_prompt_name() -> str | None:
    """Return 'query' for instruction-aware models (e.g. Qwen3-Embedding), else None."""
    model = _embedder()
    if "query" in (model.prompts or {}):
        return "query"
    return None


def _collection():
    return _client().get_or_create_collection(
        CONFIG["collection_name"],
        metadata={"hnsw:space": "cosine"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rebuild_chunk_vector_index()


if __name__ == "__main__":
    main()
