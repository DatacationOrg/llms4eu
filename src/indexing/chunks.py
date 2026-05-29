from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.db.pages import connect_pages as connect
from src.db.pages import initialize_page_artifacts_db
from src.shared.env import chroma_path, load_local_env, load_yaml
from src.shared.indexers import build_indexer

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
CONTENT_MODES = ("chunk", "summary", "chunk_summary")


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    score: float
    text: str


def rebuild_chunk_vector_index(
    method: str = "qwen", content_mode: str = "chunk_summary"
) -> None:
    _validate_content_mode(content_mode)
    load_local_env()
    initialize_page_artifacts_db()
    chunks = load_chunks()

    client = _client()
    collection_name = _collection_name(method, content_mode)
    with suppress(Exception):
        client.delete_collection(collection_name)

    texts = [_embedding_text(chunk, content_mode) for chunk in chunks]
    indexer = build_indexer(method, CONFIG)
    print(f"embedding {len(chunks)} {content_mode} texts with {indexer.name}")

    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    batch_size = CONFIG.get("index_upsert_batch_size", 128)
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]
        vectors = indexer.embed_documents(batch_texts)
        collection.upsert(
            ids=[chunk["id"] for chunk in batch_chunks],
            embeddings=vectors,
            metadatas=[_metadata(chunk) for chunk in batch_chunks],
        )
        print(f"indexed {min(start + batch_size, len(chunks))}/{len(chunks)}")
    print(f"indexed {len(chunks)} chunks into {collection_name}")


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


def search_chunk_vectors(
    query: str,
    limit: int,
    method: str = "qwen",
    content_mode: str = "chunk_summary",
) -> list[ScoredChunk]:
    _validate_content_mode(content_mode)
    load_local_env()
    vector = build_indexer(method, CONFIG).embed_query(query)
    result = _collection(method, content_mode).query(
        query_embeddings=[vector],
        n_results=limit,
        include=["metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    texts = _chunk_texts(ids)
    return [
        ScoredChunk(
            id=(metadata or {}).get("id", item_id),
            score=1 - distance,
            text=texts.get((metadata or {}).get("id", item_id), ""),
        )
        for item_id, metadata, distance in zip(ids, metadatas, distances)
    ]


def search_chunk_vectors_batch(
    queries: list[str],
    limit: int,
    method: str = "qwen",
    content_mode: str = "chunk_summary",
) -> dict[int, list[ScoredChunk]]:
    _validate_content_mode(content_mode)
    load_local_env()
    vectors = build_indexer(method, CONFIG).embed_queries(queries)
    rankings = {}
    collection = _collection(method, content_mode)
    batch_size = CONFIG.get("query_batch_size", 128)
    for start in range(0, len(vectors), batch_size):
        result = collection.query(
            query_embeddings=vectors[start : start + batch_size],
            n_results=limit,
            include=["metadatas", "distances"],
        )
        batch_ids = [
            (metadata or {}).get("id", item_id)
            for ids, metadatas in zip(
                result.get("ids", []),
                result.get("metadatas", []),
            )
            for item_id, metadata in zip(ids, metadatas)
        ]
        texts = _chunk_texts(batch_ids)
        for offset, ids in enumerate(result.get("ids", [])):
            metadatas = result.get("metadatas", [])[offset]
            distances = result.get("distances", [])[offset]
            rankings[start + offset] = [
                ScoredChunk(
                    id=(metadata or {}).get("id", item_id),
                    score=1 - distance,
                    text=texts.get((metadata or {}).get("id", item_id), ""),
                )
                for item_id, metadata, distance in zip(ids, metadatas, distances)
            ]
    return rankings


def _embedding_text(chunk: dict, content_mode: str) -> str:
    prefix = [chunk.get("title") or "", chunk.get("heading_path") or ""]
    if content_mode == "chunk":
        parts = [*prefix, chunk["text"]]
    elif content_mode == "summary":
        parts = [*prefix, chunk.get("summary") or chunk["text"]]
    else:
        parts = [*prefix, chunk.get("summary") or "", chunk["text"]]
    return "\n".join(part for part in parts if part)


def _metadata(chunk: dict) -> dict:
    return {
        "id": chunk["id"],
        "page_id": chunk["page_id"],
    }


def _chunk_texts(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    with connect() as conn:
        return {
            row["id"]: row["text"]
            for row in conn.execute(
                f"select id, text from page_chunks where id in ({placeholders})",
                ids,
            )
        }


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(chroma_path()))


def _collection_name(method: str = "qwen", content_mode: str = "chunk_summary") -> str:
    return f"{CONFIG['collection_name']}_{method}_{content_mode}"


def _collection(method: str = "qwen", content_mode: str = "chunk_summary"):
    return _client().get_or_create_collection(
        _collection_name(method, content_mode),
        metadata={"hnsw:space": "cosine"},
    )


def _validate_content_mode(content_mode: str) -> None:
    if content_mode not in CONTENT_MODES:
        raise ValueError(f"Unknown content mode: {content_mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("english", "qwen", "azure"), default="qwen"
    )
    parser.add_argument(
        "--content-mode", choices=CONTENT_MODES, default="chunk_summary"
    )
    args = parser.parse_args()
    rebuild_chunk_vector_index(args.method, args.content_mode)


if __name__ == "__main__":
    main()
