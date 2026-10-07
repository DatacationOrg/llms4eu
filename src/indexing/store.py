"""Query-time vector search over the stored chunk embeddings
(`embeddings/<provider>/<size>.npy`, written by `python -m src.indexing`), for the
CHUNK_SIZE in use. Vectors are unit length, so cosine is a dot product.

ponytail: exact search, linear in the chunk count (about a second per 64 queries at
500k chunks); an ANN index built from the same files is the follow-up when latency matters.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from src.db.dataset import Chunk, chunk_size, embeddings, load, missing, vectors_path
from src.indexing.embedders import build_indexer
from src.retrieval.base import RankedChunk, top_k
from src.shared.env import load_local_env, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
INDEXING_PROVIDERS = tuple(CONFIG["providers"])
DEFAULT_INDEXING_PROVIDER = CONFIG["default_provider"]


def enabled_provider_names() -> list[str]:
    """Embedding providers enabled for this project's chunk-vector indexes."""
    return sorted(INDEXING_PROVIDERS)


def query_chunk_vectors(provider: str, query: str, limit: int) -> list[RankedChunk]:
    return query_chunk_vectors_batch(provider, [query], limit)[0]


def query_chunk_vectors_batch(
    provider: str,
    queries: list[str],
    limit: int,
) -> dict[int, list[RankedChunk]]:
    _validate_provider(provider)
    load_local_env()
    if not collection_ready(provider):
        raise RuntimeError(
            f"Chunk vectors for {provider} at size {chunk_size()} are missing or "
            f"incomplete. Run `uv run python -m src.indexing --method {provider}`."
        )
    ids, matrix, texts = _index(provider, chunk_size())
    vectors = np.asarray(
        build_indexer(provider, CONFIG).embed_queries(queries), np.float32
    )
    rankings: dict[int, list[RankedChunk]] = {}
    for start in range(0, len(vectors), CONFIG["query_batch_size"]):
        scores = vectors[start : start + CONFIG["query_batch_size"]] @ matrix.T
        for offset, row in enumerate(scores):
            rankings[start + offset] = [
                RankedChunk(id=ids[i], score=float(row[i]), text=texts[i])
                for i in top_k(row, limit)
            ]
    return rankings


def collection_ready(provider: str) -> bool:
    """Every chunk of the size in use has a vector from this provider."""
    _validate_provider(provider)
    file = vectors_path(provider, chunk_size())
    if not file.exists():
        return False
    vectors = np.load(file, mmap_mode="r")
    rows = load(Chunk, ["id"], size=chunk_size()).num_rows
    return len(vectors) == rows and not missing(vectors).any()


@cache
def _index(provider: str, size: int) -> tuple[list[str], np.ndarray, list[str]]:
    ids, vectors = embeddings(provider, size)
    texts = load(Chunk, ["text"], size=size).column("text").to_pylist()
    return ids, np.asarray(vectors, np.float32), texts


def _validate_provider(provider: str) -> None:
    if provider not in INDEXING_PROVIDERS:
        raise ValueError(f"Unknown chunk vector provider: {provider}")
