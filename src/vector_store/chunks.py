from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import chromadb
from tqdm import tqdm

from src.db.pages import connect_pages as connect
from src.db.pages import initialize_page_artifacts_db
from src.indexing.chunk_text import (
    CHUNK_VERSIONS,
    LEGACY_CHUNK_VERSION,
    PageChunk,
    chunk_text_representation,
)
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.shared.env import chroma_path, load_local_env, load_yaml
from src.shared.indexers import build_indexer
from src.shared.indexers import provider_names as buildable_provider_names

CONFIG = load_yaml(Path(__file__).parents[1] / "indexing" / "config.yaml")
INDEXING_PROVIDERS = tuple(CONFIG["providers"])
DEFAULT_INDEXING_PROVIDER = CONFIG["default_provider"]
DEFAULT_CHUNK_VERSION = CONFIG["default_chunk_version"]


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    score: float
    text: str


def query_chunk_vectors(
    provider: str,
    query: str,
    limit: int,
    version: str = DEFAULT_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> list[ScoredChunk]:
    _validate_provider(provider)
    _validate_version(version)
    load_local_env()
    vector = build_indexer(provider, CONFIG).embed_query(query)
    result = _existing_collection(provider, version, variant).query(
        query_embeddings=[vector],
        n_results=limit,
        include=["metadatas", "distances"],
    )
    chunk_ids = _chunk_ids_from_result(result, offset=0)
    return _scored_chunks_from_result(result, offset=0, texts=_chunk_texts(chunk_ids))


def enabled_provider_names() -> list[str]:
    """Embedding providers enabled for this project's chunk-vector indexes."""
    return sorted(INDEXING_PROVIDERS)


def query_chunk_vectors_batch(
    provider: str,
    queries: list[str],
    limit: int,
    version: str = DEFAULT_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> dict[int, list[ScoredChunk]]:
    _validate_provider(provider)
    _validate_version(version)
    load_local_env()
    vectors = build_indexer(provider, CONFIG).embed_queries(queries)
    rankings: dict[int, list[ScoredChunk]] = {}
    collection = _existing_collection(provider, version, variant)
    batch_size = CONFIG.get("query_batch_size", 128)

    for start in range(0, len(vectors), batch_size):
        result = collection.query(
            query_embeddings=vectors[start : start + batch_size],
            n_results=limit,
            include=["metadatas", "distances"],
        )
        texts = _chunk_texts(_batch_chunk_ids_from_result(result))
        for offset in range(len(result.get("ids", []))):
            rankings[start + offset] = _scored_chunks_from_result(
                result,
                offset,
                texts,
            )
    return rankings


def collection_exists(
    provider: str,
    version: str = DEFAULT_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> bool:
    _validate_provider(provider)
    _validate_version(version)
    load_local_env()
    client = _client()
    return _collection_name(provider, version, variant) in {
        collection.name for collection in client.list_collections()
    }


def collection_ready(
    provider: str,
    version: str = DEFAULT_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> bool:
    _validate_provider(provider)
    _validate_version(version)
    load_local_env()
    if not collection_exists(provider, version, variant):
        return False
    collection = _client().get_collection(_collection_name(provider, version, variant))
    # Scoped to the variant: a global chunk count would never match once a second
    # variant exists, and every readiness check would fail.
    if collection.count() != _chunk_count(variant):
        return False
    metadata = collection.metadata or {}
    # A collection built at a different sequence length holds different vectors
    # for the same chunks, and the count cannot see that. Collections written
    # before this was recorded carry no length; treat those as stale rather than
    # serve vectors whose provenance is unknown.
    configured = _configured_seq_length(provider)
    if configured is not None and metadata.get("max_seq_length") != configured:
        return False
    if version == LEGACY_CHUNK_VERSION and variant == BASE_CHUNK_VARIANT:
        return True
    return (
        metadata.get("chunk_version") == version
        and metadata.get("chunk_variant", BASE_CHUNK_VARIANT) == variant
    )


def _configured_seq_length(provider: str) -> int | None:
    length = getattr(build_indexer(provider, CONFIG), "max_seq_length", None)
    return int(length) if length else None


def rebuild_chunk_collection(
    provider: str,
    version: str = DEFAULT_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> None:
    _validate_provider(provider)
    _validate_version(version)
    load_local_env()
    initialize_page_artifacts_db()
    chunks = load_chunks(variant)
    if not chunks:
        raise RuntimeError(
            f"No {variant} chunks in SQLite. Build them first with "
            f"`uv run python -m src.preprocess.chunks --variant {variant} ...`."
        )

    client = _client()
    collection_name = _collection_name(provider, version, variant)
    with suppress(Exception):
        client.delete_collection(collection_name)
    metadata = {
        "hnsw:space": "cosine",
        "chunk_version": version,
        "chunk_variant": variant,
        "representation": CONFIG["chunk_versions"][version],
    }
    # Chroma rejects a null metadata value, and an unknown length must not be
    # recorded as though it were known.
    seq_length = _configured_seq_length(provider)
    if seq_length is not None:
        metadata["max_seq_length"] = seq_length
    collection = client.create_collection(collection_name, metadata=metadata)

    representation = chunk_text_representation(version)
    indexer = build_indexer(provider, CONFIG)
    batch_size = CONFIG.get("index_upsert_batch_size", 128)
    print(f"indexing {len(chunks)} chunks into {collection_name} with {indexer.name}")

    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc="index chunks",
        unit="batch",
    ):
        batch_chunks = chunks[start : start + batch_size]
        vectors = indexer.embed_documents(
            [representation.text_for_embedding(chunk) for chunk in batch_chunks]
        )
        collection.upsert(
            ids=[chunk.id for chunk in batch_chunks],
            embeddings=vectors,
            metadatas=[_metadata(chunk) for chunk in batch_chunks],
        )
    print(f"indexed {collection.count()} chunks into {collection_name}")


def load_chunks(variant: str = BASE_CHUNK_VARIANT) -> list[PageChunk]:
    """Load one variant's chunks from SQLite without modifying stored artifacts."""
    with connect() as conn:
        return [
            PageChunk(
                id=row["id"],
                page_id=row["page_id"],
                heading_path=row["heading_path"],
                text=row["text"],
                title=row["title"],
                chunk_index=row["chunk_index"],
                source=row["source"],
                language=row["language"],
                page_kind=row["page_kind"],
            )
            for row in conn.execute(
                """
                select c.id, c.page_id, c.chunk_index, c.heading_path, c.text,
                       m.title, m.source, m.page_kind,
                       coalesce(m.language, s.language) as language
                from page_chunks c
                join page_metadata m on m.id = c.page_id
                left join page_sources s on s.source = m.source
                where c.variant = ?
                order by c.id
                """,
                (variant,),
            )
        ]


def _metadata(chunk: PageChunk) -> dict:
    return {
        "id": chunk.id,
        "page_id": chunk.page_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title or "",
        "source": chunk.source or "",
        "language": chunk.language or "",
        "page_kind": chunk.page_kind or "",
    }


def _chunk_count(variant: str = BASE_CHUNK_VARIANT) -> int:
    with connect() as conn:
        columns = {
            row["name"] for row in conn.execute("pragma table_info(page_chunks)")
        }
        if "variant" not in columns:
            # A database that predates variants holds only base chunks. Reading a
            # readiness check should not require having migrated first.
            if variant != BASE_CHUNK_VARIANT:
                return 0
            return int(conn.execute("select count(*) from page_chunks").fetchone()[0])
        return int(
            conn.execute(
                "select count(*) from page_chunks where variant = ?", (variant,)
            ).fetchone()[0]
        )


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


def _batch_chunk_ids_from_result(result: dict) -> list[str]:
    return [
        chunk_id
        for offset in range(len(result.get("ids", [])))
        for chunk_id in _chunk_ids_from_result(result, offset)
    ]


def _chunk_ids_from_result(result: dict, offset: int) -> list[str]:
    ids = result.get("ids", [])[offset]
    metadatas = result.get("metadatas", [])[offset]
    return [
        (metadata or {}).get("id", item_id) for item_id, metadata in zip(ids, metadatas)
    ]


def _scored_chunks_from_result(
    result: dict,
    offset: int,
    texts: dict[str, str],
) -> list[ScoredChunk]:
    chunk_ids = _chunk_ids_from_result(result, offset)
    distances = result.get("distances", [])[offset]
    return [
        ScoredChunk(
            id=chunk_id,
            score=1 - distance,
            text=texts.get(chunk_id, ""),
        )
        for chunk_id, distance in zip(chunk_ids, distances)
    ]


@cache
def _client() -> chromadb.PersistentClient:
    # Cache the client: a fresh PersistentClient per query leaks connections to the
    # store and eventually fails readiness checks mid-run.
    return chromadb.PersistentClient(path=str(chroma_path()))


def _existing_collection(
    provider: str,
    version: str,
    variant: str = BASE_CHUNK_VARIANT,
):
    if not collection_ready(provider, version, variant):
        raise RuntimeError(
            f"Chunk vector collection for {provider}/{version}/{variant} is missing "
            "or stale. Run `uv run python -m src.indexing.chunks "
            f"--method {provider} --chunk-version {version} --variant {variant}`."
        )
    return _client().get_collection(_collection_name(provider, version, variant))


def _collection_name(
    provider: str,
    version: str,
    variant: str = BASE_CHUNK_VARIANT,
) -> str:
    # Empty suffixes for the legacy version and the base variant, so existing
    # collections, reports and checkpoints stay valid.
    suffix = "" if version == LEGACY_CHUNK_VERSION else f"_{version}"
    if variant != BASE_CHUNK_VARIANT:
        suffix += f"_{variant}"
    return f"{CONFIG['collection_name']}{suffix}_{provider}_chunk"


def _validate_provider(provider: str) -> None:
    if provider not in buildable_provider_names(CONFIG):
        raise ValueError(f"Unknown embedding provider: {provider}")
    if provider not in INDEXING_PROVIDERS:
        raise ValueError(f"Unknown chunk vector provider: {provider}")


def _validate_version(version: str) -> None:
    if version not in CHUNK_VERSIONS or version not in CONFIG["chunk_versions"]:
        raise ValueError(f"Unknown chunk representation version: {version}")
