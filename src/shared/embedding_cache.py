from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable

from src.shared.env import ROOT

__all__ = ["cached_embeddings"]


def cached_embeddings(
    *,
    provider: str,
    model: str,
    kind: str,
    texts: list[str],
    embed_missing: Callable[[list[str]], list[list[float]]],
) -> list[list[float]]:
    if not texts:
        return []

    path = ROOT / ".local" / "embedding_cache.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [_cache_key(provider, model, kind, text) for text in texts]

    with sqlite3.connect(path) as conn:
        _initialize(conn)
        cached = _load(conn, keys)
        missing_texts = [
            text for key, text in zip(keys, texts, strict=True) if key not in cached
        ]
        if missing_texts:
            missing_vectors = embed_missing(missing_texts)
            _store(conn, provider, model, kind, missing_texts, missing_vectors)
            cached.update(
                {
                    _cache_key(provider, model, kind, text): vector
                    for text, vector in zip(missing_texts, missing_vectors, strict=True)
                }
            )
        return [cached[key] for key in keys]


def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists embedding_cache (
            key text primary key,
            provider text not null,
            model text not null,
            kind text not null,
            text_sha256 text not null,
            vector_json text not null,
            created_at text not null default current_timestamp
        )
        """
    )


def _load(conn: sqlite3.Connection, keys: list[str]) -> dict[str, list[float]]:
    rows = {}
    for start in range(0, len(keys), 500):
        batch = keys[start : start + 500]
        placeholders = ", ".join("?" for _ in batch)
        for key, vector_json in conn.execute(
            f"select key, vector_json from embedding_cache where key in ({placeholders})",
            batch,
        ):
            rows[key] = json.loads(vector_json)
    return rows


def _store(
    conn: sqlite3.Connection,
    provider: str,
    model: str,
    kind: str,
    texts: list[str],
    vectors: list[list[float]],
) -> None:
    conn.executemany(
        """
        insert or replace into embedding_cache
            (key, provider, model, kind, text_sha256, vector_json)
        values (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                _cache_key(provider, model, kind, text),
                provider,
                model,
                kind,
                _sha256(text),
                json.dumps(vector),
            )
            for text, vector in zip(texts, vectors, strict=True)
        ],
    )


def _cache_key(provider: str, model: str, kind: str, text: str) -> str:
    return _sha256("\0".join([provider, model, kind, text]))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
