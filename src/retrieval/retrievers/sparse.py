from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from src.db.pages import connect_pages as connect
from src.indexing.chunk_text import (
    LEGACY_CHUNK_VERSION,
    PageChunk,
    chunk_text_representation,
)
from src.retrieval.base import RankedChunk, retrieve_batch_default

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class SparseRetriever:
    name: str = "sparse"
    k1: float = 1.5
    b: float = 0.75
    chunk_version: str = LEGACY_CHUNK_VERSION

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        corpus = _corpus(self.chunk_version)
        query_terms = _tokens(query)
        scores = []
        for chunk in corpus:
            score = 0.0
            for term in query_terms:
                term_frequency = chunk.term_counts.get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * chunk.length / corpus.avgdl
                )
                score += corpus.idf.get(term, 0.0) * (
                    term_frequency * (self.k1 + 1) / denominator
                )
            if score:
                scores.append(RankedChunk(chunk.id, score, chunk.text))
        return sorted(scores, key=lambda chunk: chunk.score, reverse=True)[:limit]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        return retrieve_batch_default(self, queries, limit)


@dataclass(frozen=True)
class ChunkTerms:
    id: str
    text: str
    term_counts: Counter[str]
    length: int


@dataclass(frozen=True)
class Corpus:
    chunks: list[ChunkTerms]
    idf: dict[str, float]
    avgdl: float

    def __iter__(self):
        return iter(self.chunks)


@cache
def _corpus(chunk_version: str = LEGACY_CHUNK_VERSION) -> Corpus:
    representation = chunk_text_representation(chunk_version)
    with connect() as conn:
        rows = conn.execute(
            """
            select c.id, c.page_id, c.chunk_index, c.heading_path, c.text,
                   m.title, m.source, s.language, m.page_kind
            from page_chunks c
            join page_metadata m on m.id = c.page_id
            left join page_sources s on s.source = m.source
            order by c.id
            """
        ).fetchall()

    chunks = []
    document_frequency: Counter[str] = Counter()
    for row in rows:
        chunk = PageChunk(
            id=row["id"],
            page_id=row["page_id"],
            chunk_index=row["chunk_index"],
            heading_path=row["heading_path"],
            text=row["text"],
            title=row["title"],
            source=row["source"],
            language=row["language"],
            page_kind=row["page_kind"],
        )
        terms = _tokens(
            row["text"]
            if chunk_version == LEGACY_CHUNK_VERSION
            else representation.text_for_embedding(chunk)
        )
        term_counts = Counter(terms)
        chunks.append(ChunkTerms(row["id"], row["text"], term_counts, len(terms) or 1))
        document_frequency.update(term_counts)

    document_count = len(chunks)
    if not document_count:
        return Corpus([], {}, 1.0)

    idf = {
        term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
        for term, count in document_frequency.items()
    }
    avgdl = sum(chunk.length for chunk in chunks) / document_count
    return Corpus(chunks, idf, avgdl)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]
