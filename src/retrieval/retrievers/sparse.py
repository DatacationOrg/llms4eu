from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from src.db.pages import load_chunk_rows
from src.indexing.chunk_text import (
    LEGACY_CHUNK_VERSION,
    PageChunk,
    chunk_text_representation,
)
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.retrieval.base import RankedChunk, retrieve_batch_default

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class SparseRetriever:
    name: str = "sparse"
    k1: float = 1.5
    b: float = 0.75
    chunk_version: str = LEGACY_CHUNK_VERSION
    variant: str = BASE_CHUNK_VARIANT
    # A geo scope's page set. Applied inside the scan rather than to the cached
    # corpus, so the corpus stays one shared object per (version, variant).
    allowed_page_ids: frozenset[str] | None = None

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        corpus = _corpus(self.chunk_version, self.variant)
        query_terms = _tokens(query)
        scores = []
        for chunk in corpus:
            if (
                self.allowed_page_ids is not None
                and chunk.page_id not in self.allowed_page_ids
            ):
                continue
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
    page_id: str
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
def _corpus(
    chunk_version: str = LEGACY_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> Corpus:
    # The cache key must carry the variant as well as the version: keyed on the
    # version alone, a second variant silently reuses the first one's corpus.
    representation = chunk_text_representation(chunk_version)
    rows = load_chunk_rows(variant)

    chunks = []
    document_frequency: Counter[str] = Counter()
    for row in rows:
        chunk = PageChunk.from_row(row)
        terms = _tokens(
            chunk.text
            if chunk_version == LEGACY_CHUNK_VERSION
            else representation.text_for_embedding(chunk)
        )
        term_counts = Counter(terms)
        chunks.append(
            ChunkTerms(
                chunk.id, chunk.page_id, chunk.text, term_counts, len(terms) or 1
            )
        )
        document_frequency.update(term_counts.keys())

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
