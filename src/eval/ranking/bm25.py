from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from src.db.pages import connect_pages as connect
from src.eval.ranking.base import RankedChunk

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Bm25Method:
    name: str = "bm25"
    k1: float = 1.5
    b: float = 0.75

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        corpus = _corpus()
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
def _corpus() -> Corpus:
    with connect() as conn:
        rows = conn.execute("select id, text from page_chunks order by id").fetchall()

    chunks = []
    document_frequency: Counter[str] = Counter()
    for row in rows:
        terms = _tokens(row["text"])
        term_counts = Counter(terms)
        chunks.append(ChunkTerms(row["id"], row["text"], term_counts, len(terms) or 1))
        document_frequency.update(term_counts)

    document_count = len(chunks)
    idf = {
        term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
        for term, count in document_frequency.items()
    }
    avgdl = sum(chunk.length for chunk in chunks) / document_count
    return Corpus(chunks, idf, avgdl)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]
