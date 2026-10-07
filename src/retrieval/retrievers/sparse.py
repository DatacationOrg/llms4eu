from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from src.db.dataset import Chunk, chunk_size, load
from src.retrieval.base import RankedChunk, top_k
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).parents[1] / "config.yaml")
WORD_PATTERN = r"[^\W_]+"
QUERY_BATCH = 64  # queries scored at once: each is a dense row over every chunk


@dataclass(frozen=True)
class SparseRetriever:
    name: str = "sparse"
    k1: float = CONFIG["sparse_k1"]
    b: float = CONFIG["sparse_b"]

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        return self.retrieve_batch([query], limit)[0]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        corpus = _corpus(chunk_size(), self.k1, self.b)
        results = {}
        for start in range(0, len(queries), QUERY_BATCH):
            # A query term counts as often as it occurs in the query.
            terms = corpus.vectorizer.transform(queries[start : start + QUERY_BATCH])
            for offset, row in enumerate((terms @ corpus.weights.T).toarray()):
                results[start + offset] = [
                    RankedChunk(corpus.ids[i], float(row[i]), corpus.texts[i])
                    for i in top_k(row, limit)
                    if row[i] > 0
                ]
        return results


@dataclass(frozen=True)
class Corpus:
    ids: list[str]
    texts: list[str]
    vectorizer: CountVectorizer
    weights: csr_matrix  # chunks x terms, BM25 weight of each term in each chunk


@cache
def _corpus(size: int, k1: float, b: float) -> Corpus:
    # BM25 indexes the raw chunk text; title and breadcrumbs are dense-side context.
    chunks = load(Chunk, ["id", "text"], size=size).to_pydict()
    vectorizer = CountVectorizer(
        lowercase=False, preprocessor=str.casefold, token_pattern=WORD_PATTERN
    )
    counts = csr_matrix(vectorizer.fit_transform(chunks["text"]), dtype=np.float32)
    lengths = np.maximum(np.asarray(counts.sum(axis=1)).ravel(), 1)
    document_count = len(chunks["id"])
    document_frequency = np.bincount(
        counts.indices, minlength=len(vectorizer.vocabulary_)
    )
    idf = np.log(
        1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    norm = k1 * (1 - b + b * lengths / lengths.mean())
    rows = np.repeat(np.arange(document_count), np.diff(counts.indptr))
    tf = counts.data
    counts.data = idf[counts.indices] * tf * (k1 + 1) / (tf + norm[rows])
    return Corpus(chunks["id"], chunks["text"], vectorizer, counts)
