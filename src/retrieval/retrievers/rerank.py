from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from src.retrieval.base import RankedChunk, Retriever

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder import CrossEncoder


@dataclass(frozen=True)
class CrossEncoderRerankRetriever:
    """Wrap a first-stage retriever with a cross-encoder reranker."""

    name: str
    model_name: str
    base_retriever: Retriever
    candidate_limit: int
    device: str
    max_length: int
    local_files_only: bool
    batch_size: int
    prompt_name: str | None = None
    prompt: str | None = None
    trust_remote_code: bool = False
    # CrossEncoder otherwise loads in float32 whatever the checkpoint declares,
    # which quadruples a bf16 reranker's residency: 16 GB for a 4B model, beside
    # a 16 GB embedder, on one 48 GB card. Left None for the 0.6B default so its
    # published numbers keep their exact arithmetic.
    dtype: str | None = None

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        candidates = self.base_retriever.retrieve(query, self.candidate_limit)
        ranked = _rerank(
            query,
            candidates,
            self.model_name,
            self.device,
            self.max_length,
            self.local_files_only,
            self.batch_size,
            self.prompt_name,
            self.prompt,
            self.trust_remote_code,
            self.dtype,
        )
        return [
            RankedChunk(id=chunk.id, score=float(score), text=chunk.text)
            for chunk, score in ranked[:limit]
        ]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        candidate_rankings = self.base_retriever.retrieve_batch(
            queries,
            self.candidate_limit,
        )
        model = _cross_encoder(
            self.model_name,
            self.device,
            self.max_length,
            self.local_files_only,
            self.prompt_name,
            self.prompt,
            self.trust_remote_code,
            self.dtype,
        )
        pairs = []
        refs = []
        for query_index, candidates in candidate_rankings.items():
            for chunk in candidates:
                pairs.append((queries[query_index], chunk.text))
                refs.append((query_index, chunk))

        scores = model.predict(pairs, batch_size=self.batch_size)
        by_query: dict[int, list[tuple[RankedChunk, float]]] = {
            index: [] for index in range(len(queries))
        }
        for (query_index, chunk), score in zip(refs, scores, strict=True):
            by_query[query_index].append((chunk, float(score)))

        return {
            query_index: [
                RankedChunk(id=chunk.id, score=score, text=chunk.text)
                for chunk, score in sorted(
                    ranked,
                    key=lambda item: item[1],
                    reverse=True,
                )[:limit]
            ]
            for query_index, ranked in by_query.items()
        }


def _rerank(
    query: str,
    candidates: list[RankedChunk],
    model_name: str,
    device: str,
    max_length: int,
    local_files_only: bool,
    batch_size: int,
    prompt_name: str | None,
    prompt: str | None,
    trust_remote_code: bool = False,
    dtype: str | None = None,
) -> list[tuple[RankedChunk, float]]:
    scores = _cross_encoder(
        model_name,
        device,
        max_length,
        local_files_only,
        prompt_name,
        prompt,
        trust_remote_code,
        dtype,
    ).predict(
        [(query, chunk.text) for chunk in candidates],
        batch_size=batch_size,
    )
    return sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )


@cache
def _cross_encoder(
    model_name: str,
    device: str,
    max_length: int,
    local_files_only: bool,
    prompt_name: str | None,
    prompt: str | None,
    trust_remote_code: bool = False,
    dtype: str | None = None,
) -> CrossEncoder:
    import torch
    from sentence_transformers import CrossEncoder

    prompt_kwargs = _prompt_kwargs(prompt_name, prompt)
    model_kwargs = {} if dtype is None else {"dtype": getattr(torch, dtype)}
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            **prompt_kwargs,
        )


def _prompt_kwargs(prompt_name: str | None, prompt: str | None) -> dict:
    if not prompt_name or not prompt:
        return {}
    return {
        "prompts": {prompt_name: prompt},
        "default_prompt_name": prompt_name,
    }
