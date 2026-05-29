from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

from src.eval.ranking.base import RankedChunk, RankingMethod


@dataclass(frozen=True)
class CrossEncoderRerankMethod:
    name: str
    model_name: str
    base_method: RankingMethod
    candidate_limit: int
    device: str = "cuda"
    max_length: int = 512
    local_files_only: bool = True
    batch_size: int = 64
    prompt_name: str = "rag"
    prompt: str = (
        "Retrieve factual source passages that directly answer the user's question. "
        "Prefer exact evidence over topical similarity."
    )

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        candidates = self.base_method.retrieve(query, self.candidate_limit)
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
        )
        return [
            RankedChunk(id=chunk.id, score=float(score), text=chunk.text)
            for chunk, score in ranked[:limit]
        ]

    def retrieve_many(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        candidate_rankings = _retrieve_many(
            self.base_method,
            queries,
            self.candidate_limit,
        )
        model = _cross_encoder(
            self.model_name,
            self.device,
            self.max_length,
            self.local_files_only,
            self.batch_size,
            self.prompt_name,
            self.prompt,
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


def _retrieve_many(
    method: RankingMethod,
    queries: list[str],
    limit: int,
) -> dict[int, list[RankedChunk]]:
    if hasattr(method, "retrieve_many"):
        return method.retrieve_many(queries, limit)
    return {index: method.retrieve(query, limit) for index, query in enumerate(queries)}


def _rerank(
    query: str,
    candidates: list[RankedChunk],
    model_name: str,
    device: str,
    max_length: int,
    local_files_only: bool,
    batch_size: int,
    prompt_name: str,
    prompt: str,
) -> list[tuple[RankedChunk, float]]:
    scores = _cross_encoder(
        model_name,
        device,
        max_length,
        local_files_only,
        prompt_name,
        prompt,
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
    prompt_name: str,
    prompt: str,
) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,
        local_files_only=local_files_only,
        prompts={prompt_name: prompt},
        default_prompt_name=prompt_name,
    )
