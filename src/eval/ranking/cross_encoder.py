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
    prompt_name: str = "rag"
    prompt: str = (
        "Retrieve factual source passages that directly answer the user's question. "
        "Prefer exact evidence over topical similarity."
    )

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        candidates = self.base_method.retrieve(query, self.candidate_limit)
        scores = _cross_encoder(
            self.model_name,
            self.device,
            self.max_length,
            self.local_files_only,
            self.prompt_name,
            self.prompt,
        ).predict([(query, chunk.text) for chunk in candidates])
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            RankedChunk(id=chunk.id, score=float(score), text=chunk.text)
            for chunk, score in ranked[:limit]
        ]


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
