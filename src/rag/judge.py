from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.rag.search import ScoredPlace
from src.shared.llm import structured_local_model


__all__ = ["ChunkSufficiency", "SufficiencyJudge"]


class ChunkSufficiency(BaseModel):
    sufficient: bool = Field(
        description="True only when the context is enough to answer the question."
    )
    reason: str = Field(description="A brief explanation for the sufficiency decision.")
    reformulated_query: str | None = Field(
        default=None,
        description="Optional broader or clearer query when context is insufficient.",
    )


@dataclass(frozen=True)
class SufficiencyJudge:
    model: str
    min_sufficient_places: int = 2

    def evaluate(self, question: str, places: list[ScoredPlace]) -> ChunkSufficiency:
        if len(places) < self.min_sufficient_places:
            return ChunkSufficiency(
                sufficient=False,
                reason=(
                    f"Only {len(places)} retrieved places; "
                    "more evidence is required before answering."
                ),
                reformulated_query=question,
            )

        prompt = _prompt_for(question, places)
        return structured_local_model(self.model, ChunkSufficiency).invoke(prompt)


def _prompt_for(question: str, places: list[ScoredPlace]) -> str:
    context = "\n\n".join(place.context_text for place in places)
    return (
        "You are a retrieval sufficiency judge.\n"
        "Decide whether the provided context is enough to answer the question.\n"
        "Rules:\n"
        "1) sufficient=true only if key facts needed for a grounded answer are present.\n"
        "2) If insufficient, explain what is missing and propose a better query.\n"
        "3) Keep reason concise (one sentence).\n\n"
        f"question: {question}\n\n"
        f"context:\n{context}"
    )
