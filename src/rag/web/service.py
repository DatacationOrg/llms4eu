from __future__ import annotations

from pydantic import BaseModel, Field

from src.rag.answer import answer_question


__all__ = [
    "ChatAttempt",
    "ChatReply",
    "ChatSource",
    "run_chat",
]


class ChatSource(BaseModel):
    id: str
    summary: str
    score: float


class ChatAttempt(BaseModel):
    attempt: int
    strategy: str
    embedding_model: str
    limit: int
    hits: int
    sufficient: bool
    reason: str


class ChatReply(BaseModel):
    question: str
    answer: str
    agentic: bool = False
    sufficient: bool | None = None
    sources: list[ChatSource] = Field(default_factory=list)
    attempts: list[ChatAttempt] = Field(default_factory=list)


def run_chat(
    question: str,
    *,
    limit: int | None = None,
    agentic: bool = False,
) -> ChatReply:
    result = answer_question(question, limit=limit, agentic=agentic)
    attempts = []
    if result.attempts:
        attempts = [
            ChatAttempt(
                attempt=item.state.attempt,
                strategy=item.state.strategy,
                embedding_model=item.state.embedding_model,
                limit=item.state.limit,
                hits=item.hits,
                sufficient=item.sufficient,
                reason=item.reason,
            )
            for item in result.attempts
        ]
    return ChatReply(
        question=question,
        answer=result.response.answer,
        agentic=agentic,
        sufficient=result.sufficient,
        sources=[
            ChatSource(id=place.id, summary=place.summary, score=place.score)
            for place in result.places
        ],
        attempts=attempts,
    )
