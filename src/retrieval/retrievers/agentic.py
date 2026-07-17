from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.retrieval.base import RankedChunk, Retriever, retrieve_batch_default
from src.shared.llm import AzureFoundryStructuredLlm


class ChunkSufficiency(BaseModel):
    sufficient: bool = Field(
        description="True only if retrieved chunks are enough to answer the query."
    )
    reason: str = Field(description="Short reason for the verdict.")
    reformulated_query: str | None = Field(
        default=None,
        description="Optional replacement query when context is insufficient.",
    )


@dataclass
class AgenticBatchStats:
    query_attempt_counts: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.query_attempt_counts.clear()

    def record(self, attempt_count: int) -> None:
        self.query_attempt_counts.append(attempt_count)

    def total_queries(self) -> int:
        return sum(self.query_attempt_counts)

    def average_queries_per_question(self) -> float:
        if not self.query_attempt_counts:
            return 0.0
        return self.total_queries() / len(self.query_attempt_counts)


@dataclass(frozen=True)
class AgenticRetriever:
    """Retry a base retriever until chunk context is judged sufficient."""

    name: str
    base_retriever: Retriever
    judge_retries: int = 3
    max_attempts: int = 3
    min_sufficient_chunks: int = 2
    initial_limit: int = 5
    limit_step: int = 5
    max_limit: int = 30
    batch_stats: AgenticBatchStats = field(
        default_factory=AgenticBatchStats,
        init=False,
        repr=False,
        compare=False,
    )

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        current_query = query
        current_limit = min(limit, self.initial_limit)
        best_chunks: list[RankedChunk] = []
        attempt_count = 0

        for _ in range(max(1, self.max_attempts)):
            attempt_count += 1
            chunks = self.base_retriever.retrieve(current_query, current_limit)
            if len(chunks) > len(best_chunks):
                best_chunks = chunks

            verdict = self._evaluate_sufficiency(current_query, chunks)
            if verdict.sufficient:
                self.batch_stats.record(attempt_count)
                # Never return fewer than `limit` just because an early attempt
                # retrieved a smaller pool; top up so ranking metrics aren't capped.
                if current_limit < limit:
                    chunks = self.base_retriever.retrieve(current_query, limit)
                return chunks[:limit]

            reformulated = (verdict.reformulated_query or "").strip()
            if reformulated and reformulated != current_query:
                current_query = reformulated
                continue

            if current_limit < self.max_limit:
                current_limit = min(self.max_limit, current_limit + self.limit_step)
                continue

            break

        self.batch_stats.record(attempt_count)
        return best_chunks[:limit]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        self.batch_stats.reset()
        return retrieve_batch_default(self, queries, limit)

    def average_queries_per_question(self) -> float:
        return self.batch_stats.average_queries_per_question()

    def total_queries(self) -> int:
        return self.batch_stats.total_queries()

    def _evaluate_sufficiency(
        self,
        query: str,
        chunks: list[RankedChunk],
    ) -> ChunkSufficiency:
        if len(chunks) < self.min_sufficient_chunks:
            return ChunkSufficiency(
                sufficient=False,
                reason=(f"Only {len(chunks)} chunks returned; more evidence required."),
                reformulated_query=query,
            )

        prompt = _sufficiency_prompt(query, chunks)
        return _judge_with_azure(prompt, retries=self.judge_retries)


def _judge_with_azure(prompt: str, retries: int) -> ChunkSufficiency:
    return AzureFoundryStructuredLlm.from_env().structured_output(
        prompt,
        ChunkSufficiency,
        retries=retries,
    )


def _sufficiency_prompt(query: str, chunks: list[RankedChunk]) -> str:
    context = "\n\n".join(
        f"id: {chunk.id}\nscore: {chunk.score:.4f}\ntext: {chunk.text}"
        for chunk in chunks
    )
    return (
        "You are a retrieval sufficiency judge for chunked documents.\n"
        "Decide if the chunks below contain enough grounded facts to answer the query.\n"
        "Return sufficient=true only if the available chunks are enough.\n"
        "If insufficient, provide a concise reason and an improved reformulated_query.\n\n"
        f"query: {query}\n\n"
        f"chunks:\n{context}"
    )
