from __future__ import annotations

from dataclasses import dataclass, field
import time

from pydantic import BaseModel, Field

from src.retrieval.base import RankedChunk, Retriever, retrieve_batch_default
from src.shared.llm import (
    AzureFoundryStructuredLlm,
    LocalOllamaStructuredLlm,
    StructuredLlm,
)

DEFAULT_JUDGE_PROVIDER = "ollama"
DEFAULT_JUDGE_MODEL = "gpt-oss:20b"
DEFAULT_JUDGE_REASONING = "low"
DEFAULT_JUDGE_NUM_CTX = 32_768
DEFAULT_JUDGE_NUM_PREDICT = 1024
DEFAULT_JUDGE_METHOD = "function_calling"


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
    action_log: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.query_attempt_counts.clear()
        self.action_log.clear()

    def record(self, attempt_count: int) -> None:
        self.query_attempt_counts.append(attempt_count)

    def record_action(
        self,
        original_query: str,
        attempt: int,
        judge_query: str,
        chunks: list[RankedChunk],
        verdict: ChunkSufficiency,
        retrieval_ms: float,
        judge_ms: float,
    ) -> None:
        self.action_log.append(
            {
                "original_query": original_query,
                "attempt": attempt,
                "judge_query": judge_query,
                "chunks": [
                    {"id": c.id, "score": c.score, "text": c.text} for c in chunks
                ],
                "verdict": verdict.model_dump(),
                "retrieval_ms": retrieval_ms,
                "judge_ms": judge_ms,
                "top_up_ms": 0.0,
            }
        )

    def record_top_up(self, elapsed_ms: float) -> None:
        if self.action_log:
            self.action_log[-1]["top_up_ms"] = elapsed_ms

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
    initial_limit: int = 10
    limit_step: int = 5
    max_limit: int = 30
    judge: StructuredLlm | None = None
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
            retrieval_started = time.perf_counter()
            chunks = self.base_retriever.retrieve(current_query, current_limit)
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            if len(chunks) > len(best_chunks):
                best_chunks = chunks

            judge_started = time.perf_counter()
            verdict = self._evaluate_sufficiency(current_query, chunks)
            judge_ms = (time.perf_counter() - judge_started) * 1000
            self.batch_stats.record_action(
                original_query=query,
                attempt=attempt_count,
                judge_query=current_query,
                chunks=chunks,
                verdict=verdict,
                retrieval_ms=retrieval_ms,
                judge_ms=judge_ms,
            )
            if verdict.sufficient:
                self.batch_stats.record(attempt_count)
                # Never return fewer than `limit` just because an early attempt
                # retrieved a smaller pool; top up so ranking metrics aren't capped.
                if current_limit < limit:
                    top_up_started = time.perf_counter()
                    chunks = self.base_retriever.retrieve(current_query, limit)
                    self.batch_stats.record_top_up(
                        (time.perf_counter() - top_up_started) * 1000
                    )
                # Preserve results beyond the requested benchmark cutoff when the
                # agent expanded its search so that expansion can be scored.
                return chunks

            reformulated = (verdict.reformulated_query or "").strip()
            if reformulated and reformulated != current_query:
                current_query = reformulated
                continue

            if current_limit < self.max_limit:
                current_limit = min(self.max_limit, current_limit + self.limit_step)
                continue

            break

        self.batch_stats.record(attempt_count)
        return best_chunks

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
        return _judge_locally(prompt, retries=self.judge_retries, judge=self.judge)


def default_judge(config: dict | None = None) -> StructuredLlm:
    """The judge every agent uses unless a retriever injects its own.

    `agentic_judge_provider` picks the backend. `azure` is the Azure AI Foundry
    deployment named by `AZURE_AI_MODEL` (DeepSeek); the `agentic_judge_model` /
    reasoning / structured-method settings are Ollama-only and ignored for it.
    `ollama` is the local judge those settings describe. Both share retries,
    the retry temperature and the circuit breaker.
    """
    config = config or {}
    provider = str(config.get("agentic_judge_provider", DEFAULT_JUDGE_PROVIDER))
    if provider == "azure":
        return AzureFoundryStructuredLlm.from_env(
            max_tokens=config.get(
                "agentic_judge_num_predict", DEFAULT_JUDGE_NUM_PREDICT
            ),
            timeout_seconds=config.get("agentic_judge_azure_timeout_seconds", 90),
            retry_temperature=config.get("agentic_judge_retry_temperature", 0.3),
            max_failure_rate=config.get("agentic_judge_max_failure_rate"),
            breaker_min_calls=config.get("agentic_judge_breaker_min_calls", 20),
        )
    if provider != "ollama":
        raise ValueError(
            f"agentic_judge_provider must be 'azure' or 'ollama', got {provider!r}"
        )
    return LocalOllamaStructuredLlm(
        model_id=config.get("agentic_judge_model", DEFAULT_JUDGE_MODEL),
        reasoning=config.get("agentic_judge_reasoning", DEFAULT_JUDGE_REASONING),
        num_ctx=config.get("agentic_judge_num_ctx", DEFAULT_JUDGE_NUM_CTX),
        num_predict=config.get("agentic_judge_num_predict", DEFAULT_JUDGE_NUM_PREDICT),
        method=config.get("agentic_judge_structured_method", DEFAULT_JUDGE_METHOD),
        retry_temperature=config.get("agentic_judge_retry_temperature", 0.3),
        keep_alive=config.get("agentic_judge_keep_alive"),
        max_failure_rate=config.get("agentic_judge_max_failure_rate"),
        breaker_min_calls=config.get("agentic_judge_breaker_min_calls", 20),
    )


def _judge_locally(
    prompt: str,
    retries: int,
    judge: StructuredLlm | None = None,
) -> ChunkSufficiency:
    client = judge or default_judge()
    try:
        return client.structured_output(prompt, ChunkSufficiency, retries=retries)
    except RuntimeError as exc:
        # A question the judge cannot answer must not end a multi-hour run:
        # resume would replay the same prompt and fail at the same place. Treat
        # it as insufficient so the agent widens its search, and record why in
        # the verdict so the action log and diagnostics show it.
        print(f"agentic judge failed, treating as insufficient: {exc}", flush=True)
        return ChunkSufficiency(
            sufficient=False,
            reason=f"judge unavailable: {exc}",
            reformulated_query=None,
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
