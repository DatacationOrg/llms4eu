from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.retrieval.base import RankedChunk, Retriever, retrieve_batch_default
from src.retrieval.retrievers.agentic import AgenticBatchStats, default_judge
from src.retrieval.retrievers.page_tools import (
    PageRef,
    list_sections,
    page_refs_for_chunks,
    search_in_page,
)
from src.shared.llm import StructuredLlm

TOOL_ACTIONS = (
    "sufficient",
    "reformulate",
    "expand",
    "list_sections",
    "search_in_page",
)


class ToolAction(BaseModel):
    """One step chosen by the tool-using retrieval agent."""

    action: Literal[
        "sufficient",
        "reformulate",
        "expand",
        "list_sections",
        "search_in_page",
    ] = Field(description="The next step to take.")
    reason: str = Field(default="", description="Short reason for the choice.")
    reformulated_query: str | None = Field(
        default=None,
        description="Replacement query, required when action is reformulate.",
    )
    page_id: str | None = Field(
        default=None,
        description="Page to inspect, required for list_sections and search_in_page.",
    )
    term: str | None = Field(
        default=None,
        description="Search term within the page, required for search_in_page.",
    )


@dataclass(frozen=True)
class AgenticToolRetriever:
    """Agentic retriever that can also navigate inside a retrieved page.

    On top of the reformulate/expand loop it may call `list_sections` to read a
    page's table of contents and `search_in_page` to pull sibling chunks the
    first-stage ranking missed. Chunks the agent finds are promoted directly
    after the ranked chunk from the same page, so its choices are scored.
    """

    name: str
    base_retriever: Retriever
    judge_retries: int = 3
    max_attempts: int = 4
    min_sufficient_chunks: int = 2
    initial_limit: int = 10
    limit_step: int = 5
    max_limit: int = 30
    max_tool_calls: int = 3
    section_limit: int = 40
    search_limit: int = 5
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
        chunks: list[RankedChunk] = []
        best_chunks: list[RankedChunk] = []
        promoted: list[tuple[str, RankedChunk]] = []
        observations: list[str] = []
        attempted: set[tuple[str, str, str]] = set()
        tool_calls = 0
        attempt_count = 0

        for _ in range(max(1, self.max_attempts)):
            attempt_count += 1
            retrieval_started = time.perf_counter()
            # A tool step keeps the current ranking; only a new query or a wider
            # limit needs a fresh first-stage retrieval.
            if not chunks:
                chunks = self.base_retriever.retrieve(current_query, current_limit)
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            ranked = _merge(chunks, promoted)
            if len(ranked) > len(best_chunks):
                best_chunks = ranked

            judge_started = time.perf_counter()
            action = self._next_action(current_query, chunks, promoted, observations)
            judge_ms = (time.perf_counter() - judge_started) * 1000
            self._record(
                query,
                attempt_count,
                current_query,
                ranked,
                action,
                retrieval_ms,
                judge_ms,
            )

            if action.action == "sufficient":
                self.batch_stats.record(attempt_count)
                if current_limit < limit:
                    top_up_started = time.perf_counter()
                    chunks = self.base_retriever.retrieve(current_query, limit)
                    self.batch_stats.record_top_up(
                        (time.perf_counter() - top_up_started) * 1000
                    )
                return _merge(chunks, promoted)

            if action.action in {"list_sections", "search_in_page"}:
                if tool_calls >= self.max_tool_calls:
                    observations.append(
                        "Tool budget exhausted; decide with what you have."
                    )
                    continue
                signature = (
                    action.action,
                    (action.page_id or "").strip(),
                    (action.term or "").strip().casefold(),
                )
                if signature in attempted:
                    # Repeating a call that already ran wastes the budget on an
                    # answer the agent has been given; say so instead of rerunning.
                    observations.append(
                        f"You already ran {action.action} with that page and term. "
                        "Do not repeat it: choose a different term, another page, "
                        "or a different action."
                    )
                    continue
                attempted.add(signature)
                tool_calls += 1
                observation, found = self._run_tool(action, chunks)
                observations.append(observation)
                promoted.extend(found)
                continue

            reformulated = (action.reformulated_query or "").strip()
            if action.action == "reformulate" and reformulated != current_query:
                current_query = reformulated or current_query
                observations.clear()
                chunks = []
                continue

            if current_limit < self.max_limit:
                current_limit = min(self.max_limit, current_limit + self.limit_step)
                observations.clear()
                chunks = []
                continue

            break

        self.batch_stats.record(attempt_count)
        return _merge(best_chunks, promoted)

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

    def _next_action(
        self,
        query: str,
        chunks: list[RankedChunk],
        promoted: list[tuple[str, RankedChunk]],
        observations: list[str],
    ) -> ToolAction:
        if len(chunks) + len(promoted) < self.min_sufficient_chunks:
            return ToolAction(
                action="expand",
                reason=f"Only {len(chunks)} chunks returned; more evidence required.",
            )
        pages = page_refs_for_chunks([chunk.id for chunk in chunks])
        prompt = _action_prompt(query, chunks, pages, observations)
        client = self.judge or default_judge()
        try:
            return client.structured_output(
                prompt, ToolAction, retries=self.judge_retries
            )
        except RuntimeError as exc:
            # Never let one unanswerable step end a long benchmark run.
            print(f"agentic tool judge failed, expanding instead: {exc}", flush=True)
            return ToolAction(action="expand", reason=f"judge unavailable: {exc}")

    def _run_tool(
        self,
        action: ToolAction,
        chunks: list[RankedChunk],
    ) -> tuple[str, list[tuple[str, RankedChunk]]]:
        page_id = (action.page_id or "").strip()
        if not page_id:
            return ("The tool call named no page_id; nothing was opened.", [])

        if action.action == "list_sections":
            sections = list_sections(page_id, limit=self.section_limit)
            if not sections:
                return (f"list_sections({page_id}) returned no sections.", [])
            body = "\n".join(
                f"  - {section.heading_path} ({section.chunk_count} chunks)"
                for section in sections
            )
            return (f"list_sections({page_id}):\n{body}", [])

        term = (action.term or "").strip()
        if not term:
            return ("search_in_page needs a term; nothing was searched.", [])
        hits = search_in_page(page_id, term, limit=self.search_limit)
        if not hits:
            return (f'search_in_page({page_id}, "{term}") found nothing.', [])

        known = {chunk.id for chunk in chunks}
        found = [
            (page_id, RankedChunk(id=hit.id, score=0.0, text=hit.text))
            for hit in hits
            if hit.id not in known
        ]
        body = "\n".join(
            f"  - {hit.id} [{hit.heading_path}]: {hit.text[:200]}" for hit in hits
        )
        return (f'search_in_page({page_id}, "{term}"):\n{body}', found)

    def _record(
        self,
        query: str,
        attempt: int,
        judge_query: str,
        ranked: list[RankedChunk],
        action: ToolAction,
        retrieval_ms: float,
        judge_ms: float,
    ) -> None:
        self.batch_stats.action_log.append(
            {
                "original_query": query,
                "attempt": attempt,
                "judge_query": judge_query,
                "chunks": [
                    {"id": c.id, "score": c.score, "text": c.text} for c in ranked
                ],
                # Keep the sufficiency shape so existing agentic diagnostics read it.
                "verdict": {
                    "sufficient": action.action == "sufficient",
                    "reason": action.reason,
                    "reformulated_query": action.reformulated_query,
                    "action": action.action,
                    "page_id": action.page_id,
                    "term": action.term,
                },
                "retrieval_ms": retrieval_ms,
                "judge_ms": judge_ms,
                "top_up_ms": 0.0,
            }
        )


def _merge(
    chunks: list[RankedChunk],
    promoted: list[tuple[str, RankedChunk]],
) -> list[RankedChunk]:
    """Insert agent-found chunks straight after the ranked chunk of their page.

    The anchor's rank carries the retrieval system's confidence in that page;
    a sibling the agent deliberately searched out inherits that neighbourhood
    rather than being appended where it could never affect hit@1..hit@5.
    """
    if not promoted:
        return list(chunks)

    by_page: dict[str, list[RankedChunk]] = {}
    for page_id, chunk in promoted:
        by_page.setdefault(page_id, []).append(chunk)

    pages = page_refs_for_chunks([chunk.id for chunk in chunks])
    merged: list[RankedChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.id not in seen:
            merged.append(chunk)
            seen.add(chunk.id)
        page = pages.get(chunk.id)
        if page is None:
            continue
        for extra in by_page.pop(page.page_id, []):
            if extra.id not in seen:
                merged.append(extra)
                seen.add(extra.id)
    for remaining in by_page.values():
        for extra in remaining:
            if extra.id not in seen:
                merged.append(extra)
                seen.add(extra.id)
    return merged


def _action_prompt(
    query: str,
    chunks: list[RankedChunk],
    pages: dict[str, PageRef],
    observations: list[str],
) -> str:
    shown_per_page: dict[str, int] = {}
    for chunk in chunks:
        page = pages.get(chunk.id)
        if page is not None:
            shown_per_page[page.page_id] = shown_per_page.get(page.page_id, 0) + 1

    lines = []
    for chunk in chunks:
        page = pages.get(chunk.id)
        if page is None:
            location = "page_id: unknown"
        else:
            unseen = page.chunk_count - shown_per_page.get(page.page_id, 0)
            location = (
                f"page_id: {page.page_id} | page: {page.title or page.url} | "
                f"showing {shown_per_page[page.page_id]} of {page.chunk_count} "
                f"chunks from this page ({unseen} not shown)"
            )
        lines.append(
            f"id: {chunk.id}\n{location}\nscore: {chunk.score:.4f}\ntext: {chunk.text}"
        )
    context = "\n\n".join(lines)
    history = (
        "\n\nTool results so far:\n" + "\n".join(observations) if observations else ""
    )
    return (
        "You are a retrieval agent for chunked documents.\n\n"
        "Answer this first: does one of the chunks below state the answer to the "
        "query literally, in its own text? Not 'is about the topic', not 'the page "
        "probably says it somewhere' -- the answering words are present.\n\n"
        "If yes, choose sufficient.\n"
        "If no, you must NOT choose sufficient. Choose a step that finds the "
        "missing text:\n"
        "- search_in_page (page_id, term): the answer is likely on a page you can "
        "already see, in a section that was not retrieved. Look at how many chunks "
        "of a page are not shown. Use a keyword from the query, in the language of "
        "that page, and prefer a distinctive noun or number over a common word.\n"
        "- list_sections (page_id): you suspect the right page but do not know "
        "which section to search for.\n"
        "- expand: retrieve more chunks for the same query.\n"
        "- reformulate (reformulated_query): last resort, because it throws away "
        "the pages you already found. Use it only when none of the pages shown "
        "cover the subject of the query at all.\n\n"
        f"query: {query}\n\n"
        f"chunks:\n{context}{history}"
    )
