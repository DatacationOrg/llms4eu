from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from src.retrieval.base import RankedChunk
from src.retrieval.retrievers.agentic import (
    AgenticRetriever,
    AgentScratch,
    default_judge,
)
from src.retrieval.retrievers.page_tools import (
    PageRef,
    list_sections,
    page_refs_for_chunks,
    search_in_page,
)

TOOL_ACTIONS = (
    "sufficient",
    "reformulate",
    "expand",
    "list_sections",
    "search_in_page",
)
PAGE_TOOLS = {"list_sections", "search_in_page"}


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

    @property
    def sufficient(self) -> bool:
        return self.action == "sufficient"

    def verdict_dict(self) -> dict:
        # Keep the sufficiency shape so existing agentic diagnostics read it.
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "reformulated_query": self.reformulated_query,
            "action": self.action,
            "page_id": self.page_id,
            "term": self.term,
        }


@dataclass(frozen=True)
class AgenticToolRetriever(AgenticRetriever):
    """Agentic retriever that can also navigate inside a retrieved page.

    On top of the reformulate/expand loop it may call `list_sections` to read a
    page's table of contents and `search_in_page` to pull sibling chunks the
    first-stage ranking missed. Chunks the agent finds are promoted directly
    after the ranked chunk from the same page, so its choices are scored.
    """

    max_attempts: int = 4
    max_tool_calls: int = 3
    section_limit: int = 40
    search_limit: int = 5

    def _rank(
        self,
        chunks: list[RankedChunk],
        scratch: AgentScratch,
    ) -> list[RankedChunk]:
        return _merge(chunks, scratch.promoted)

    def _next_step(
        self,
        query: str,
        chunks: list[RankedChunk],
        scratch: AgentScratch,
    ) -> ToolAction:
        if len(chunks) + len(scratch.promoted) < self.min_sufficient_chunks:
            return ToolAction(
                action="expand",
                reason=f"Only {len(chunks)} chunks returned; more evidence required.",
            )
        pages = page_refs_for_chunks([chunk.id for chunk in chunks])
        prompt = _action_prompt(query, chunks, pages, scratch.observations)
        client = self.judge or default_judge()
        try:
            return client.structured_output(
                prompt, ToolAction, retries=self.judge_retries
            )
        except RuntimeError as exc:
            # Never let one unanswerable step end a long benchmark run.
            print(f"agentic tool judge failed, expanding instead: {exc}", flush=True)
            return ToolAction(action="expand", reason=f"judge unavailable: {exc}")

    def _take_tool_step(
        self,
        step: ToolAction,
        chunks: list[RankedChunk],
        scratch: AgentScratch,
    ) -> bool:
        if step.action not in PAGE_TOOLS:
            return False
        if scratch.tool_calls >= self.max_tool_calls:
            scratch.observations.append(
                "Tool budget exhausted; decide with what you have."
            )
            return True
        signature = (
            step.action,
            (step.page_id or "").strip(),
            (step.term or "").strip().casefold(),
        )
        if signature in scratch.attempted:
            # Repeating a call that already ran wastes the budget on an
            # answer the agent has been given; say so instead of rerunning.
            scratch.observations.append(
                f"You already ran {step.action} with that page and term. "
                "Do not repeat it: choose a different term, another page, "
                "or a different action."
            )
            return True
        scratch.attempted.add(signature)
        scratch.tool_calls += 1
        observation, found = self._run_tool(step, chunks)
        scratch.observations.append(observation)
        scratch.promoted.extend(found)
        return True

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
