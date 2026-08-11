"""Direct Corpus Interaction over a BM25-bounded working set (RISE-style).

The agent never sees a vector index. BM25 selects a shortlist of at most K
pages, which are materialized as line-numbered files by `src.retrieval.workspace`,
and the agent explores them with `search`, `read` and `toc` until it can name the
chunk ids that answer the query.

Deliberate deviation from the papers: they hand the agent a general `bash()`.
This gives it three bounded tools instead. `search` shells out to ripgrep with a
fixed argument vector (never a shell string) restricted to the shortlisted
files, so an agent-authored string is always a search pattern and never a
command. The retrieval capability is the same; arbitrary command execution
driven by model output is not something a benchmark needs.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.retrieval.base import RankedChunk, Retriever, retrieve_batch_default
from src.retrieval.retrievers.agentic import AgenticBatchStats
from src.retrieval.workspace import CorpusWorkspace, PageDocument, load_workspace
from src.shared.llm import StructuredLlm

DCI_ACTIONS = ("search", "read", "toc", "answer")


class CorpusAction(BaseModel):
    """One exploration step over the working directory."""

    action: Literal["search", "read", "toc", "answer"] = Field(
        description="The next step to take."
    )
    reason: str = Field(default="", description="Short reason for the choice.")
    pattern: str | None = Field(
        default=None, description="Regex to search for, required when action is search."
    )
    path: str | None = Field(
        default=None, description="File path, required for read and toc."
    )
    start_line: int | None = Field(default=None, description="First line for read.")
    end_line: int | None = Field(default=None, description="Last line for read.")
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ids that answer the query, best first, for answer.",
    )


@dataclass(frozen=True)
class DirectCorpusRetriever:
    """Explore a bounded corpus directory instead of querying a vector index."""

    name: str
    shortlist_retriever: Retriever
    judge: StructuredLlm
    shortlist_k: int = 1000
    max_documents: int = 1000
    max_steps: int = 8
    search_limit: int = 30
    read_limit: int = 120
    judge_retries: int = 3
    workspace: CorpusWorkspace | None = None
    batch_stats: AgenticBatchStats = field(
        default_factory=AgenticBatchStats,
        init=False,
        repr=False,
        compare=False,
    )

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        workspace = self.workspace or load_workspace()
        fallback = self.shortlist_retriever.retrieve(query, self.shortlist_k)
        documents = _shortlist_documents(workspace, fallback, self.max_documents)
        if not documents:
            self.batch_stats.record(1)
            return fallback[:limit]

        texts = {chunk.id: chunk.text for chunk in fallback}
        observations: list[str] = []
        attempted: set[tuple] = set()
        cited: list[str] = []
        steps = 0

        for _ in range(max(1, self.max_steps)):
            steps += 1
            started = time.perf_counter()
            action = self._next_action(query, documents, observations)
            judge_ms = (time.perf_counter() - started) * 1000
            self._record(query, steps, action, judge_ms)

            if action.action == "answer":
                cited = [chunk_id for chunk_id in action.chunk_ids if chunk_id]
                break

            signature = _signature(action)
            if signature in attempted:
                # Without this an agent that likes one pattern spends every
                # remaining step re-running it against an unchanged corpus.
                observations.append(
                    f"You already ran that exact {action.action}. Its result is "
                    "above. Try a different pattern or file, or answer with what "
                    "you have seen."
                )
                continue
            attempted.add(signature)
            observations.append(self._run(action, workspace, documents))

        self.batch_stats.record(steps)
        return _ranking(cited, fallback, texts, workspace, limit)

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
        documents: list[PageDocument],
        observations: list[str],
    ) -> CorpusAction:
        prompt = _corpus_prompt(query, documents, observations)
        try:
            return self.judge.structured_output(
                prompt, CorpusAction, retries=self.judge_retries
            )
        except RuntimeError as exc:
            print(f"corpus agent failed, answering with shortlist: {exc}", flush=True)
            return CorpusAction(action="answer", reason=f"agent unavailable: {exc}")

    def _run(
        self,
        action: CorpusAction,
        workspace: CorpusWorkspace,
        documents: list[PageDocument],
    ) -> str:
        allowed = {document.path for document in documents}
        if action.action == "toc":
            path = (action.path or "").strip()
            if path not in allowed:
                return f"toc: {path or '(none)'} is not in the working set."
            return workspace.by_path[path].toc()

        if action.action == "read":
            path = (action.path or "").strip()
            if path not in allowed:
                return f"read: {path or '(none)'} is not in the working set."
            start = action.start_line or 1
            end = action.end_line or (start + self.read_limit - 1)
            end = min(end, start + self.read_limit - 1)
            lines = workspace.read_lines(path, start, end)
            if not lines:
                return f"read: {path} lines {start}-{end} is empty."
            body = "\n".join(f"{number:>6}  {text}" for number, text in lines)
            return f"read({path}, {start}-{end}):\n{body}"

        pattern = (action.pattern or "").strip()
        if not pattern:
            return "search: no pattern given."
        return _format_matches(
            pattern,
            _search(workspace, documents, pattern, self.search_limit),
            workspace,
        )

    def _record(
        self,
        query: str,
        step: int,
        action: CorpusAction,
        judge_ms: float,
    ) -> None:
        self.batch_stats.action_log.append(
            {
                "original_query": query,
                "attempt": step,
                "judge_query": query,
                "chunks": [],
                "verdict": {
                    "sufficient": action.action == "answer",
                    "reason": action.reason,
                    "action": action.action,
                    "pattern": action.pattern,
                    "path": action.path,
                    "chunk_ids": action.chunk_ids,
                },
                "retrieval_ms": 0.0,
                "judge_ms": judge_ms,
                "top_up_ms": 0.0,
            }
        )


def _signature(action: CorpusAction) -> tuple:
    return (
        action.action,
        (action.pattern or "").strip().casefold(),
        (action.path or "").strip(),
        action.start_line,
        action.end_line,
    )


def _shortlist_documents(
    workspace: CorpusWorkspace,
    shortlist: list[RankedChunk],
    max_documents: int,
) -> list[PageDocument]:
    """Pages behind the BM25 shortlist, in shortlist order and deduplicated.

    Capped at `max_documents`: this is the bounded interaction space, and it is
    the only thing standing between the agent and a corpus that will not fit.
    """
    documents: list[PageDocument] = []
    seen: set[str] = set()
    for chunk in shortlist:
        document = workspace.document_for_chunk(chunk.id)
        if document is not None and document.page_id not in seen:
            documents.append(document)
            seen.add(document.page_id)
            if len(documents) >= max_documents:
                break
    return documents


def _search(
    workspace: CorpusWorkspace,
    documents: list[PageDocument],
    pattern: str,
    limit: int,
) -> list[tuple[str, int, str]]:
    """Ripgrep over the shortlisted files, argv-only so the pattern stays data."""
    paths = [str(workspace.root / document.path) for document in documents]
    if not paths:
        return []
    try:
        result = subprocess.run(
            [
                "rg",
                "--no-heading",
                "--line-number",
                # rg drops the filename when given a single path, which would
                # make every one-document search parse as a miss.
                "--with-filename",
                # The workspace lives under .local/, which is gitignored.
                "--no-ignore",
                "--ignore-case",
                "--max-count",
                str(limit),
                "--regexp",
                pattern,
                *paths,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode not in (0, 1):
        return []

    matches: list[tuple[str, int, str]] = []
    root = str(workspace.root) + "/"
    for line in result.stdout.splitlines()[:limit]:
        location, _, text = line.partition(":")
        number, _, body = text.partition(":")
        if not number.isdigit():
            continue
        matches.append((location.removeprefix(root), int(number), body.strip()))
    return matches


def _format_matches(
    pattern: str,
    matches: list[tuple[str, int, str]],
    workspace: CorpusWorkspace,
) -> str:
    if not matches:
        return f'search("{pattern}"): no matches.'
    lines = []
    for path, number, body in matches:
        chunk_id = workspace.chunk_for(path, number)
        lines.append(f"  {path}:{number} [chunk {chunk_id}] {body[:200]}")
    return f'search("{pattern}"): {len(matches)} matches\n' + "\n".join(lines)


def _ranking(
    cited: list[str],
    fallback: list[RankedChunk],
    texts: dict[str, str],
    workspace: CorpusWorkspace,
    limit: int,
) -> list[RankedChunk]:
    """Agent citations first, then the BM25 shortlist as a safety net.

    The shortlist tail matters: a benchmark ranking that stops after two cited
    chunks would score worse than BM25 purely for being short.
    """
    ranked: list[RankedChunk] = []
    seen: set[str] = set()
    for position, chunk_id in enumerate(cited):
        # A hallucinated id must never enter the ranking: it would be scored as
        # a confident wrong answer rather than as the fabrication it is.
        if chunk_id in seen or workspace.document_for_chunk(chunk_id) is None:
            continue
        ranked.append(
            RankedChunk(
                id=chunk_id,
                score=float(len(cited) - position),
                text=texts.get(chunk_id, ""),
            )
        )
        seen.add(chunk_id)
    for chunk in fallback:
        if len(ranked) >= limit:
            break
        if chunk.id not in seen:
            ranked.append(RankedChunk(id=chunk.id, score=0.0, text=chunk.text))
            seen.add(chunk.id)
    return ranked[:limit]


def _corpus_prompt(
    query: str,
    documents: list[PageDocument],
    observations: list[str],
) -> str:
    listing = "\n".join(
        f"  {document.path}  {document.title or document.url} "
        f"({document.line_count} lines)"
        for document in documents[:200]
    )
    more = (
        f"\n  ... and {len(documents) - 200} more files" if len(documents) > 200 else ""
    )
    history = (
        "\n\nWhat you have done so far:\n" + "\n\n".join(observations)
        if observations
        else ""
    )
    return (
        "You are searching a corpus of Markdown files directly, the way a "
        "developer searches a codebase. There is no semantic index: you find "
        "evidence by matching text.\n\n"
        "Every chunk of every file starts with a marker line "
        "'<!-- chunk: <id> | <heading> -->'. The id on that marker is what you "
        "must return.\n\n"
        "Actions:\n"
        "- search (pattern): case-insensitive regular expression over all files "
        "below. Prefer a distinctive word, name or number from the query. Search "
        "in the language of the corpus, and try a different spelling or a "
        "shorter stem when a search returns nothing.\n"
        "- toc (path): list the chunk ids and line ranges of one file.\n"
        "- read (path, start_line, end_line): read exact lines of one file.\n"
        "- answer (chunk_ids): the chunk ids whose text answers the query, best "
        "first. Only ids you actually saw in a marker line.\n\n"
        "Answer as soon as you have seen text that states the answer. If "
        "searching keeps failing, answer with the most relevant chunk ids you "
        "have seen.\n\n"
        f"query: {query}\n\n"
        f"files ({len(documents)}):\n{listing}{more}{history}"
    )
