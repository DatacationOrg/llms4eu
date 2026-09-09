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
from src.preprocess.chunks import BASE_CHUNK_VARIANT
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
    # A cited page id is expanded to at most this many of that page's chunks.
    # The agent naming a page is real evidence -- it read something there -- but
    # a page can hold hundreds of chunks and promoting all of them would push the
    # shortlist out of the ranking entirely and score as a flood, not a find.
    page_expansion_limit: int = 3
    # How many times an `answer` carrying nothing usable is sent back before its
    # emptiness is accepted. Bounded because a model that answers empty twice is
    # not going to be talked round on the third ask.
    max_answer_retries: int = 2
    workspace: CorpusWorkspace | None = None
    variant: str = BASE_CHUNK_VARIANT
    batch_stats: AgenticBatchStats = field(
        default_factory=AgenticBatchStats,
        init=False,
        repr=False,
        compare=False,
    )

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        workspace = self.workspace or load_workspace(variant=self.variant)
        fallback = self.shortlist_retriever.retrieve(query, self.shortlist_k)
        documents = _shortlist_documents(workspace, fallback, self.max_documents)
        if not documents:
            self.batch_stats.record(1)
            return fallback[:limit]

        texts = {chunk.id: chunk.text for chunk in fallback}
        observations: list[str] = []
        attempted: set[tuple] = set()
        cited: list[str] = []
        answer_retries = 0
        steps = 0

        for _ in range(max(1, self.max_steps)):
            steps += 1
            started = time.perf_counter()
            action = self._next_action(query, documents, observations)
            judge_ms = (time.perf_counter() - started) * 1000
            self._record(query, steps, action, judge_ms)

            if action.action == "answer":
                raw = [chunk_id for chunk_id in action.chunk_ids if chunk_id]
                cited, report = _resolve_citations(
                    raw, workspace, fallback, self.page_expansion_limit
                )
                self._record_resolution(report)
                # An answer that resolves to nothing is the failure this guards:
                # `_ranking` would drop every id and the run would silently score
                # the BM25 shortlist while wearing this method's name. Say what
                # was wrong and spend a step asking again instead.
                if not cited and answer_retries < self.max_answer_retries:
                    answer_retries += 1
                    observations.append(_answer_rejection(report))
                    continue
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

    def _record_resolution(self, report: dict) -> None:
        """Attach citation resolution to the answer step, and say so out loud.

        Silence is what made this bug survive a smoke test: `_ranking` dropped
        every unusable id without a word, so a run that scored BM25 looked
        exactly like a run where the agent had chosen BM25's chunks.
        """
        if self.batch_stats.action_log:
            self.batch_stats.action_log[-1]["citation_resolution"] = report
        if report["dropped"]:
            print(
                f"dci dropped {len(report['dropped'])} unresolvable citation(s): "
                f"{', '.join(report['dropped'][:5])}",
                flush=True,
            )
        if report["expanded"]:
            print(
                f"dci expanded {len(report['expanded'])} cited page id(s) to "
                f"{sum(len(v) for v in report['expanded'].values())} chunk(s)",
                flush=True,
            )


def _signature(action: CorpusAction) -> tuple:
    """Identity of a step, over only the fields that step actually uses.

    Including every field made the guard useless: `_run` ignores `path` for a
    search, so a model that emitted a stray path alongside an unchanged pattern
    produced a fresh signature each time and re-ran the identical ripgrep. One
    observed question spent three of its eight steps searching "Slovenia".
    """
    if action.action == "search":
        return ("search", (action.pattern or "").strip().casefold())
    if action.action == "toc":
        return ("toc", (action.path or "").strip())
    if action.action == "read":
        return (
            "read",
            (action.path or "").strip(),
            action.start_line,
            action.end_line,
        )
    return (action.action,)


def _resolve_citations(
    cited: list[str],
    workspace: CorpusWorkspace,
    fallback: list[RankedChunk],
    page_expansion_limit: int,
) -> tuple[list[str], dict]:
    """Turn what the agent named into chunk ids, and report what could not be.

    The agent is asked for chunk ids and frequently returns page ids instead:
    the workspace names each file after its page, so the listing it reads is full
    of bare page ids that look exactly like a chunk id minus the `:index`. The
    old code passed those to `_ranking`, which dropped them as fabrications and
    fell back to BM25 -- measured over 25 questions, 13 of 27 citations were page
    ids and 15 questions ended with nothing usable, so the method scored BM25
    under its own name.

    A page id is not a fabrication, it is a coarser answer, so it resolves to
    that page's chunks: the ones the shortlist already ranked first, in shortlist
    order, then document order for any the shortlist never returned. Anything
    that is neither a chunk nor a page stays dropped -- that is a real
    fabrication and must not enter the ranking.
    """
    exact: list[str] = []
    expanded: dict[str, list[str]] = {}
    dropped: list[str] = []
    resolved: list[str] = []
    seen: set[str] = set()

    shortlist_by_page: dict[str, list[str]] = {}
    for chunk in fallback:
        document = workspace.document_for_chunk(chunk.id)
        if document is not None:
            shortlist_by_page.setdefault(document.page_id, []).append(chunk.id)

    for chunk_id in cited:
        if workspace.document_for_chunk(chunk_id) is not None:
            exact.append(chunk_id)
            if chunk_id not in seen:
                resolved.append(chunk_id)
                seen.add(chunk_id)
            continue

        document = workspace.by_page_id.get(chunk_id)
        if document is None:
            dropped.append(chunk_id)
            continue

        ranked = list(shortlist_by_page.get(chunk_id, []))
        ranked_set = set(ranked)
        ranked.extend(
            span.chunk_id for span in document.spans if span.chunk_id not in ranked_set
        )
        promoted = ranked[: max(0, page_expansion_limit)]
        expanded[chunk_id] = promoted
        for promoted_id in promoted:
            if promoted_id not in seen:
                resolved.append(promoted_id)
                seen.add(promoted_id)

    return resolved, {
        "cited": list(cited),
        "exact": exact,
        "expanded": expanded,
        "dropped": dropped,
    }


def _answer_rejection(report: dict) -> str:
    """What to tell an agent whose answer resolved to no scoreable chunk."""
    if report["dropped"]:
        names = ", ".join(report["dropped"][:5])
        return (
            f"Your answer named {names}, which match no chunk and no file in the "
            "working set. A chunk id is the exact string on a "
            "'<!-- chunk: <id> | ... -->' marker line. Search or read until you "
            "see a marker line, then answer with the id written on it."
        )
    return (
        "Your answer carried no chunk ids at all, so nothing can be returned. "
        "Answer again with the ids from the marker lines of the most relevant "
        "text you have seen, best first, even if you are not certain."
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
        "A file name is NOT a chunk id. Files below are named after their page, "
        "so 'wikipedia/abc123.md' is a page and 'abc123' on its own identifies "
        "nothing you can return. The chunk id is longer than the file's name: it "
        "is written in full on the marker line, and it ends with a colon and a "
        "number, like 'abc123:7'. Copy it from the marker, do not build it from "
        "a file name.\n\n"
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
