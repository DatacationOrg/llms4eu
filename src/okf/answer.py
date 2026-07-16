from __future__ import annotations

import argparse
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.okf.document import OKFDocument, OKFDocumentError
from src.okf.source import load_source_markdown
from src.okf.window import extract_windows
from src.shared.env import ROOT, load_local_env, load_yaml
from src.shared.llm import AzureFoundryStructuredLlm

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+\.md)(?:#[^)]+)?\)")

SourceReader = Callable[[str], str | None]


class NavigationAction(BaseModel):
    action: Literal["open", "open_source", "answer", "abstain"]
    path: str | None = Field(
        default=None, description="Advertised relative Markdown path to open"
    )
    page_id: str | None = Field(
        default=None,
        description="Source page id to open as raw article windows",
    )
    answer: str | None = None
    citations: list[str] = Field(
        default_factory=list,
        description="Source page ids of the raw windows that support the answer",
    )
    reason: str = ""


class EvidenceAssessment(BaseModel):
    sufficient: bool
    reason: str


@dataclass(frozen=True)
class SourceRef:
    page_id: str
    title: str
    url: str
    concept: str


@dataclass(frozen=True)
class OKFAnswer:
    answer: str
    citations: list[str]
    visited: list[str]
    query_count: int
    sufficient: bool
    reason: str
    sources: list[str] = field(default_factory=list)
    trace: list[dict[str, object]] = field(default_factory=list)


def answer_question(
    question: str,
    *,
    bundle_root: Path | None = None,
    client: AzureFoundryStructuredLlm | None = None,
    candidate_paths: Iterable[str] = (),
    source_reader: SourceReader | None = None,
) -> OKFAnswer:
    load_local_env()
    root = (bundle_root or ROOT / CONFIG["bundle_path"]).resolve()
    model = client or _azure_client()
    read_source = source_reader or _default_source_reader()
    navigation = CONFIG["navigation"]
    current = (root / "index.md").resolve()
    if not current.exists():
        raise RuntimeError(f"OKF root index is missing: {current}")

    visited: list[Path] = []
    documents: dict[Path, str] = {}
    allowed: set[Path] = {current}
    candidates = _valid_candidate_paths(root, candidate_paths)
    allowed.update(candidates)
    openable: dict[str, SourceRef] = {}
    windows: dict[str, list[str]] = {}
    opened_order: list[str] = []
    source_chars = 0
    query_count = 0
    trace: list[dict[str, object]] = []
    rejected_answers: list[str] = []
    for _ in range(navigation["max_steps"]):
        if current not in visited:
            raw_text = current.read_text(encoding="utf-8")
            if current.name == "index.md":
                context_text = raw_text
            else:
                context_text, refs = _routing_view(root, current, raw_text)
                for ref in refs:
                    openable.setdefault(ref.page_id, ref)
            if (
                sum(len(value) for value in documents.values()) + len(context_text)
                > navigation["max_context_chars"]
            ):
                return _abstain(
                    visited,
                    root,
                    "Context budget exhausted before opening the selected document.",
                    query_count,
                    opened_order,
                    trace,
                )
            visited.append(current)
            documents[current] = context_text
            allowed.update(_advertised_paths(root, current, raw_text))

        action = model.structured_output(
            _navigation_prompt(
                question,
                root,
                documents,
                allowed - set(visited),
                {
                    page_id: ref
                    for page_id, ref in openable.items()
                    if page_id not in windows
                },
                windows,
                candidates,
                rejected_answers,
            ),
            NavigationAction,
            retries=CONFIG["retries"],
        )
        query_count += 1
        trace.append(
            {
                "kind": "navigation",
                "action": action.action,
                "path": action.path,
                "page_id": action.page_id,
                "citations": action.citations,
                "reason": action.reason,
            }
        )
        if action.action == "answer":
            valid_citations = _valid_citations(action.citations, windows)
            if not action.answer or not valid_citations:
                rejected_answers.append(
                    "The answer lacked text or cited a source page that was not "
                    "opened as raw windows."
                )
                continue
            assessment = model.structured_output(
                _evidence_prompt(question, action.answer, valid_citations, windows),
                EvidenceAssessment,
                retries=CONFIG["retries"],
            )
            query_count += 1
            trace.append(
                {
                    "kind": "evidence_assessment",
                    "sufficient": assessment.sufficient,
                    "reason": assessment.reason,
                }
            )
            if assessment.sufficient:
                return OKFAnswer(
                    answer=action.answer,
                    citations=valid_citations,
                    visited=[path.relative_to(root).as_posix() for path in visited],
                    query_count=query_count,
                    sufficient=True,
                    reason=assessment.reason or action.reason,
                    sources=opened_order,
                    trace=trace,
                )
            rejected_answers.append(assessment.reason)
            continue
        if action.action == "abstain":
            return _abstain(
                visited,
                root,
                action.reason or "The bundle does not contain enough evidence.",
                query_count,
                opened_order,
                trace,
            )
        if action.action == "open_source":
            page_id = action.page_id
            if page_id not in openable or page_id in windows:
                rejected_answers.append(
                    "The navigator selected a source page that was not advertised "
                    "or was already opened."
                )
                continue
            if len(opened_order) >= navigation["max_sources"]:
                return _abstain(
                    visited,
                    root,
                    "Source page budget exhausted.",
                    query_count,
                    opened_order,
                    trace,
                )
            markdown = read_source(page_id) or ""
            extracted = extract_windows(
                markdown,
                question,
                window_chars=navigation["window_chars"],
                max_windows=navigation["max_windows_per_source"],
            )
            extracted, source_chars = _fit_source_budget(
                extracted,
                source_chars,
                navigation["max_source_context_chars"],
            )
            windows[page_id] = extracted
            opened_order.append(page_id)
            trace.append(
                {
                    "kind": "open_source",
                    "page_id": page_id,
                    "windows": len(extracted),
                }
            )
            continue
        target = _resolve_action_path(root, action.path)
        if target not in allowed or target in visited:
            return _abstain(
                visited,
                root,
                "The navigator selected a path that was not advertised or was already visited.",
                query_count,
                opened_order,
                trace,
            )
        concept_documents = [
            path for path in visited if path.name not in {"index.md", "log.md"}
        ]
        if (
            target.name not in {"index.md", "log.md"}
            and len(concept_documents) >= navigation["max_documents"]
        ):
            return _abstain(
                visited,
                root,
                "Document navigation budget exhausted.",
                query_count,
                opened_order,
                trace,
            )
        current = target

    return _abstain(
        visited,
        root,
        "Navigation step budget exhausted.",
        query_count,
        opened_order,
        trace,
    )


def _valid_candidate_paths(root: Path, values: Iterable[str]) -> set[Path]:
    candidates = set()
    for value in values:
        resolved = _resolve_action_path(root, value)
        if (
            resolved.exists()
            and root in resolved.parents
            and resolved.suffix == ".md"
            and resolved.name not in {"index.md", "log.md"}
        ):
            candidates.add(resolved)
    return candidates


def _azure_client() -> AzureFoundryStructuredLlm:
    missing = [
        name
        for name in ("AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY", "AZURE_AI_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError("Missing Azure configuration: " + ", ".join(missing))
    return AzureFoundryStructuredLlm.from_env(
        timeout_seconds=CONFIG["request_timeout_seconds"],
        max_tokens=CONFIG["max_output_tokens"],
    )


def _default_source_reader() -> SourceReader:
    db_path = ROOT / CONFIG["source_db"]

    def read(page_id: str) -> str | None:
        return load_source_markdown(db_path, [page_id]).get(page_id)

    return read


def _routing_view(
    root: Path, current: Path, raw_text: str
) -> tuple[str, list[SourceRef]]:
    concept = current.relative_to(root).with_suffix("").as_posix()
    try:
        document = OKFDocument.parse(raw_text)
    except OKFDocumentError:
        return f"CONCEPT: {concept}\n(unparseable metadata)", []
    frontmatter = document.frontmatter
    page_ids = [str(page_id) for page_id in frontmatter.get("source_page_ids", [])]
    evidence = frontmatter.get("source_evidence", {}) or {}
    refs: list[SourceRef] = []
    listing: list[str] = []
    for page_id in page_ids:
        details = evidence.get(page_id, {}) or {}
        title = str(details.get("title") or "")
        url = str(details.get("url") or "")
        refs.append(SourceRef(page_id=page_id, title=title, url=url, concept=concept))
        label = " — ".join(part for part in (title, url) if part) or "(source page)"
        listing.append(f"- [{page_id}] {label}")
    lines = [
        f"CONCEPT: {concept}",
        f"type: {frontmatter.get('type', 'Concept')}",
        f"title: {frontmatter.get('title', concept)}",
        f"description: {frontmatter.get('description', '')}",
        f"tags: {_join(frontmatter.get('tags'))}",
        f"aliases: {_join(frontmatter.get('aliases'))}",
        f"search_terms: {_join(frontmatter.get('search_terms'), limit=30)}",
        "source_pages (open with open_source to read verbatim article text):",
        *(listing or ["(none)"]),
    ]
    return "\n".join(lines), refs


def _join(values: object, *, limit: int | None = None) -> str:
    if not isinstance(values, list):
        return ""
    items = [str(value) for value in values]
    if limit is not None:
        items = items[:limit]
    return ", ".join(items)


def _fit_source_budget(
    extracted: list[str], used: int, budget: int
) -> tuple[list[str], int]:
    kept: list[str] = []
    for window in extracted:
        if used + len(window) > budget:
            break
        kept.append(window)
        used += len(window)
    return kept, used


def _advertised_paths(root: Path, current: Path, text: str) -> set[Path]:
    paths = set()
    for target in LINK_RE.findall(text):
        if target.startswith(("http://", "https://")):
            continue
        resolved = (current.parent / target).resolve()
        if (resolved == root or root in resolved.parents) and resolved.exists():
            paths.add(resolved)
    return paths


def _resolve_action_path(root: Path, value: str | None) -> Path:
    if not value:
        return root / "__invalid__"
    candidate = Path(value)
    if candidate.is_absolute():
        return root / "__invalid__"
    resolved = (root / candidate).resolve()
    if root not in resolved.parents:
        return root / "__invalid__"
    return resolved


def _valid_citations(
    citations: list[str], windows: dict[str, list[str]]
) -> list[str]:
    return list(
        dict.fromkeys(
            citation for citation in citations if windows.get(citation)
        )
    )


def _navigation_prompt(
    question: str,
    root: Path,
    documents: dict[Path, str],
    allowed: set[Path],
    openable: dict[str, SourceRef],
    windows: dict[str, list[str]],
    candidates: set[Path],
    rejected_answers: list[str],
) -> str:
    routing = "\n\n".join(
        f"FILE: {path.relative_to(root).as_posix()}\n{text}"
        for path, text in documents.items()
    )
    choices = (
        "\n".join(f"- {path.relative_to(root).as_posix()}" for path in sorted(allowed))
        or "(none)"
    )
    source_choices = (
        "\n".join(
            f"- {page_id} — "
            + (
                " — ".join(part for part in (ref.title, ref.url) if part)
                or "(source page)"
            )
            for page_id, ref in sorted(openable.items())
        )
        or "(none)"
    )
    raw_windows = (
        "\n\n".join(
            f"SOURCE PAGE: {page_id}\n" + "\n---\n".join(chunks)
            for page_id, chunks in windows.items()
            if chunks
        )
        or "(none opened yet)"
    )
    candidate_text = (
        "\n".join(
            f"- {path.relative_to(root).as_posix()}" for path in sorted(candidates)
        )
        or "(none; use the hierarchy)"
    )
    rejected = "\n".join(f"- {reason}" for reason in rejected_answers) or "(none)"
    return f"""You navigate an Open Knowledge Format bundle using progressive disclosure.
Concept metadata and indexes are ROUTING ONLY and are never valid evidence.
Answer strictly from RAW SOURCE WINDOWS, which are verbatim slices of the underlying articles.
Never answer from concept descriptions, summaries, or extracted facts.
Choose exactly one action:
- open: select one path exactly from AVAILABLE PATHS to route toward relevant concepts;
- open_source: select one page id from AVAILABLE SOURCE PAGES to read its verbatim article windows;
- answer: answer only when RAW SOURCE WINDOWS contain explicit support, citing the supporting source page ids;
- abstain: stop when no path or source page can supply explicit raw evidence.
Keep the answer concise and in the question's language.
If an answer was rejected, open a different source page or route elsewhere instead of repeating it.

QUESTION:
{question}

AVAILABLE PATHS:
{choices}

AVAILABLE SOURCE PAGES:
{source_choices}

METADATA SEARCH CANDIDATES:
{candidate_text}

REJECTED ANSWERS:
{rejected}

ROUTING METADATA READ:
{routing}

RAW SOURCE WINDOWS:
{raw_windows}
"""


def _evidence_prompt(
    question: str,
    answer: str,
    citations: list[str],
    windows: dict[str, list[str]],
) -> str:
    evidence = "\n\n".join(
        f"SOURCE PAGE: {page_id}\n" + "\n---\n".join(windows[page_id])
        for page_id in citations
    )
    return f"""Independently verify whether the cited verbatim source windows directly support the proposed answer.
These windows are raw article text, not summaries. Return sufficient=false for thematic similarity,
inference without explicit support, contradictions, or missing answer details.

QUESTION:
{question}

PROPOSED ANSWER:
{answer}

CITED SOURCE WINDOWS:
{evidence}
"""


def _abstain(
    visited: list[Path],
    root: Path,
    reason: str,
    query_count: int,
    opened_order: list[str],
    trace: list[dict[str, object]],
) -> OKFAnswer:
    return OKFAnswer(
        answer="The OKF bundle does not contain enough evidence to answer this question.",
        citations=[],
        visited=[path.relative_to(root).as_posix() for path in visited],
        query_count=query_count,
        sufficient=False,
        reason=reason,
        sources=opened_order,
        trace=trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    result = answer_question(args.question, bundle_root=args.bundle)
    print("Visited:")
    for path in result.visited:
        print(f"- {path}")
    print("\nSources opened:")
    for page_id in result.sources:
        print(f"- {page_id}")
    print("\nCitations:")
    for citation in result.citations:
        print(f"- {citation}")
    print("\nAnswer:")
    print(result.answer)
    if result.reason:
        print(f"\nReason: {result.reason}")


if __name__ == "__main__":
    main()
