from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.shared.env import ROOT, load_local_env, load_yaml
from src.shared.llm import LocalOllamaStructuredLlm, StructuredLlm

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+\.md)(?:#[^)]+)?\)")
WORD_RE = re.compile(r"[^\W_]{3,}", re.UNICODE)
EXCERPT_CHUNK_CHARS = 3_000


class NavigationAction(BaseModel):
    action: Literal["open", "answer", "abstain"]
    path: str | None = Field(
        default=None, description="Advertised relative Markdown path to open"
    )
    answer: str | None = None
    citations: list[str] = Field(
        default_factory=list,
        description="Opened concept paths that support the answer",
    )
    reason: str = ""


@dataclass(frozen=True)
class OKFAnswer:
    answer: str
    citations: list[str]
    visited: list[str]
    query_count: int
    sufficient: bool
    reason: str
    trace: list[dict[str, object]] = field(default_factory=list)


def answer_question(
    question: str,
    *,
    bundle_root: Path | None = None,
    client: StructuredLlm | None = None,
) -> OKFAnswer:
    load_local_env()
    root = (bundle_root or ROOT / CONFIG["bundle_path"]).resolve()
    model = client or _llm_client()
    navigation = CONFIG["navigation"]
    current = (root / "index.md").resolve()
    if not current.exists():
        raise RuntimeError(f"OKF root index is missing: {current}")

    visited: list[Path] = []
    documents: dict[Path, str] = {}
    allowed: set[Path] = {current}
    query_count = 0
    trace: list[dict[str, object]] = []
    rejected_answers: list[str] = []
    for _ in range(navigation["max_steps"]):
        if current not in visited:
            text = current.read_text(encoding="utf-8")
            used_chars = sum(len(value) for value in documents.values())
            remaining_chars = navigation["max_context_chars"] - used_chars
            if remaining_chars <= 0:
                return _abstain(
                    visited,
                    root,
                    "Context budget exhausted before opening the selected document.",
                    query_count,
                    trace,
                )
            context = _document_context(text, question, remaining_chars)
            visited.append(current)
            documents[current] = context
            allowed.update(_advertised_paths(root, current, text))
            if len(context) < len(text):
                trace.append(
                    {
                        "kind": "document_excerpt",
                        "path": current.relative_to(root).as_posix(),
                        "source_chars": len(text),
                        "context_chars": len(context),
                    }
                )

        action = model.structured_output(
            _navigation_prompt(
                question,
                root,
                documents,
                allowed - set(visited),
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
                "citations": action.citations,
                "reason": action.reason,
            }
        )
        if action.action == "answer":
            valid_citations = _valid_citations(action.citations, documents, root)
            if not action.answer or not valid_citations:
                rejected_answers.append(
                    "The answer lacked text or cited a concept that was not opened."
                )
                continue
            return OKFAnswer(
                answer=action.answer,
                citations=valid_citations,
                visited=[path.relative_to(root).as_posix() for path in visited],
                query_count=query_count,
                sufficient=True,
                reason=action.reason,
                trace=trace,
            )
        if action.action == "abstain":
            return _abstain(
                visited,
                root,
                action.reason or "The bundle does not contain enough evidence.",
                query_count,
                trace,
            )

        target = _resolve_action_path(root, action.path)
        if target not in allowed or target in visited:
            return _abstain(
                visited,
                root,
                "The navigator selected a path that was not advertised or was already visited.",
                query_count,
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
                trace,
            )
        current = target

    return _abstain(
        visited,
        root,
        "Navigation step budget exhausted.",
        query_count,
        trace,
    )


def _llm_client() -> LocalOllamaStructuredLlm:
    return LocalOllamaStructuredLlm(
        model_id=CONFIG["model"],
        reasoning=CONFIG["model_reasoning"],
        num_ctx=CONFIG["model_num_ctx"],
        num_predict=CONFIG["model_num_predict"],
        method=CONFIG["model_structured_method"],
    )


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


def _document_context(text: str, question: str, max_chars: int) -> str:
    """Fit a concept into the context budget while retaining query evidence."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""

    chunk_chars = min(EXCERPT_CHUNK_CHARS, max(500, max_chars // 2))
    chunks = [
        text[start : start + chunk_chars] for start in range(0, len(text), chunk_chars)
    ]
    query_terms = set(WORD_RE.findall(question.casefold()))
    ranked = sorted(
        range(len(chunks)),
        key=lambda index: (
            len(query_terms.intersection(WORD_RE.findall(chunks[index].casefold()))),
            -index,
        ),
        reverse=True,
    )
    selected = {0}
    used = len(chunks[0])
    separator = "\n\n[... excerpt ...]\n\n"
    for index in ranked:
        if index in selected:
            continue
        required = len(chunks[index]) + len(separator)
        if used + required > max_chars:
            continue
        selected.add(index)
        used += required

    return separator.join(chunks[index] for index in sorted(selected))[:max_chars]


def _valid_citations(
    citations: list[str], documents: dict[Path, str], root: Path
) -> list[str]:
    opened_concepts = {
        path.relative_to(root).as_posix()
        for path in documents
        if path.name not in {"index.md", "log.md"}
    }
    return list(
        dict.fromkeys(citation for citation in citations if citation in opened_concepts)
    )


def _navigation_prompt(
    question: str,
    root: Path,
    documents: dict[Path, str],
    allowed: set[Path],
    rejected_answers: list[str],
) -> str:
    opened = "\n\n".join(
        f"FILE: {path.relative_to(root).as_posix()}\n{text}"
        for path, text in documents.items()
    )
    choices = (
        "\n".join(f"- {path.relative_to(root).as_posix()}" for path in sorted(allowed))
        or "(none)"
    )
    rejected = "\n".join(f"- {reason}" for reason in rejected_answers) or "(none)"
    return f"""You navigate an Open Knowledge Format bundle using progressive disclosure.
Indexes route to concepts. Opened concept files are complete answer evidence.
Choose exactly one action:
- open: select one path exactly from AVAILABLE PATHS;
- answer: answer only when one or more opened concept files explicitly support it, citing their exact FILE paths;
- abstain: stop when the opened concepts and available paths cannot supply explicit support.
Large concepts may be represented by query-focused excerpts. If an available
path is plausibly relevant, open it before abstaining; index descriptions are
routing hints, not evidence that the concept lacks the answer.
Keep the answer concise and in the question's language.
If an answer was rejected, open a different relevant concept instead of repeating it.

QUESTION:
{question}

AVAILABLE PATHS:
{choices}

REJECTED ANSWERS:
{rejected}

OPENED FILES:
{opened}
"""


def _abstain(
    visited: list[Path],
    root: Path,
    reason: str,
    query_count: int,
    trace: list[dict[str, object]],
) -> OKFAnswer:
    return OKFAnswer(
        answer="The OKF bundle does not contain enough evidence to answer this question.",
        citations=[],
        visited=[path.relative_to(root).as_posix() for path in visited],
        query_count=query_count,
        sufficient=False,
        reason=reason,
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
    print("\nCitations:")
    for citation in result.citations:
        print(f"- {citation}")
    print("\nAnswer:")
    print(result.answer)
    if result.reason:
        print(f"\nReason: {result.reason}")


if __name__ == "__main__":
    main()
