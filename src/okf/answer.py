from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from src.okf.paths import concept_id_for
from src.shared.env import ROOT, load_local_env, load_yaml
from src.shared.llm import AzureFoundryStructuredLlm

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+\.md)(?:#[^)]+)?\)")


class NavigationAction(BaseModel):
    action: Literal["open", "answer", "abstain"]
    path: str | None = Field(
        default=None, description="Advertised relative Markdown path to open"
    )
    answer: str | None = None
    citations: list[str] = Field(
        default_factory=list,
        description="Bundle-relative concept ids without .md",
    )
    reason: str = ""


class EvidenceAssessment(BaseModel):
    sufficient: bool
    reason: str


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
    client: AzureFoundryStructuredLlm | None = None,
    candidate_paths: Iterable[str] = (),
) -> OKFAnswer:
    load_local_env()
    root = (bundle_root or ROOT / CONFIG["bundle_path"]).resolve()
    model = client or _azure_client()
    navigation = CONFIG["navigation"]
    current = (root / "index.md").resolve()
    if not current.exists():
        raise RuntimeError(f"OKF root index is missing: {current}")

    visited: list[Path] = []
    documents: dict[Path, str] = {}
    allowed: set[Path] = {current}
    candidates = _valid_candidate_paths(root, candidate_paths)
    allowed.update(candidates)
    query_count = 0
    trace: list[dict[str, object]] = []
    rejected_answers: list[str] = []
    for _ in range(navigation["max_steps"]):
        if current not in visited:
            text = current.read_text(encoding="utf-8")
            if (
                sum(len(value) for value in documents.values()) + len(text)
                > navigation["max_context_chars"]
            ):
                return _abstain(
                    visited,
                    root,
                    "Context budget exhausted before opening the selected document.",
                    query_count,
                    trace,
                )
            visited.append(current)
            documents[current] = text
            allowed.update(_advertised_paths(root, current, text))

        action = model.structured_output(
            _navigation_prompt(
                question,
                root,
                documents,
                allowed - set(visited),
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
                "citations": action.citations,
                "reason": action.reason,
            }
        )
        if action.action == "answer":
            valid_citations = _valid_citations(action.citations, visited, root)
            if not action.answer or not valid_citations:
                rejected_answers.append(
                    "The proposed answer lacked an answer or valid concept citations."
                )
                continue
            assessment = model.structured_output(
                _evidence_prompt(
                    question,
                    action.answer,
                    valid_citations,
                    root,
                    documents,
                ),
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
                    trace=trace,
                )
            rejected_answers.append(assessment.reason)
            if not allowed - set(visited):
                return _abstain(
                    visited,
                    root,
                    assessment.reason,
                    query_count,
                    trace,
                )
            continue
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
    citations: list[str], visited: list[Path], root: Path
) -> list[str]:
    visited_ids = {
        concept_id_for(root, path)
        for path in visited
        if path.name not in {"index.md", "log.md"}
    }
    return list(
        dict.fromkeys(citation for citation in citations if citation in visited_ids)
    )


def _navigation_prompt(
    question: str,
    root: Path,
    documents: dict[Path, str],
    allowed: set[Path],
    candidates: set[Path],
    rejected_answers: list[str],
) -> str:
    context = "\n\n".join(
        f"FILE: {path.relative_to(root).as_posix()}\n{text}"
        for path, text in documents.items()
    )
    choices = (
        "\n".join(f"- {path.relative_to(root).as_posix()}" for path in sorted(allowed))
        or "(none)"
    )
    candidate_text = (
        "\n".join(
            f"- {path.relative_to(root).as_posix()}" for path in sorted(candidates)
        )
        or "(none; use the hierarchy)"
    )
    rejected = "\n".join(f"- {reason}" for reason in rejected_answers) or "(none)"
    return f"""You navigate an Open Knowledge Format bundle using progressive disclosure.
Answer only from complete concept documents already shown. Index files advertise documents but are not factual evidence.
Choose exactly one action:
- open: select one path exactly from AVAILABLE PATHS that is likely to contain evidence;
- answer: answer when complete concept documents contain enough evidence, citing their bundle-relative concept ids without `.md`;
- abstain: stop when no advertised path is useful or evidence is insufficient.
Never cite an index file. Keep the answer concise and in the question's language.
If an answer was rejected, backtrack to the best unvisited alternative instead of repeating it.

QUESTION:
{question}

AVAILABLE PATHS:
{choices}

METADATA SEARCH CANDIDATES:
{candidate_text}

REJECTED ANSWERS:
{rejected}

FILES READ:
{context}
"""


def _evidence_prompt(
    question: str,
    answer: str,
    citations: list[str],
    root: Path,
    documents: dict[Path, str],
) -> str:
    cited = set(citations)
    evidence = "\n\n".join(
        f"CONCEPT: {concept_id_for(root, path)}\n{text}"
        for path, text in documents.items()
        if path.name not in {"index.md", "log.md"}
        and concept_id_for(root, path) in cited
    )
    return f"""Independently verify whether the cited complete OKF concepts directly support the proposed answer.
Return sufficient=false for thematic similarity, inference without explicit support, contradictions, or missing answer details.

QUESTION:
{question}

PROPOSED ANSWER:
{answer}

CITED CONCEPT EVIDENCE:
{evidence}
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
