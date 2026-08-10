from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.db.pages import connect_pages
from src.eval.metrics import plain_table
from src.okf.answer import answer_question
from src.okf.evidence import (
    invert_page_map,
    load_bundle_page_map,
    project_pages_to_concepts,
)
from src.retrieval.methods import build_retriever, ensure_retrievers_ready
from src.shared.env import ROOT, load_local_env

DEFAULT_METHODS = ("qwen_hybrid", "qwen_hybrid_agentic")
DEFAULT_BUNDLE = ROOT / "data/okf/tourism"
REPORTS_DIR = ROOT / ".local/reports"


@dataclass(frozen=True)
class EvidenceRun:
    name: str
    seconds: float
    query_count: int
    rankings: dict[str, list[str]]
    failures: int = 0


def load_covered_questions(
    page_ids: set[str], limit: int | None = None
) -> tuple[list[dict], dict[str, set[str]], dict[str, str]]:
    if not page_ids:
        return [], {}, {}
    placeholders = ", ".join("?" for _ in page_ids)
    with connect_pages() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                select distinct q.id, q.question, q.answer
                from eval_questions q
                join eval_relevant_chunks r on r.question_id = q.id
                join page_chunks c on c.id = r.chunk_id
                where q.approved = 1 and c.page_id in ({placeholders})
                order by q.id
                """,
                sorted(page_ids),
            )
        ]
        if limit is not None:
            rows = rows[:limit]
        question_ids = [row["id"] for row in rows]
        if not question_ids:
            return [], {}, {}
        question_placeholders = ", ".join("?" for _ in question_ids)
        relevance: dict[str, set[str]] = {
            question_id: set() for question_id in question_ids
        }
        for row in conn.execute(
            f"""
            select r.question_id, c.page_id
            from eval_relevant_chunks r
            join page_chunks c on c.id = r.chunk_id
            where r.question_id in ({question_placeholders})
            """,
            question_ids,
        ):
            relevance[row["question_id"]].add(row["page_id"])
        chunk_pages = {
            row["id"]: row["page_id"]
            for row in conn.execute("select id, page_id from page_chunks")
        }
    return rows, relevance, chunk_pages


def run_rag(
    name: str,
    questions: list[dict],
    chunk_pages: dict[str, str],
    page_concepts: dict[str, list[str]],
) -> EvidenceRun:
    retriever = build_retriever(name)
    started = time.perf_counter()
    batches = retriever.retrieve_batch(
        [row["question"] for row in questions],
        10,
    )
    seconds = time.perf_counter() - started
    rankings = {
        row["id"]: project_pages_to_concepts(
            [
                chunk_pages[chunk.id]
                for chunk in batches[index]
                if chunk.id in chunk_pages
            ],
            page_concepts,
        )
        for index, row in enumerate(questions)
    }
    query_count = (
        retriever.total_queries()
        if hasattr(retriever, "total_queries")
        else len(questions)
    )
    return EvidenceRun(name, seconds, query_count, rankings)


def run_okf(
    questions: list[dict], bundle_root: Path, concepts: set[str]
) -> tuple[EvidenceRun, list[dict]]:
    started = time.perf_counter()
    rankings = {}
    observations = []
    query_count = 0
    failures = 0
    for index, row in enumerate(questions, start=1):
        question_started = time.perf_counter()
        try:
            result = answer_question(row["question"], bundle_root=bundle_root)
            elapsed = time.perf_counter() - question_started
            query_count += result.query_count
            ranked_concepts = _unique(
                [path for path in result.citations if path in concepts]
                + [path for path in result.visited if path in concepts]
            )
            rankings[row["id"]] = ranked_concepts
            observations.append(
                {
                    "question_id": row["id"],
                    "question": row["question"],
                    "reference_answer": row["answer"],
                    "answer": result.answer,
                    "sufficient": result.sufficient,
                    "reason": result.reason,
                    "citations": result.citations,
                    "visited": result.visited,
                    "queries": result.query_count,
                    "seconds": elapsed,
                }
            )
        except Exception as exc:
            failures += 1
            rankings[row["id"]] = []
            observations.append(
                {
                    "question_id": row["id"],
                    "question": row["question"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.perf_counter() - question_started,
                }
            )
        print(f"[okf {index}/{len(questions)}] {row['question'][:70]}", flush=True)
    return (
        EvidenceRun(
            "okf",
            time.perf_counter() - started,
            query_count,
            rankings,
            failures,
        ),
        observations,
    )


def build_report(
    runs: list[EvidenceRun],
    questions: list[dict],
    relevance: dict[str, set[str]],
    bundle_root: Path,
) -> str:
    question_count = len(questions)
    speed_rows = []
    quality_rows = []
    for run in runs:
        seconds_per_question = run.seconds / question_count
        speed_rows.append(
            [
                run.name,
                f"{run.seconds:.2f}",
                f"{seconds_per_question * 1000:.1f}",
                f"{run.query_count / question_count:.2f}",
                run.query_count,
            ]
        )
        reciprocal_ranks = []
        hits = 0
        for row in questions:
            relevant = relevance[row["id"]]
            ranked = run.rankings.get(row["id"], [])
            rank = next(
                (
                    index
                    for index, concept in enumerate(ranked, start=1)
                    if concept in relevant
                ),
                None,
            )
            hits += rank is not None
            reciprocal_ranks.append(1 / rank if rank else 0.0)
        quality_rows.append(
            [
                run.name,
                f"{hits / question_count:.3f}",
                f"{sum(reciprocal_ranks) / question_count:.3f}",
                run.failures,
            ]
        )
    return "\n\n".join(
        [
            "# OKF vs RAG Evidence Acquisition Pilot",
            (
                f"Questions: {question_count} approved questions whose gold source "
                f"pages occur in `{bundle_root.relative_to(ROOT)}`."
            ),
            (
                "This pilot compares evidence acquisition, not final answer quality. "
                "RAG queries count retrieval attempts; OKF queries count "
                "navigation actions."
            ),
            plain_table(
                ["method", "seconds", "ms/query", "queries/query", "queries"],
                speed_rows,
            ),
            "## Concept-level evidence",
            plain_table(
                ["method", "concept hit", "concept MRR", "failures"],
                quality_rows,
            ),
        ]
    )


def _unique(values) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = parser.parse_args()
    load_local_env()
    bundle_root = args.bundle.resolve()
    page_map = load_bundle_page_map(bundle_root)
    concepts = set(page_map)
    page_concepts = invert_page_map(page_map)
    bundle_page_ids = {
        page_id for page_ids in page_map.values() for page_id in page_ids
    }
    questions, relevance, chunk_pages = load_covered_questions(
        bundle_page_ids, limit=args.limit
    )
    if not questions:
        raise RuntimeError("No approved eval questions are covered by the OKF bundle")
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    ensure_retrievers_ready(methods)
    concept_relevance = {
        question_id: set(project_pages_to_concepts(list(page_ids), page_concepts))
        for question_id, page_ids in relevance.items()
    }
    runs = [run_rag(name, questions, chunk_pages, page_concepts) for name in methods]
    okf_run, observations = run_okf(questions, bundle_root, concepts)
    runs.append(okf_run)
    report = build_report(runs, questions, concept_relevance, bundle_root)
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    latest = REPORTS_DIR / "okf_rag_latest.md"
    snapshot = REPORTS_DIR / f"okf_rag_{timestamp}.md"
    observations_path = REPORTS_DIR / f"okf_rag_{timestamp}.jsonl"
    latest.write_text(report + "\n", encoding="utf-8")
    snapshot.write_text(report + "\n", encoding="utf-8")
    observations_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in observations),
        encoding="utf-8",
    )
    print(f"\nSaved report: {latest}")
    print(f"Saved snapshot: {snapshot}")
    print(f"Saved OKF observations: {observations_path}")


if __name__ == "__main__":
    main()
