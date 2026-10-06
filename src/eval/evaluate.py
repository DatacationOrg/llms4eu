from __future__ import annotations

from typing import Any
import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.db.pages import chunk_variant, variant_tag
from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db as initialize_eval_db,
)
from src.eval.metrics import (
    bold_best_table,
    first_relevant_rank,
    plain_table,
    score_rankings,
)
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    ensure_retrievers_ready,
    list_retrievers,
)
from src.shared.env import ROOT, data_path, load_local_env, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
REPORTS_DIR = ROOT / ".local" / "reports"


@dataclass(frozen=True)
class EvalRun:
    questions: list[dict[str, Any]]
    relevance: list[dict[str, Any]]
    methods: list[str]
    rankings: dict[str, dict[str, list[str]]]
    timings: dict[str, dict[str, float]]
    score_names: list[str]
    scores: dict[str, dict[str, float]]


def evaluate(
    method_names: list[str],
    show_ranks: bool = False,
    limit: int | None = None,
    category: str | None = None,
    checkpoint: bool = False,
) -> None:
    run = run_eval(method_names, limit=limit, category=category, checkpoint=checkpoint)
    if not run.questions:
        print("No eval questions found. Run src.eval.generate_dataset first.")
        return

    report = _build_eval_report(
        run,
        show_ranks=show_ranks,
        category=category,
        limit=limit,
    )
    print(report)
    latest_path, snapshot_path = _write_eval_report(report)
    print()
    print(f"Saved report: {latest_path}")
    print(f"Saved snapshot: {snapshot_path}")


def format_eval_report(
    run: EvalRun,
    include_categories: bool = True,
) -> str:
    sections = [
        f"Evaluating {len(run.questions)} questions",
        _overall_table(run),
        "Speed",
        _timing_table(run),
    ]
    if include_categories:
        sections.extend(
            [
                f"hit@{CONFIG['category_hit_k']} by category",
                _category_table(run),
            ]
        )
    return "\n\n".join(sections)


def _build_eval_report(
    run: EvalRun,
    show_ranks: bool,
    category: str | None,
    limit: int | None,
) -> str:
    lines = [
        f"Methods: {', '.join(run.methods)}",
        f"Chunk variant: {chunk_variant()}",
        f"Questions: {len(run.questions)}",
        f"Limit: {limit if limit is not None else 'all'}",
        f"Category: {category or 'all'}",
        "",
        format_eval_report(run, include_categories=category is None),
    ]
    if show_ranks:
        lines.extend(["", _rank_table(run)])
    return "\n".join(lines)


def _write_eval_report(report: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    latest_path = REPORTS_DIR / "eval_latest.txt"
    snapshot_path = REPORTS_DIR / f"eval_{timestamp}.txt"
    content = report + "\n"
    latest_path.write_text(content, encoding="utf-8")
    snapshot_path.write_text(content, encoding="utf-8")
    return latest_path, snapshot_path


def run_eval(
    method_names: list[str],
    limit: int | None = None,
    category: str | None = None,
    checkpoint: bool = False,
) -> EvalRun:
    initialize_eval_db()
    questions, relevance = load_eval_rows(limit=limit, category=category)
    if not questions:
        return EvalRun([], [], [], {}, {}, [], {})

    resolved_methods = _resolve_methods(method_names)
    ensure_retrievers_ready(resolved_methods)
    retrievers = {name: build_retriever(name) for name in resolved_methods}
    signature = [row["id"] for row in questions]
    done = _load_checkpoint(signature) if checkpoint else {}

    method_rankings = {}
    timings = {}
    for name in resolved_methods:
        if name in done:
            print(f"{name}: resumed from checkpoint")
            method_rankings[name] = done[name]["rankings"]
            timings[name] = done[name]["timing"]
            continue
        started = time.perf_counter()
        method_rankings[name], effort = _retrieve_rankings(retrievers[name], questions)
        elapsed = time.perf_counter() - started
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / len(questions),
            "queries_per_query": effort["queries_per_query"],
            "total_queries": effort["total_queries"],
        }
        if checkpoint:
            done[name] = {"rankings": method_rankings[name], "timing": timings[name]}
            _save_checkpoint(signature, done)

    score_names, scores = score_eval_rankings(
        relevance, method_rankings, resolved_methods
    )
    return EvalRun(
        questions=questions,
        relevance=relevance,
        methods=resolved_methods,
        rankings=method_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
    )


# ponytail: resume is per method, not per question. A crash loses at most the
# method in flight. Go per question only if one method's run outgrows a sitting.
def _checkpoint_path() -> Path:
    return data_path("checkpoints", f"eval{variant_tag(chunk_variant(), '-')}.json")


def _load_checkpoint(signature: list[str]) -> dict[str, Any]:
    path = _checkpoint_path()
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("questions") != signature:
        print("Checkpoint covers a different question set; starting fresh.")
        return {}
    return saved["methods"]


def _save_checkpoint(signature: list[str], methods: dict[str, Any]) -> None:
    _checkpoint_path().write_text(
        json.dumps({"questions": signature, "methods": methods}),
        encoding="utf-8",
    )


def load_eval_rows(
    limit: int | None = None,
    category: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # A question counts for a variant only when its evidence quote was found in
    # that variant's chunks, so every variant is scored on labelled questions.
    with connect() as conn:
        question_sql = """
                select q.id, q.question, q.answer, q.question_type, q.question_language
                from eval_questions q
                where q.approved = 1
                  and exists (
                    select 1
                    from eval_relevant_chunks r
                    join page_chunks c on c.id = r.chunk_id
                    where r.question_id = q.id and c.variant = :variant
                  )
                """
        params: dict[str, Any] = {"variant": chunk_variant()}
        if category is not None:
            question_sql += "\n                and q.question_type = :category"
            params["category"] = category
        question_sql += "\n                order by q.id"
        if limit is not None:
            question_sql += "\n                limit :limit"
            params["limit"] = limit
        questions = [
            dict(row)
            for row in conn.execute(
                question_sql,
                params,
            )
        ]
        question_ids = [row["id"] for row in questions]
        if not question_ids:
            return questions, []
        relevance = [
            dict(row)
            for row in conn.execute(
                """
                select r.question_id, r.chunk_id
                from eval_relevant_chunks r
                join page_chunks c on c.id = r.chunk_id
                where c.variant = ?
                  and r.question_id in (select value from json_each(?))
                order by r.question_id, r.chunk_id
                """,
                [chunk_variant(), json.dumps(question_ids)],
            )
        ]
    return questions, relevance


def _retrieve_rankings(
    retriever: Retriever,
    questions: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, float]]:
    batches = retriever.retrieve_batch(
        [row["question"] for row in questions],
        CONFIG["result_limit"],
    )
    rankings = {
        row["id"]: [chunk.id for chunk in batches[index]]
        for index, row in enumerate(questions)
    }
    return rankings, _query_effort(retriever, question_count=len(questions))


def _overall_table(run: EvalRun) -> str:
    rows = [
        [
            name,
            *(run.scores[name].get(score_name, "-") for score_name in run.score_names),
        ]
        for name in run.methods
    ]
    return "\n".join(
        [
            "Overall",
            bold_best_table(["method", *run.score_names], rows),
        ]
    )


def score_eval_rankings(
    relevance: list[dict[str, Any]],
    rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    metric_ks = tuple(CONFIG["metric_ks"])
    standard_k = max(metric_ks)
    score_names = [f"hit@{k}" for k in metric_ks]
    score_names.extend([f"recall@{standard_k}", f"mrr@{CONFIG['mrr_k']}"])
    scores = {
        name: score_rankings(
            relevance,
            rankings[name],
            ks=metric_ks,
            mrr_k=CONFIG["mrr_k"],
            recall_k=standard_k,
        )
        for name in method_names
    }

    return score_names, scores


def _timing_table(run: EvalRun) -> str:
    return plain_table(
        ["method", "seconds", "ms/query", "queries/query", "queries"],
        [
            [
                name,
                f"{run.timings[name]['seconds']:.2f}",
                f"{run.timings[name]['ms_per_query']:.1f}",
                f"{run.timings[name]['queries_per_query']:.2f}",
                int(run.timings[name]["total_queries"]),
            ]
            for name in run.methods
        ],
    )


def _query_effort(retriever: Retriever, question_count: int) -> dict[str, float]:
    if question_count == 0:
        return {"queries_per_query": 0.0, "total_queries": 0.0}

    average = getattr(retriever, "average_queries_per_question", None)
    total = getattr(retriever, "total_queries", None)
    if average is not None and total is not None:
        return {
            "queries_per_query": float(average()),
            "total_queries": float(total()),
        }

    return {
        "queries_per_query": 1.0,
        "total_queries": float(question_count),
    }


def _category_table(run: EvalRun) -> str:
    question_type_by_id = {row["id"]: row["question_type"] for row in run.questions}
    types = sorted(set(question_type_by_id.values()))
    relevance_by_type = defaultdict(list)
    for row in run.relevance:
        relevance_by_type[question_type_by_id[row["question_id"]]].append(row)

    rows = []
    for method_name in run.methods:
        cells: list[Any] = [method_name]
        for question_type in types:
            type_question_ids = {
                item["question_id"] for item in relevance_by_type[question_type]
            }
            rankings = {
                question_id: run.rankings[method_name][question_id]
                for question_id in type_question_ids
            }
            score = score_rankings(
                relevance_by_type[question_type],
                rankings,
                ks=(CONFIG["category_hit_k"],),
                mrr_k=CONFIG["mrr_k"],
            )[f"hit@{CONFIG['category_hit_k']}"]
            cells.append(score)
        rows.append(cells)
    return bold_best_table(["method", *types], rows)


def _rank_table(run: EvalRun) -> str:
    relevant_by_question = defaultdict(set)
    for row in run.relevance:
        relevant_by_question[row["question_id"]].add(row["chunk_id"])

    rows = []
    for question in run.questions:
        row = [question["question_type"], question["question"][:80]]
        for method_name in run.methods:
            rank = first_relevant_rank(
                run.rankings[method_name][question["id"]],
                relevant_by_question[question["id"]],
            )
            row.append(rank or "-")
        rows.append(row)
    return plain_table(["type", "question", *run.methods], rows)


def _resolve_methods(method_names: list[str]) -> list[str]:
    if not method_names:
        return CONFIG["default_methods"]
    if method_names == ["all"]:
        return list_retrievers()
    unknown = sorted(set(method_names) - set(list_retrievers()))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return method_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(CONFIG["default_methods"]))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--show-ranks", action="store_true")
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Resume a long run; finished methods are skipped.",
    )
    args = parser.parse_args()

    load_local_env()
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]

    evaluate(
        methods,
        show_ranks=args.show_ranks,
        limit=args.limit,
        category=args.category,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
