from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db as initialize_eval_db,
)
from src.eval.metrics import bold_best_table, plain_table, score_rankings
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    ensure_retrievers_ready,
    list_retrievers,
)
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


@dataclass(frozen=True)
class EvalRun:
    questions: list[dict]
    relevance: list[dict]
    methods: list[str]
    warmup_count: int
    rankings: dict[str, dict[str, list[str]]]
    timings: dict[str, dict[str, float]]
    score_names: list[str]
    scores: dict[str, dict[str, float]]


def evaluate(
    method_names: list[str],
    show_ranks: bool = False,
    limit: int | None = None,
    category: str | None = None,
) -> None:
    run = run_eval(method_names, limit=limit, category=category)
    if not run.questions:
        print("No eval questions found. Run src.eval.generate_dataset first.")
        return

    print(f"Evaluating {len(run.questions)} questions")
    print(_overall_table(run))
    print()
    print("Speed")
    print(_timing_table(run))
    if category is None:
        print()
        print(f"hit@{CONFIG['category_hit_k']} by category")
        print(_category_table(run))
    if show_ranks:
        print()
        print(_rank_table(run))


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


def run_eval(
    method_names: list[str],
    limit: int | None = None,
    category: str | None = None,
    warmup: int = 0,
) -> EvalRun:
    initialize_eval_db()
    questions, relevance = load_eval_rows(limit=limit, category=category)
    if not questions:
        return EvalRun([], [], [], 0, {}, {}, [], {})

    resolved_methods = _resolve_methods(method_names)
    ensure_retrievers_ready(resolved_methods)
    retrievers = {name: build_retriever(name) for name in resolved_methods}
    warmup = min(warmup, max(len(questions) - 1, 0))
    warmup_questions = questions[:warmup]
    timed_questions = questions[warmup:]
    if warmup_questions:
        for name in resolved_methods:
            _retrieve_rankings(retrievers[name], warmup_questions)

    method_rankings = {}
    timings = {}
    for name in resolved_methods:
        started = time.perf_counter()
        method_rankings[name] = _retrieve_rankings(retrievers[name], timed_questions)
        elapsed = time.perf_counter() - started
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / len(timed_questions),
        }

    timed_question_ids = {row["id"] for row in timed_questions}
    timed_relevance = [
        row for row in relevance if row["question_id"] in timed_question_ids
    ]
    score_names = [f"hit@{k}" for k in CONFIG["metric_ks"]]
    score_names.append(f"mrr@{CONFIG['mrr_k']}")
    scores = {
        name: score_rankings(
            timed_relevance,
            method_rankings[name],
            ks=tuple(CONFIG["metric_ks"]),
            mrr_k=CONFIG["mrr_k"],
        )
        for name in resolved_methods
    }
    return EvalRun(
        questions=timed_questions,
        relevance=timed_relevance,
        methods=resolved_methods,
        warmup_count=warmup,
        rankings=method_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
    )


def load_eval_rows(
    limit: int | None = None,
    category: str | None = None,
) -> tuple[list[dict], list[dict]]:
    with connect() as conn:
        question_sql = """
                select id, question, answer, question_type, question_language
                from eval_questions
                where approved = 1
                """
        params = {}
        if category is not None:
            question_sql += "\n                and question_type = :category"
            params["category"] = category
        question_sql += "\n                order by id"
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
        placeholders = ", ".join("?" for _ in question_ids)
        relevance = [
            dict(row)
            for row in conn.execute(
                f"""
                select question_id, chunk_id
                from eval_relevant_chunks
                where question_id in ({placeholders})
                order by question_id, chunk_id
                """,
                question_ids,
            )
        ]
    return questions, relevance


def _retrieve_rankings(
    retriever: Retriever,
    questions: list[dict],
) -> dict[str, list[str]]:
    batches = retriever.retrieve_batch(
        [row["question"] for row in questions],
        CONFIG["result_limit"],
    )
    return {
        row["id"]: [chunk.id for chunk in batches[index]]
        for index, row in enumerate(questions)
    }


def _overall_table(run: EvalRun) -> str:
    rows = [
        [name, *(run.scores[name][score_name] for score_name in run.score_names)]
        for name in run.methods
    ]
    return "\n".join(
        [
            "Overall",
            bold_best_table(["method", *run.score_names], rows),
        ]
    )


def _timing_table(run: EvalRun) -> str:
    return plain_table(
        ["method", "seconds", "ms/query"],
        [
            [
                name,
                f"{run.timings[name]['seconds']:.2f}",
                f"{run.timings[name]['ms_per_query']:.1f}",
            ]
            for name in run.methods
        ],
    )


def _category_table(run: EvalRun) -> str:
    question_type_by_id = {row["id"]: row["question_type"] for row in run.questions}
    types = sorted(set(question_type_by_id.values()))
    relevance_by_type = defaultdict(list)
    for row in run.relevance:
        relevance_by_type[question_type_by_id[row["question_id"]]].append(row)

    rows = []
    for method_name in run.methods:
        row = [method_name]
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
            row.append(score)
        rows.append(row)
    return bold_best_table(["method", *types], rows)


def _rank_table(run: EvalRun) -> str:
    relevant_by_question = defaultdict(set)
    for row in run.relevance:
        relevant_by_question[row["question_id"]].add(row["chunk_id"])

    rows = []
    for question in run.questions:
        row = [question["question_type"], question["question"][:80]]
        for method_name in run.methods:
            rank = _first_rank(
                run.rankings[method_name][question["id"]],
                relevant_by_question[question["id"]],
            )
            row.append(rank or "-")
        rows.append(row)
    return plain_table(["type", "question", *run.methods], rows)


def _first_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return index
    return None


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
    args = parser.parse_args()
    evaluate(
        [name.strip() for name in args.methods.split(",") if name.strip()],
        show_ranks=args.show_ranks,
        limit=args.limit,
        category=args.category,
    )


if __name__ == "__main__":
    main()
