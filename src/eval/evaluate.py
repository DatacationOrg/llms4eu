from __future__ import annotations

import argparse
from collections import defaultdict

from src.eval.db import connect, initialize_eval_db
from src.eval.metrics import bold_best_table, plain_table, score_rankings
from src.eval.ranking import available_ranking_methods, get_ranking_method


def evaluate(
    method_names: list[str],
    show_ranks: bool = False,
    limit: int | None = None,
) -> None:
    initialize_eval_db()
    questions, relevance = _load_eval_rows(limit)
    if not questions:
        print("No eval questions found. Run src.eval.generate_dataset first.")
        return

    print(f"Evaluating {len(questions)} questions")
    resolved_methods = _resolve_methods(method_names)
    methods = {name: get_ranking_method(name) for name in resolved_methods}
    method_rankings = {
        name: {
            row["id"]: [
                chunk.id for chunk in methods[name].retrieve(row["question"], 10)
            ]
            for row in questions
        }
        for name in resolved_methods
    }

    overall_rows = []
    for name in resolved_methods:
        scores = score_rankings(relevance, method_rankings[name])
        overall_rows.append(
            [name, scores["hit@1"], scores["hit@5"], scores["hit@10"], scores["mrr@10"]]
        )

    print("Overall")
    print(
        bold_best_table(["method", "hit@1", "hit@5", "hit@10", "mrr@10"], overall_rows)
    )
    print()
    print("hit@5 by category")
    print(_category_table(questions, relevance, method_rankings, resolved_methods))
    if show_ranks:
        print()
        print(_rank_table(questions, relevance, method_rankings, resolved_methods))


def _load_eval_rows(limit: int | None = None) -> tuple[list[dict], list[dict]]:
    with connect() as conn:
        question_sql = """
                select id, question, answer, question_type, question_language
                from eval_questions
                where approved = 1
                order by id
                """
        if limit is not None:
            question_sql += "\n                limit :limit"
        questions = [
            dict(row)
            for row in conn.execute(
                question_sql,
                {"limit": limit},
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


def _category_table(
    questions: list[dict],
    relevance: list[dict],
    method_rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
) -> str:
    question_type_by_id = {row["id"]: row["question_type"] for row in questions}
    types = sorted(set(question_type_by_id.values()))
    relevance_by_type = defaultdict(list)
    for row in relevance:
        relevance_by_type[question_type_by_id[row["question_id"]]].append(row)

    rows = []
    for method_name in method_names:
        row = [method_name]
        for question_type in types:
            type_question_ids = {
                item["question_id"] for item in relevance_by_type[question_type]
            }
            rankings = {
                question_id: method_rankings[method_name][question_id]
                for question_id in type_question_ids
            }
            score = score_rankings(
                relevance_by_type[question_type],
                rankings,
                ks=(5,),
            )["hit@5"]
            row.append(score)
        rows.append(row)
    return bold_best_table(["method", *types], rows)


def _rank_table(
    questions: list[dict],
    relevance: list[dict],
    method_rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
) -> str:
    relevant_by_question = defaultdict(set)
    for row in relevance:
        relevant_by_question[row["question_id"]].add(row["chunk_id"])

    rows = []
    for question in questions:
        row = [question["question_type"], question["question"][:80]]
        for method_name in method_names:
            rank = _first_rank(
                method_rankings[method_name][question["id"]],
                relevant_by_question[question["id"]],
            )
            row.append(rank or "-")
        rows.append(row)
    return plain_table(["type", "question", *method_names], rows)


def _first_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return index
    return None


def _resolve_methods(method_names: list[str]) -> list[str]:
    if method_names == ["all"]:
        return sorted(available_ranking_methods())
    unknown = sorted(set(method_names) - available_ranking_methods())
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return method_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show-ranks", action="store_true")
    args = parser.parse_args()
    evaluate(
        [name.strip() for name in args.methods.split(",") if name.strip()],
        show_ranks=args.show_ranks,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
