from __future__ import annotations

import argparse

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db as initialize_eval_db,
)
from src.eval.metrics import plain_table


def inspect_dataset(limit: int) -> None:
    initialize_eval_db()
    with connect() as conn:
        summary = conn.execute(
            """
            select question_type, question_language, count(*),
              min(length(question)), max(length(question)),
              min(length(answer)), max(length(answer))
            from eval_questions
            where approved = 1
            group by question_type, question_language
            order by question_type, question_language
            """
        ).fetchall()
        rows = conn.execute(
            """
            select approved, question_type, question_language, length(question),
              length(answer), question, answer
            from eval_questions
            order by rowid desc
            limit ?
            """,
            (limit,),
        ).fetchall()

    print("Dataset shape")
    print(
        plain_table(
            ["type", "lang", "n", "q_min", "q_max", "a_min", "a_max"],
            [list(row) for row in summary],
        )
    )
    print()
    print("Recent rows")
    print(
        plain_table(
            ["ok", "type", "lang", "q_len", "a_len", "question", "answer"],
            [list(row) for row in rows],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    inspect_dataset(args.limit)


if __name__ == "__main__":
    main()
