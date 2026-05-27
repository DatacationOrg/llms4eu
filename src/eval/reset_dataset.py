from __future__ import annotations

import argparse

from src.eval.db import connect, initialize_eval_db


def reset_dataset() -> None:
    initialize_eval_db()
    with connect() as conn:
        conn.execute("delete from eval_relevant_chunks")
        conn.execute("delete from eval_questions")
    print("deleted eval questions and relevance labels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    reset_dataset()


if __name__ == "__main__":
    main()
