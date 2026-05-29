from __future__ import annotations

import argparse

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db as initialize_eval_db,
)


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
