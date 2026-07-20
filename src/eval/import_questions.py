from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db as initialize_eval_db,
)
from src.eval.generate_dataset import (
    QUESTION_TYPES,
    QuestionCandidate,
    _valid_question,
    insert_questions,
)


def import_questions(path: Path) -> None:
    initialize_eval_db()
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("["):
        rows = json.loads(content)
    else:
        rows = [json.loads(line) for line in content.splitlines()]
    inserted = 0
    skipped = 0
    with connect() as conn:
        requested_chunk_ids = {row["chunk_id"] for row in rows}
        if not requested_chunk_ids:
            print("inserted 0, skipped 0")
            return
        chunk_ids = {
            row["id"]
            for row in conn.execute(
                f"select id from page_chunks where id in ({','.join('?' for _ in requested_chunk_ids)})",
                tuple(requested_chunk_ids),
            )
        }
        for row in rows:
            question_type = row["question_type"]
            item = QuestionCandidate(
                question=row["question"],
                answer=row["answer"],
                question_language=row["question_language"],
            )
            if (
                row["chunk_id"] not in chunk_ids
                or question_type not in QUESTION_TYPES
                or not _valid_question(question_type, item)
            ):
                skipped += 1
                continue
            inserted += insert_questions(conn, row["chunk_id"], [(question_type, item)])
    print(f"inserted {inserted}, skipped {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    import_questions(args.path)


if __name__ == "__main__":
    main()
