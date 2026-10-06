"""Anchor each eval answer to a verbatim quote, and derive chunk labels from it.

Questions are generated from one chunk, so their label is a chunk id, which a
rechunk destroys. A quote from the page does not depend on how the page is cut:
`backfill` asks the model for one, once, and keeps it only if it really appears
in the chunk; `relabel` points every question at the chunks now containing it.
Both work on the active chunk variant (CHUNK_VARIANT), so each variant carries
its own labels for the same questions.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from pydantic import BaseModel, Field

from src.db.legacy.pages import chunk_variant
from src.db.legacy.pages import connect_pages as connect
from src.db.legacy.pages import initialize_page_artifacts_db
from src.shared.env import load_yaml
from src.shared.llm import LocalOllamaStructuredLlm
from src.shared.prompts import render

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
MIN_QUOTE_CHARS = 20


class Evidence(BaseModel):
    quote: str = Field(description="Verbatim sentences from the chunk.")


def normalize(text: str) -> str:
    return " ".join(text.split())


def backfill(limit: int | None = None) -> None:
    initialize_page_artifacts_db()
    with connect() as conn:
        rows = conn.execute(
            """
            select q.id, q.question, q.answer, c.page_id, c.text
            from eval_questions q
            join page_chunks c on c.id = (
              select min(r.chunk_id) from eval_relevant_chunks r
              join page_chunks labelled on labelled.id = r.chunk_id
              where r.question_id = q.id and labelled.variant = ?
            )
            where not exists (select 1 from eval_evidence e where e.question_id = q.id)
            order by q.id
            """,
            (chunk_variant(),),
        ).fetchall()
    rows = rows[:limit] if limit else rows
    llm = LocalOllamaStructuredLlm(
        CONFIG["question_model"],
        reasoning=CONFIG["question_model_reasoning"],
        num_ctx=CONFIG["question_model_num_ctx"],
        num_predict=CONFIG["question_model_num_predict"],
    )
    failed = 0
    for index, row in enumerate(rows, start=1):
        prompt = render(
            "evidence", question=row["question"], answer=row["answer"], text=row["text"]
        )
        try:
            quote = normalize(llm.structured_output(prompt, Evidence).quote)
        except RuntimeError as exc:
            print(f"[{index}/{len(rows)}] {row['id']}: {exc}", flush=True)
            quote = ""
        if len(quote) < MIN_QUOTE_CHARS or quote not in normalize(row["text"]):
            failed += 1
            continue
        with connect() as conn:
            conn.execute(
                "insert into eval_evidence (question_id, page_id, quote) values (?, ?, ?)",
                (row["id"], row["page_id"], quote),
            )
    print(f"anchored {len(rows) - failed} of {len(rows)}; rerun to retry the rest")


def matching_chunks(quote: str, chunks: dict[str, str]) -> list[str]:
    """Chunks containing the quote, else those containing any of its sentences."""
    texts = {chunk_id: normalize(text) for chunk_id, text in chunks.items()}
    found = [chunk_id for chunk_id, text in texts.items() if quote in text]
    if found:
        return found
    # ponytail: a quote cut by a chunk boundary labels each side by sentence; a
    # single sentence split mid-way is lost. Overlap > one sentence avoids it.
    sentences = [
        s for s in re.split(r"(?<=[.!?])\s+", quote) if len(s) >= MIN_QUOTE_CHARS
    ]
    return [cid for cid, text in texts.items() if any(s in text for s in sentences)]


def relabel() -> None:
    variant = chunk_variant()
    initialize_page_artifacts_db()
    with connect() as conn:
        evidence = conn.execute(
            "select question_id, page_id, quote from eval_evidence"
        ).fetchall()
        chunks_by_page: dict[str, dict[str, str]] = {}
        for row in conn.execute(
            "select id, page_id, text from page_chunks where variant = ?", (variant,)
        ):
            chunks_by_page.setdefault(row["page_id"], {})[row["id"]] = row["text"]
        unmatched = 0
        for row in evidence:
            found = matching_chunks(
                row["quote"], chunks_by_page.get(row["page_id"], {})
            )
            unmatched += not found
            conn.execute(
                """
                delete from eval_relevant_chunks where question_id = ?
                  and chunk_id in (select id from page_chunks where variant = ?)
                """,
                (row["question_id"], variant),
            )
            conn.executemany(
                "insert into eval_relevant_chunks (question_id, chunk_id) values (?, ?)",
                [(row["question_id"], chunk_id) for chunk_id in found],
            )
    print(
        f"relabelled {len(evidence) - unmatched} questions on {variant}, "
        f"{unmatched} unmatched"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("backfill").add_argument("--limit", type=int)
    commands.add_parser("relabel")
    args = parser.parse_args()
    if args.command == "backfill":
        backfill(args.limit)
    else:
        relabel()


if __name__ == "__main__":
    main()
