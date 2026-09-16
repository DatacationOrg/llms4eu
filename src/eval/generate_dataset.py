from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from src.db.pages import connect_pages as connect
from src.db.pages import initialize_page_artifacts_db as initialize_eval_db
from src.shared.env import load_yaml
from src.shared.llm import (
    structured_local_model,
)

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

QUESTION_TYPES = tuple(CONFIG["question_target_chars"])
QUESTION_TARGET_CHARS = CONFIG["question_target_chars"]
QUESTION_MAX_CHARS = max(QUESTION_TARGET_CHARS.values()) * 2
ANSWER_TARGET_CHARS = CONFIG["answer_target_chars"]
ANSWER_MAX_CHARS = ANSWER_TARGET_CHARS + 512
LANGUAGE_NAMES = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "ga": "Irish",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
}


class QuestionCandidate(BaseModel):
    question: str = Field(
        max_length=QUESTION_MAX_CHARS,
        description="A factual question answerable only from the source chunk. Aim for the question type's target size, not this safety limit.",
    )
    answer: str = Field(
        max_length=ANSWER_MAX_CHARS,
        description=f"A concise factual answer in the same language as the question. Aim for about {ANSWER_TARGET_CHARS} characters.",
    )
    question_language: str = Field(
        description="BCP-47/ISO-style language code of the question text."
    )


class EvalQuestionBatch(BaseModel):
    same_language: QuestionCandidate | None = Field(
        default=None,
        description=f"Question in the chunk's own language, about {QUESTION_TARGET_CHARS['same_language']} characters.",
    )
    cross_language: QuestionCandidate | None = Field(
        default=None,
        description=f"Question in the injected target language, about {QUESTION_TARGET_CHARS['cross_language']} characters, answered in that language.",
    )


def generate_dataset(
    limit: int | None,
    model_id: str | None = None,
    reasoning: bool | str | None = None,
) -> None:
    initialize_eval_db()
    chunks = _eligible_unprocessed_chunks(limit)
    print(f"found {len(chunks)} eligible unprocessed chunks", flush=True)
    model_name = model_id or CONFIG["question_model"]
    structured_model = structured_local_model(
        model_name,
        EvalQuestionBatch,
        reasoning=CONFIG["question_model_reasoning"]
        if reasoning is None
        else reasoning,
        num_ctx=CONFIG["question_model_num_ctx"],
        num_predict=CONFIG["question_model_num_predict"],
    )

    inserted = 0
    for index, chunk in enumerate(chunks, start=1):
        with connect() as conn:
            print(
                f"[{index}/{len(chunks)}] generating questions for {chunk['id']}",
                flush=True,
            )
            target_language = _cross_language(chunk["id"], chunk["language"])
            result = structured_model.invoke(_messages(chunk, target_language))
            questions = [
                item for item in _flatten_questions(result) if _valid_question(*item)
            ]
            inserted += insert_questions(conn, chunk["id"], questions)
            print(f"[{index}/{len(chunks)}] got {len(questions)} questions", flush=True)
    print(
        f"processed {len(chunks)} chunks, inserted up to {inserted} questions",
        flush=True,
    )


def _eligible_unprocessed_chunks(limit: int | None) -> list[dict]:
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                select c.id, c.text, c.heading_path, m.title,
                       coalesce(m.language, s.language) as language
                from page_chunks c
                join page_metadata m on m.id = c.page_id
                join page_sources s on s.source = m.source
                where not exists (
                  select 1
                  from eval_relevant_chunks r
                  where r.chunk_id = c.id
                )
                order by c.id
                """
            )
        ]
    chunks = [row for row in rows if _is_fact_dense(row["text"])]
    return chunks[:limit] if limit else chunks


def _is_fact_dense(text: str) -> bool:
    lowered = text.lower()
    if len(text) < CONFIG["fact_dense_min_chars"]:
        return False
    if any(term in lowered for term in ("privacy", "cookie", "gdpr", "consent")):
        return False
    if any(
        term in lowered
        for term in (
            "isbn",
            "issn",
            "accessed",
            "retrieved",
            "archived",
            "wayback machine",
            "citeref",
        )
    ):
        return False
    link_count = text.count("](")
    word_count = len(text.split())
    if link_count > 20 and link_count > word_count / 20:
        return False
    return bool(re.search(r"\d{3,4}|[^\W\d_]{3,}\s+[^\W\d_]{3,}", text, re.UNICODE))


def _flatten_questions(batch: EvalQuestionBatch) -> list[tuple[str, QuestionCandidate]]:
    return [
        (question_type, item)
        for question_type in QUESTION_TYPES
        for item in [getattr(batch, question_type)]
        if item is not None
    ]


def insert_questions(
    conn,
    chunk_id: str,
    questions: list[tuple[str, QuestionCandidate]],
) -> int:
    inserted = 0
    for question_type, item in questions:
        question_id = _question_id(chunk_id, item)
        cursor = conn.execute(
            """
            insert or ignore into eval_questions (
              id, question, answer, question_type, question_language
            ) values (?, ?, ?, ?, ?)
            """,
            (
                question_id,
                item.question.strip(),
                item.answer.strip(),
                question_type,
                item.question_language,
            ),
        )
        conn.execute(
            """
            insert or ignore into eval_relevant_chunks (question_id, chunk_id)
            values (?, ?)
            """,
            (question_id, chunk_id),
        )
        inserted += cursor.rowcount
    return inserted


def _valid_question(question_type: str, item: QuestionCandidate) -> bool:
    question = item.question.strip()
    answer = item.answer.strip()
    if not question or not answer:
        return False
    return (
        len(answer) <= ANSWER_MAX_CHARS
        and len(question) <= QUESTION_TARGET_CHARS[question_type] * 2
    )


def _messages(chunk: dict, target_language: str | None) -> list[tuple[str, str]]:
    return [
        ("system", _system_prompt(chunk, target_language)),
        ("human", _human_prompt(chunk)),
    ]


def _system_prompt(chunk: dict, target_language: str | None) -> str:
    source_language_name = LANGUAGE_NAMES.get(chunk["language"], chunk["language"])
    target_language = target_language or "en"
    target_language_name = LANGUAGE_NAMES[target_language]
    return f"""
Generate one question and one answer for each question type. Use only facts in the chunk.

Write same_language in {source_language_name}, about {QUESTION_TARGET_CHARS["same_language"]} characters.
Write cross_language in {target_language_name} and set question_language to "{target_language}", about {QUESTION_TARGET_CHARS["cross_language"]} characters.
Answers must use the same language as their questions.

Answer target: about {ANSWER_TARGET_CHARS} characters.
If a type is not supported by the chunk, leave it null.
"""


def _human_prompt(chunk: dict) -> str:
    return f"""
Title: {chunk["title"] or ""}
Heading: {chunk["heading_path"] or ""}
Content language: {chunk["language"]}

Chunk:
{chunk["text"]}
"""


def _cross_language(chunk_id: str, source_language: str) -> str:
    """Sample one target language per chunk, never the chunk's own."""
    targets = tuple(code for code in LANGUAGE_NAMES if code != source_language)
    return targets[uuid.uuid5(uuid.NAMESPACE_URL, chunk_id).int % len(targets)]


def _question_id(chunk_id: str, question: QuestionCandidate) -> str:
    key = f"{chunk_id}\n{question.question_language}\n{question.question}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--reasoning", action="store_true")
    args = parser.parse_args()
    generate_dataset(args.limit, args.model, args.reasoning or None)


if __name__ == "__main__":
    main()
