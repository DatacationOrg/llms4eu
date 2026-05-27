from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from src.eval.db import connect, initialize_eval_db
from src.shared.env import load_yaml
from src.shared.llm import structured_local_model

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

QUESTION_TYPES = (
    "direct_short",
    "direct_long",
    "vague_short",
    "vague_long",
    "crosslingual",
)
EU_LANGUAGES = (
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "ga",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "es",
    "sv",
)
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
        max_length=1024,
        description="A factual question answerable only from the source chunk.",
    )
    answer: str = Field(
        max_length=128,
        description="A brief factual answer, ideally 64 chars or less, in the same language as the question.",
    )
    question_language: str = Field(
        description="BCP-47/ISO-style language code of the question text."
    )


class EvalQuestionBatch(BaseModel):
    direct_short: QuestionCandidate | None = Field(
        default=None,
        description="Source-language question under 128 chars with wording close to the chunk.",
    )
    direct_long: QuestionCandidate | None = Field(
        default=None,
        description="Source-language question from 128 to 1024 chars with extra context.",
    )
    vague_short: QuestionCandidate | None = Field(
        default=None,
        description="Source-language question under 128 chars with indirect wording.",
    )
    vague_long: QuestionCandidate | None = Field(
        default=None,
        description="Source-language question from 128 to 1024 chars with indirect wording.",
    )
    crosslingual: QuestionCandidate | None = Field(
        default=None,
        description="Question in the requested target language, with answer in that language.",
    )


def generate_dataset(
    limit: int | None,
    model_id: str | None = None,
    reasoning: bool | str | None = None,
) -> None:
    initialize_eval_db()
    chunks = _eligible_unprocessed_chunks(limit)
    print(f"found {len(chunks)} eligible unprocessed chunks", flush=True)
    model_name = _resolve_model_name(model_id or CONFIG["question_model"])
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
            target_language = _crosslingual_language(chunk["id"])
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
                select c.id, c.text, c.heading_path, m.title, s.language
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


def _resolve_model_name(model_name: str) -> str:
    return CONFIG.get("model_aliases", {}).get(model_name, model_name)


def _is_fact_dense(text: str) -> bool:
    lowered = text.lower()
    if len(text) < 300:
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
    if len(answer) >= 64 or len(question) > 1024:
        return False
    if question_type.endswith("_short"):
        return len(question) < 128
    if question_type.endswith("_long"):
        return len(question) >= 128
    return True


def _messages(chunk: dict, target_language: str | None) -> list[tuple[str, str]]:
    return [
        ("system", _system_prompt(chunk, target_language)),
        ("human", _human_prompt(chunk)),
    ]


def _system_prompt(chunk: dict, target_language: str | None) -> str:
    source_language_name = LANGUAGE_NAMES.get(chunk["language"], chunk["language"])
    target_language_name = LANGUAGE_NAMES[target_language]
    return f"""
Generate factual retrieval-evaluation questions from this source chunk.

Rules:
- Fill each output field if the chunk supports it.
- Use the source content language ({source_language_name}) for direct_short, direct_long, vague_short, and vague_long.
- direct_short: under 128 chars, obvious and close to the source.
- direct_long: at least 128 chars and at most 1024 chars, includes context but asks one factual answer.
- vague_short: under 128 chars, indirect but answerable from the chunk.
- vague_long: at least 128 chars and at most 1024 chars, indirect/contextual but answerable.
- crosslingual: write the question entirely in {target_language_name}. Set question_language to "{target_language}".
- Do not anchor every question on the main named entity. Prefer a mix:
  - some obvious "who/where/when/what" questions,
  - some descriptive questions using roles, dates, places, institutions, or events,
  - some semantic questions that can be answered without repeating the exact title/name.
- For vague questions, avoid putting the main person's/place's full name in the question unless needed for clarity.
- Every answer must be in the same language as its question. Translate non-name answers for crosslingual.
- Every answer should be shorter than 64 characters. If the factual answer needs more space, leave that field null.
- Ask only factual questions with answers explicitly present in the chunk.
- Do not ask opinions, recommendations, or questions needing outside knowledge.
- Avoid bibliographic/citation facts such as ISBNs, URLs, access dates, or reference lists.
- If a long question would be shorter than 128 chars, leave that field null.
- If the chunk has too little factual information, leave every field null.
"""


def _human_prompt(chunk: dict) -> str:
    return f"""
Title: {chunk["title"] or ""}
Heading: {chunk["heading_path"] or ""}
Content language: {chunk["language"]}

Chunk:
{chunk["text"]}
"""


def _crosslingual_language(chunk_id: str) -> str:
    index = int(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id).int % len(EU_LANGUAGES))
    return EU_LANGUAGES[index]


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
