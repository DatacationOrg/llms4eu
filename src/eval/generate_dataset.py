from __future__ import annotations

import argparse
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, ValidationError

from src.eval.db import connect, initialize_eval_db
from src.shared.env import load_local_env, load_yaml
from src.shared.llm import structured_local_model

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

QUESTION_TYPES = (
    "direct_short",
    "direct_long",
    "vague_short",
    "vague_long",
    "crosslingual",
)
QUESTION_TARGET_CHARS = {
    "direct_short": 128,
    "direct_long": 512,
    "vague_short": 128,
    "vague_long": 512,
    "crosslingual": 128,
}
QUESTION_MAX_CHARS = max(QUESTION_TARGET_CHARS.values()) * 2
ANSWER_TARGET_CHARS = 64
ANSWER_MAX_CHARS = ANSWER_TARGET_CHARS + 512
SUMMARY_MAX_WORDS = 30
SUMMARY_MAX_TOKENS = 128
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


class FoundryQuestionAnswer(BaseModel):
    question: str | None = Field(default=None, max_length=QUESTION_MAX_CHARS)
    answer: str | None = Field(default=None, max_length=ANSWER_MAX_CHARS)


class EvalQuestionBatch(BaseModel):
    direct_short: QuestionCandidate | None = Field(
        default=None,
        description=f"Source-language direct question, about {QUESTION_TARGET_CHARS['direct_short']} characters.",
    )
    direct_long: QuestionCandidate | None = Field(
        default=None,
        description=f"Source-language direct question with context, about {QUESTION_TARGET_CHARS['direct_long']} characters.",
    )
    vague_short: QuestionCandidate | None = Field(
        default=None,
        description=f"Source-language indirect question, about {QUESTION_TARGET_CHARS['vague_short']} characters.",
    )
    vague_long: QuestionCandidate | None = Field(
        default=None,
        description=f"Source-language indirect question with context, about {QUESTION_TARGET_CHARS['vague_long']} characters.",
    )
    crosslingual: QuestionCandidate | None = Field(
        default=None,
        description=f"Target-language direct question, about {QUESTION_TARGET_CHARS['crosslingual']} characters, with answer in that language.",
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
Generate one question and one answer for each supported question type. Use only facts in the chunk.

Write direct_short, direct_long, vague_short, and vague_long in {source_language_name}.
Write crosslingual in {target_language_name} and set question_language to "{target_language}".
Answers must use the same language as their questions.

Question targets:
- direct_short: direct wording, about {QUESTION_TARGET_CHARS["direct_short"]} characters.
- direct_long: direct wording with context, about {QUESTION_TARGET_CHARS["direct_long"]} characters.
- vague_short: indirect wording, about {QUESTION_TARGET_CHARS["vague_short"]} characters.
- vague_long: indirect wording with context, about {QUESTION_TARGET_CHARS["vague_long"]} characters.
- crosslingual: direct wording, about {QUESTION_TARGET_CHARS["crosslingual"]} characters.

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


def _crosslingual_language(chunk_id: str) -> str:
    index = int(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id).int % len(EU_LANGUAGES))
    return EU_LANGUAGES[index]


def _question_id(chunk_id: str, question: QuestionCandidate) -> str:
    key = f"{chunk_id}\n{question.question_language}\n{question.question}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def generate_missing_with_foundry(limit: int | None, workers: int = 20) -> None:
    initialize_eval_db()
    load_local_env()
    settings = _foundry_settings()
    tasks = _missing_question_tasks(limit)
    print(f"found {len(tasks)} missing question slots", flush=True)

    inserted = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_foundry_question, task, settings) for task in tasks]
        for future in as_completed(futures):
            completed += 1
            task, item = future.result()
            if item and _valid_question(task["question_type"], item):
                with connect() as conn:
                    inserted += insert_questions(
                        conn, task["id"], [(task["question_type"], item)]
                    )
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"processed {completed}/{len(tasks)}, inserted {inserted}",
                    flush=True,
                )
    print(f"done, inserted {inserted} questions", flush=True)


def _foundry_settings() -> dict[str, str]:
    endpoint = os.environ["AZURE_AI_ENDPOINT"].rstrip("/")
    return {
        "url": f"{endpoint}/chat/completions",
        "api_key": os.environ["AZURE_AI_API_KEY"],
        "model": os.environ["AZURE_AI_MODEL"],
    }


def _missing_question_tasks(limit: int | None) -> list[dict]:
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                select c.id, c.text, c.heading_path, m.title, s.language,
                       group_concat(q.question_type, ',') as question_types
                from page_chunks c
                join page_metadata m on m.id = c.page_id
                join page_sources s on s.source = m.source
                left join eval_relevant_chunks r on r.chunk_id = c.id
                left join eval_questions q on q.id = r.question_id and q.approved = 1
                group by c.id, c.text, c.heading_path, m.title, s.language
                order by c.id
                """
            )
        ]

    tasks = []
    for chunk in rows:
        if not _is_fact_dense(chunk["text"]):
            continue
        existing = set((chunk["question_types"] or "").split(",")) - {""}
        for question_type in QUESTION_TYPES:
            if question_type not in existing:
                task = {
                    k: chunk[k]
                    for k in ("id", "text", "heading_path", "title", "language")
                }
                task["question_type"] = question_type
                task["target_language"] = _crosslingual_language(chunk["id"])
                tasks.append(task)
                if limit and len(tasks) >= limit:
                    return tasks
    return tasks


def _foundry_question(
    task: dict, settings: dict[str, str]
) -> tuple[dict, QuestionCandidate | None]:
    language = (
        task["target_language"]
        if task["question_type"] == "crosslingual"
        else task["language"]
    )
    payload = {
        "model": settings["model"],
        "messages": [
            {
                "role": "system",
                "content": _single_question_system_prompt(task, language),
            },
            {"role": "user", "content": _human_prompt(task)},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }
    try:
        with httpx.Client(timeout=90) as client:
            response = client.post(
                settings["url"],
                headers={
                    "Authorization": f"Bearer {settings['api_key']}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        qa = FoundryQuestionAnswer.model_validate_json(_json_object(content))
        if not qa.question or not qa.answer:
            return task, None
        return task, QuestionCandidate(
            question=qa.question,
            answer=qa.answer,
            question_language=language,
        )
    except (httpx.HTTPError, KeyError, ValueError, ValidationError) as exc:
        print(f"failed {task['id']} {task['question_type']}: {exc}", flush=True)
        return task, None


def _single_question_system_prompt(task: dict, language: str) -> str:
    question_type = task["question_type"]
    language_name = LANGUAGE_NAMES.get(language, language)
    source_language_name = LANGUAGE_NAMES.get(task["language"], task["language"])
    type_prompts = {
        "direct_short": f"Write a direct question in {language_name}, about {QUESTION_TARGET_CHARS['direct_short']} characters.",
        "direct_long": f"Write a direct question with context in {language_name}, about {QUESTION_TARGET_CHARS['direct_long']} characters.",
        "vague_short": f"Write an indirect question in {language_name}, about {QUESTION_TARGET_CHARS['vague_short']} characters.",
        "vague_long": f"Write an indirect question with context in {language_name}, about {QUESTION_TARGET_CHARS['vague_long']} characters.",
        "crosslingual": f"Write a direct question in {language_name}, about {QUESTION_TARGET_CHARS['crosslingual']} characters; translate the answer too.",
    }
    return f"""
Generate one question and one answer from the chunk.

Question type: {question_type}
Source language: {source_language_name}
{type_prompts[question_type]}
Use only facts explicitly present in the chunk.
Answer in the same language as the question, about {ANSWER_TARGET_CHARS} characters.
If no good question is possible, return null values.
Return only JSON: {{"question": string|null, "answer": string|null}}
"""


def _json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("response did not contain a JSON object")
    return match.group(0)


def generate_missing_summaries(limit: int | None, workers: int = 20) -> None:
    initialize_eval_db()
    load_local_env()
    settings = _foundry_settings()
    chunks = _chunks_missing_summary(limit)
    print(f"found {len(chunks)} chunks missing summaries", flush=True)

    completed = 0
    updated = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_foundry_summary, chunk, settings) for chunk in chunks]
        for future in as_completed(futures):
            completed += 1
            chunk, summary = future.result()
            if summary:
                with connect() as conn:
                    conn.execute(
                        "update page_chunks set summary = ? where id = ?",
                        (summary, chunk["id"]),
                    )
                updated += 1
            if completed % 25 == 0 or completed == len(chunks):
                print(
                    f"processed {completed}/{len(chunks)}, updated {updated}",
                    flush=True,
                )
    print(f"done, updated {updated} summaries", flush=True)


def _chunks_missing_summary(limit: int | None) -> list[dict]:
    sql_limit = "limit ?" if limit else ""
    params = (limit,) if limit else ()
    with connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                f"""
                select c.id, c.text, c.heading_path, m.title, s.language
                from page_chunks c
                join page_metadata m on m.id = c.page_id
                join page_sources s on s.source = m.source
                where c.summary is null or trim(c.summary) = ''
                order by c.id
                {sql_limit}
                """,
                params,
            )
        ]


def _foundry_summary(chunk: dict, settings: dict[str, str]) -> tuple[dict, str | None]:
    summarizer = RunnableLambda(
        lambda item: _clean_summary(_request_foundry_summary(item, settings))
    )
    try:
        summary = summarizer.with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
        ).invoke(chunk)
        return chunk, summary
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        print(f"failed summary {chunk['id']}: {exc}", flush=True)
        return chunk, None


def _request_foundry_summary(chunk: dict, settings: dict[str, str]) -> str:
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": _summary_system_prompt()},
            {"role": "user", "content": _summary_user_prompt(chunk)},
        ],
        "temperature": 0,
        "max_tokens": SUMMARY_MAX_TOKENS,
    }
    with httpx.Client(timeout=90) as client:
        response = client.post(
            settings["url"],
            headers={
                "Authorization": f"Bearer {settings['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError(f"summary exceeded {SUMMARY_MAX_TOKENS} output tokens")
    return choice["message"]["content"]


def _summary_system_prompt() -> str:
    return f"Summarize this document chunk for search embeddings in the same language. Max {SUMMARY_MAX_WORDS} words. Return only the summary."


def _summary_user_prompt(chunk: dict) -> str:
    return f"""
Title: {chunk["title"] or ""}
Heading: {chunk["heading_path"] or ""}
Language: {chunk["language"]}

Chunk:
{chunk["text"]}
"""


def _clean_summary(text: str) -> str:
    summary = re.sub(r"\s+", " ", text.strip().strip('"')).strip()
    if not summary:
        raise ValueError("empty summary")
    if len(summary.split()) > SUMMARY_MAX_WORDS:
        raise ValueError(f"summary exceeded {SUMMARY_MAX_WORDS} words")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--foundry-missing", action="store_true")
    parser.add_argument("--summarize-missing", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if args.summarize_missing:
        generate_missing_summaries(args.limit, args.workers)
    elif args.foundry_missing:
        generate_missing_with_foundry(args.limit, args.workers)
    else:
        generate_dataset(args.limit, args.model, args.reasoning or None)


if __name__ == "__main__":
    main()
