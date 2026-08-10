from __future__ import annotations

import argparse
import re
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.db.pages import connect_pages as connect
from src.db.pages import initialize_page_artifacts_db as initialize_eval_db
from src.indexing.chunk_text import BASE_CHUNK_VARIANT
from src.shared.env import load_local_env, load_yaml
from src.shared.llm import (
    LocalOllamaStructuredLlm,
    StructuredLlm,
    structured_local_model,
)

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


class SingleQuestionAnswer(BaseModel):
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
    variant: str = BASE_CHUNK_VARIANT,
    sample: int | None = None,
) -> None:
    initialize_eval_db()
    chunks = _eligible_unprocessed_chunks(limit, variant, sample)
    print(
        f"found {len(chunks)} eligible unprocessed {variant} chunks",
        flush=True,
    )
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
            inserted += insert_questions(conn, chunk["id"], questions, variant)
            print(f"[{index}/{len(chunks)}] got {len(questions)} questions", flush=True)
    print(
        f"processed {len(chunks)} chunks, inserted up to {inserted} questions",
        flush=True,
    )


def _eligible_unprocessed_chunks(
    limit: int | None,
    variant: str = BASE_CHUNK_VARIANT,
    sample: int | None = None,
) -> list[dict]:
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                select c.id, c.text, c.heading_path, m.title, m.source,
                       m.page_kind, s.language
                from page_chunks c
                join page_metadata m on m.id = c.page_id
                join page_sources s on s.source = m.source
                where c.variant = ?
                  and not exists (
                    select 1
                    from eval_relevant_chunks r
                    where r.chunk_id = c.id
                  )
                order by c.id
                """,
                (variant,),
            )
        ]
    chunks = [row for row in rows if _is_fact_dense(row["text"])]
    if sample:
        chunks = _stratified_sample(chunks, sample)
    return chunks[:limit] if limit else chunks


def _stratified_sample(chunks: list[dict], sample: int) -> list[dict]:
    """Pick `sample` chunks spread over source, page kind and chunk length.

    A finer variant has thousands of chunks, and generating five questions for
    every one of them costs more than the ablation needs. Sampling is
    deterministic — strata are visited in a fixed interleaved order and chunks
    within a stratum in `uuid5` order — so a rerun selects the same chunks
    without carrying a random seed.
    """
    if sample >= len(chunks):
        return chunks

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for chunk in chunks:
        key = (chunk["source"], chunk["page_kind"], _length_bucket(chunk["text"]))
        strata[key].append(chunk)
    for items in strata.values():
        items.sort(key=lambda chunk: uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"]).int)

    keys = _interleave(list(strata))
    picked: list[dict] = []
    depth = 0
    while len(picked) < sample:
        progressed = False
        for key in keys:
            if depth < len(strata[key]):
                picked.append(strata[key][depth])
                progressed = True
                if len(picked) == sample:
                    break
        if not progressed:
            break
        depth += 1
    return sorted(picked, key=lambda chunk: chunk["id"])


def _interleave(keys: list[tuple]) -> list[tuple]:
    """Order stratum keys so each dimension rotates before the next repeats.

    Visiting sorted keys instead would drain the alphabetically first source
    before reaching any other, which is exactly the imbalance stratifying is
    supposed to prevent when there are more strata than the sample size.
    """
    if not keys:
        return []
    if len(keys[0]) == 1:
        return sorted(keys)

    tails: dict = defaultdict(list)
    for key in keys:
        tails[key[0]].append(key[1:])
    ordered = {head: _interleave(items) for head, items in tails.items()}

    result: list[tuple] = []
    heads = sorted(ordered)
    depth = 0
    while len(result) < len(keys):
        for head in heads:
            if depth < len(ordered[head]):
                result.append((head, *ordered[head][depth]))
        depth += 1
    return result


def _length_bucket(text: str) -> int:
    return len(text) // 500


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
    variant: str = BASE_CHUNK_VARIANT,
) -> int:
    inserted = 0
    for question_type, item in questions:
        question_id = _question_id(chunk_id, item)
        cursor = conn.execute(
            """
            insert or ignore into eval_questions (
              id, question, answer, question_type, question_language, variant
            ) values (?, ?, ?, ?, ?, ?)
            """,
            (
                question_id,
                item.question.strip(),
                item.answer.strip(),
                question_type,
                item.question_language,
                variant,
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


def generate_missing_questions(
    limit: int | None,
    workers: int = 4,
    model_id: str | None = None,
    variant: str = BASE_CHUNK_VARIANT,
) -> None:
    initialize_eval_db()
    load_local_env()
    client = LocalOllamaStructuredLlm(
        model_id=_resolve_model_name(model_id or CONFIG["question_model"]),
        reasoning=CONFIG["question_model_reasoning"],
        num_ctx=CONFIG["question_model_num_ctx"],
        num_predict=512,
        method="function_calling",
    )
    tasks = _missing_question_tasks(limit, variant)
    print(f"found {len(tasks)} missing {variant} question slots", flush=True)

    inserted = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_single_question, task, client) for task in tasks]
        for future in as_completed(futures):
            completed += 1
            task, item = future.result()
            if item and _valid_question(task["question_type"], item):
                with connect() as conn:
                    inserted += insert_questions(
                        conn, task["id"], [(task["question_type"], item)], variant
                    )
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"processed {completed}/{len(tasks)}, inserted {inserted}",
                    flush=True,
                )
    print(f"done, inserted {inserted} questions", flush=True)


def _missing_question_tasks(
    limit: int | None,
    variant: str = BASE_CHUNK_VARIANT,
) -> list[dict]:
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
                where c.variant = ?
                group by c.id, c.text, c.heading_path, m.title, s.language
                order by c.id
                """,
                (variant,),
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


def _single_question(
    task: dict,
    client: StructuredLlm,
) -> tuple[dict, QuestionCandidate | None]:
    language = (
        task["target_language"]
        if task["question_type"] == "crosslingual"
        else task["language"]
    )
    try:
        qa = client.structured_output(
            [
                ("system", _single_question_system_prompt(task, language)),
                ("human", _human_prompt(task)),
            ],
            SingleQuestionAnswer,
            retries=3,
        )
        if not qa.question or not qa.answer:
            return task, None
        return task, QuestionCandidate(
            question=qa.question,
            answer=qa.answer,
            question_language=language,
        )
    except (RuntimeError, ValidationError) as exc:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--fill-missing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--variant",
        default=BASE_CHUNK_VARIANT,
        help="Chunking variant to generate questions for",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Generate for this many chunks, stratified over source, kind and length",
    )
    args = parser.parse_args()
    if args.fill_missing:
        generate_missing_questions(args.limit, args.workers, args.model, args.variant)
    else:
        generate_dataset(
            args.limit,
            args.model,
            args.reasoning or None,
            args.variant,
            args.sample,
        )


if __name__ == "__main__":
    main()
