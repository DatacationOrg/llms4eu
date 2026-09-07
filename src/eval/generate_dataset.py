from __future__ import annotations

import argparse
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, model_validator

from src.db.pages import connect_pages as connect
from src.db.pages import initialize_page_artifacts_db as initialize_eval_db
from src.shared.env import load_local_env, load_yaml
from src.shared.llm import structured_local_model

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

# Chunks shorter than this are treated as too thin to carry a factual question.
# It is a real constraint at small chunk sizes rather than a formality: it excludes
# 17 of 726 `base` chunks but 612 of 2,609 `tok256` chunks, so a variant sweep must
# see the exclusion count rather than discover it as a missing question set.
MIN_CHUNK_CHARS = 300

# One question per this many of the chunk's own tokens, in density mode. It is
# what makes a chunk-size comparison fair on the question set: the legacy design
# asks one question of every type of every chunk, so a variant is probed once per
# *chunk* and tok256 gets 4x tok1024's questions per character of the same corpus
# — 7,801 against 1,451 in the 2026-08-18 sweep. Scaling the count with the chunk
# instead gives 256 one question, 512 two and 1024 four, so every variant covers
# the corpus at one question per 256 tokens and their totals match.
#
# 256 rather than any other figure because it is the smallest variant in the grid,
# which makes its budget exactly 1 and every larger one a whole multiple.
QUESTION_DENSITY_TOKENS = 256

# Fallback ratio when no local tokenizer can be loaded, so density mode degrades
# to a character proxy instead of refusing to run. Measured on this Slovenian
# corpus (src/eval/config.yaml carries the same figure for the budget metrics);
# it is a corpus constant, not a model one, so a different language needs its own.
TOKENS_PER_CHAR = 0.41

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
        default="",
        description="BCP-47/ISO-style language code of the question text.",
    )


class EvalQuestionBatch(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, value):
        """Accept the shapes the model returns instead of one object per chunk.

        Measured over 550 tok256 chunks: 11 of 28 failures were the whole object
        wrapped in a list, or a list of per-type items carrying their own `type`
        key. Asking for a *subset* of the types — density mode — made a list the
        model's usual answer rather than an occasional one, and on 8 real tok1024
        chunks every reply was one of these five shapes, each holding complete and
        usable questions:

            [{"vague_long": "<question>", "answer": "...", "direct_short": null}]
            [{"vague_short": {"question": "...", "answer": "..."}}, ...]
            [{"question_type": "crosslingual", "question": "...", "answer": "..."}]
            {"direct_long": {"question": null, "answer": null}, ...}
            {"direct_short_2": {...}}      invented key beyond the vocabulary

        All five are normalised here rather than retried. The remaining failures
        are prose rather than JSON, which is a retry rather than something to
        parse.
        """
        if isinstance(value, list):
            if not value:
                return value
            merged = _merge_type_items(value)
            return merged or (value[0] if len(value) == 1 else value)
        if isinstance(value, dict):
            return _clean_type_fields(value)
        return value

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


# Density mode reuses the one structured-output shape measured reliable for this
# model — the multi-field `EvalQuestionBatch` through `json_schema` — and changes
# only *which* of its fields are asked for. A 256-token chunk is asked for one
# type, a 512-token chunk two, a 1,024-token chunk four.
#
# The shapes that would have allowed an arbitrary count were measured first, on 4
# real tok1024 chunks asking for 4 pairs each:
#
#   one object holding a list of pairs, json_schema        0/4
#   one object holding a list of pairs, function_calling   2/4
#   one field per numbered slot, json_schema               0/4
#   one field per numbered slot, function_calling          1/4
#   one question per call (a two-field pair, tools)         1/28 slots
#
# None is usable for a benchmark. `src/eval/config.yaml` already records the
# reason for half of it: this model returns a bare value rather than an object for
# a single-field schema, and function calling is what gpt-oss was validated on,
# not gemma4.
#
# Staying on the proven shape costs two things. A ceiling of five questions per
# chunk, since a type cannot be filled twice — `density_budget` enforces and
# reports it, and it binds above about 1,280 tokens. And a lower yield the more
# questions one call is asked for: measured over 163 real chunks, tok256 delivered
# 49 of 49 requested, tok512 112 of 114, tok1024 172 of 215. Asking for a subset
# of the types also made a list reply the norm rather than the exception, which is
# what `EvalQuestionBatch._unwrap` had to grow to absorb — on the first 8 tok1024
# chunks it took the yield from 0 of 36 to 31 of 36.


def density_budget(
    text: str,
    unit_tokens: int = QUESTION_DENSITY_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
    maximum: int = len(QUESTION_TYPES),
) -> int:
    """How many questions this chunk is worth, at one per `unit_tokens`.

    Rounded from the chunk's *actual* token count rather than read off its
    variant's nominal size, which matters at both ends of a page: the last chunk
    of a page is routinely a third of its target and would otherwise be asked for
    four questions it cannot support, and a variant cut with overlap has no single
    nominal size at all. Never below one, so no chunk long enough to pass the
    length floor is left unprobed.

    Capped at `maximum`, which is the number of question types, because one field
    per type is what makes the request reliable and a type cannot be filled twice.
    That caps the design at a 1,280-token cut; a longer one would be probed less
    densely than the rest and the cap is reported rather than applied quietly.
    """
    tokens = count_tokens(text) if count_tokens else int(len(text) * TOKENS_PER_CHAR)
    return max(1, min(round(tokens / unit_tokens), maximum))


def density_counter(provider: str | None = "qwen") -> Callable[[str], int] | None:
    """The tokenizer the variants were cut with, or None to fall back to chars.

    Density mode is a fairness argument about tokens, so it counts them with the
    same tokenizer that decided the chunk boundaries. It does not *require* that
    tokenizer: a machine with no local weights still gets a usable budget from the
    character proxy, and says so rather than failing an hours-long run at the
    first chunk.
    """
    if not provider:
        return None
    try:
        from src.shared.tokenizers import token_counter

        return token_counter(provider)
    except Exception as error:
        print(
            f"density: no local tokenizer for {provider} ({error}); "
            f"budgeting from characters at {TOKENS_PER_CHAR} tokens/char",
            flush=True,
        )
        return None


def slot_types(chunk_id: str, count: int) -> list[str]:
    """Which question types this chunk's `count` slots ask for.

    Rotated by chunk id rather than fixed, because in density mode a chunk does
    not get every type: a 256-token chunk gets exactly one. Taking that one from
    the head of the list would make the whole tok256 question set `direct_short`
    and delete the category breakdown; rotating spreads the five types evenly over
    a variant's chunks, so every variant's per-type sample stays about a fifth of
    its total and the category table keeps comparing like with like.
    """
    offset = int(uuid.uuid5(uuid.NAMESPACE_URL, f"slots:{chunk_id}").int) % len(
        QUESTION_TYPES
    )
    return [QUESTION_TYPES[(offset + index) % len(QUESTION_TYPES)] for index in range(count)]


def _merge_type_items(items: list) -> dict:
    """Fold a list of partial per-type objects into one batch object.

    Each item names one or more question types, either as a key or through its own
    `type`/`question_type` field, and a type's payload is either the pair object or
    the bare question string with `answer` alongside it.
    """
    rebuilt: dict = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        tagged = item.get("type") or item.get("question_type")
        if tagged in QUESTION_TYPES:
            rebuilt.setdefault(tagged, _pair(item.get("question"), item))
            continue
        for name in QUESTION_TYPES:
            if item.get(name) is None:
                continue
            payload = item[name]
            rebuilt.setdefault(
                name,
                _clean_pair(payload)
                if isinstance(payload, (dict, QuestionCandidate))
                else _pair(payload, item),
            )
    return {name: pair for name, pair in rebuilt.items() if pair}


def _pair(question, item: dict) -> dict | None:
    """One question type's pair, taking `answer` from its sibling keys."""
    return _clean_pair(
        {
            "question": question,
            "answer": item.get("answer"),
            "question_language": item.get("question_language", ""),
        }
    )


def _clean_pair(payload):
    """A pair, or None where the model filled the object with nulls.

    `{"question": null, "answer": null}` is how it declines a type it was asked
    for. Left as-is it fails validation on the whole batch and takes the types it
    *did* answer down with it, which is the difference between a chunk yielding
    three questions and none.

    An already-built `QuestionCandidate` passes straight through: this validator
    runs on direct construction too, not only on parsed model output.
    """
    if isinstance(payload, QuestionCandidate):
        return payload
    if not isinstance(payload, dict):
        return None
    question = payload.get("question")
    answer = payload.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    if not question.strip() or not answer.strip():
        return None
    language = payload.get("question_language") or ""
    return {"question": question, "answer": answer, "question_language": language}


def _clean_type_fields(value: dict) -> dict:
    """Keep the known types that hold a real pair, drop the rest.

    Unknown keys are dropped rather than passed through — the model invents
    `direct_short_2` when asked for more questions than there are types — and a
    known key holding nulls is dropped for the reason `_clean_pair` gives.
    """
    if any(name in value for name in QUESTION_TYPES):
        return {
            name: pair
            for name in QUESTION_TYPES
            if name in value
            and (
                pair := (
                    _clean_pair(value[name])
                    if isinstance(value[name], (dict, QuestionCandidate))
                    else _pair(value[name], value)
                )
            )
        }
    return value


def generate_dataset(
    limit: int | None,
    model_id: str | None = None,
    reasoning: bool | str | None = None,
    variant: str | None = None,
    workers: int = 1,
    min_chars: int = MIN_CHUNK_CHARS,
    density_tokens: int | None = None,
    density_provider: str = "qwen",
) -> None:
    """Generate questions from chunks that have none yet.

    `variant` restricts generation to one chunking. Each variant then owns its own
    questions, because a question's id derives from the chunk id and chunk ids
    carry the variant — so no schema change, and `load_eval_rows(variant)` already
    returns only that variant's questions.

    That is the point of generating per variant: a question written from a `base`
    chunk is answerable from a `base` chunk by construction, which flatters `base`
    and any cutting close to it. Per-variant questions remove that. They introduce
    a different confound instead — a variant with more chunks is a harder haystack,
    2,609 candidates against 618 — so the two designs are cross-checks on each
    other, not one replacing the other.

    `density_tokens` sets how many questions a chunk is asked for: one per that
    many of its own tokens, so 256 gets one, 512 two and 1,024 four. Without it
    every chunk is asked for one question of every type regardless of size, which
    probes the corpus once per chunk — tok256 four times more densely than tok1024
    over the same pages, and with the harder facts, since squeezing five questions
    out of 256 tokens reaches further down the chunk than five out of 1,024. That
    is a bias in the large cut's favour and it is the one the 2026-08-18 sweep's
    every-row win for tok1024 cannot be separated from.

    What density mode does *not* fix: each variant is still scored on questions
    written from its own chunks, with its own boundaries in view. Equalising the
    counts is what makes those per-variant sets poolable — anchor them
    (`src/eval/anchors.py`), project them onto every variant
    (`src/eval/relabel.py`) and score every variant on the union — and only the
    pooled set removes home-turf advantage as well as density.
    """
    initialize_eval_db()
    chunks, excluded = _eligible_unprocessed_chunks(limit, variant, min_chars)
    scope = f" in {variant}" if variant else ""
    print(
        f"found {len(chunks)} eligible unprocessed chunks{scope}; "
        f"{excluded} excluded as too short or not fact-dense "
        f"(min_chars={min_chars})",
        flush=True,
    )
    if not chunks:
        return
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
    count_tokens = density_counter(density_provider) if density_tokens else None
    budgets: dict[str, int] = {}
    if density_tokens:
        budgets = {
            chunk["id"]: density_budget(chunk["text"], density_tokens, count_tokens)
            for chunk in chunks
        }
        requested = sum(budgets.values())
        capped = sum(
            1
            for chunk in chunks
            if density_budget(chunk["text"], density_tokens, count_tokens, maximum=10**6)
            > len(QUESTION_TYPES)
        )
        print(
            f"density: one question per {density_tokens} tokens, "
            f"{requested} questions across {len(chunks)} chunks "
            f"(mean {requested / len(chunks):.2f} per chunk)",
            flush=True,
        )
        if capped:
            # Silence here would read as a variant that simply scored differently,
            # when what happened is that it was probed less densely than the rest.
            print(
                f"density: {capped} chunks hit the {len(QUESTION_TYPES)}-question "
                f"cap, so this variant is probed below the requested density. "
                f"Raise --density-tokens or drop the variant from the grid.",
                flush=True,
            )

    inserted = 0
    completed = 0
    # Model calls run in parallel, writes stay on this thread. One chunk yields its
    # whole question budget in a single call, so this path is far cheaper per
    # question than --fill-missing; sequentially it was also too slow to be usable
    # across five variants (4,700 chunks).
    def submit(pool, chunk):
        if density_tokens:
            types = slot_types(chunk["id"], budgets[chunk["id"]])
            return pool.submit(_chunk_questions, chunk, structured_model, types)
        return pool.submit(_chunk_questions, chunk, structured_model)

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {submit(pool, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            completed += 1
            chunk, questions = future.result()
            if questions:
                with connect() as conn:
                    inserted += insert_questions(conn, chunk["id"], questions)
            if completed % 25 == 0 or completed == len(chunks):
                print(
                    f"[{completed}/{len(chunks)}] inserted {inserted} questions",
                    flush=True,
                )
    print(
        f"processed {len(chunks)} chunks, inserted up to {inserted} questions",
        flush=True,
    )
    if budgets:
        # The design's own claim is that every variant ends up with the same number
        # of questions over the same corpus. A shortfall breaks that claim quietly,
        # so it is reported next to the total rather than left to be discovered as
        # a differently sized column three hours into a sweep.
        requested = sum(budgets.values())
        print(
            f"density: {inserted} of {requested} requested questions "
            f"({1 - inserted / requested:.1%} short; refusals, duplicates within a "
            f"chunk, and model failures all land here)",
            flush=True,
        )


def _chunk_questions(
    chunk: dict,
    structured_model,
    types: tuple[str, ...] = QUESTION_TYPES,
) -> tuple[dict, list[tuple[str, QuestionCandidate]]]:
    """The requested question types for one chunk, in one call.

    `types` is every type by default and the chunk's density budget's worth in
    density mode — the only difference between the two designs. Fields outside the
    request are ignored even when the model fills them anyway, because the density
    claim is a count and a model being generous would break it.

    One bad chunk must not end the run.
    """
    target_language = _crosslingual_language(chunk["id"])
    try:
        result = structured_model.invoke(_messages(chunk, target_language, types))
    except Exception as error:
        # Deliberately broad. This runs for hours over thousands of chunks, and
        # the parser raises its own exception type rather than ValidationError,
        # which is how one malformed answer killed a whole run.
        print(f"failed {chunk['id']}: {type(error).__name__}: {error}", flush=True)
        return chunk, []

    questions = []
    seen: set[str] = set()
    for question_type, item in _flatten_questions(result, types):
        item = _with_language(item, question_type, chunk, target_language)
        if not _valid_question(question_type, item):
            continue
        # Two slots of one chunk asking the same thing share a question id, so
        # `insert or ignore` would collapse them and the count would come out
        # short with nothing said. Under density mode that count is the claim.
        key = " ".join(item.question.lower().split())
        if key in seen:
            continue
        seen.add(key)
        questions.append((question_type, item))
    return chunk, questions


def _with_language(
    item: QuestionCandidate,
    question_type: str,
    chunk: dict,
    target_language: str,
) -> QuestionCandidate:
    """Fill in the question's language when the model left it out.

    It was never something the model had to tell us: every type but
    `crosslingual` is written in the chunk's own language, and `crosslingual` is
    written in the language we asked for. Requiring the field turned a routine
    omission into a parse error that aborted the run.
    """
    if item.question_language:
        return item
    language = target_language if question_type == "crosslingual" else chunk["language"]
    return item.model_copy(update={"question_language": language})


def _eligible_unprocessed_chunks(
    limit: int | None,
    variant: str | None = None,
    min_chars: int = MIN_CHUNK_CHARS,
) -> tuple[list[dict], int]:
    """Chunks with no question yet, and how many were filtered out."""
    sql = """
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
        """
    params: list = []
    if variant is not None:
        sql += "\n          and c.variant = ?"
        params.append(variant)
    sql += "\n        order by c.id"
    with connect() as conn:
        rows = [dict(row) for row in conn.execute(sql, params)]
    chunks = [row for row in rows if _is_fact_dense(row["text"], min_chars)]
    return (chunks[:limit] if limit else chunks), len(rows) - len(chunks)


def _resolve_model_name(model_name: str) -> str:
    return CONFIG.get("model_aliases", {}).get(model_name, model_name)


def _is_fact_dense(text: str, min_chars: int = MIN_CHUNK_CHARS) -> bool:
    lowered = text.lower()
    if len(text) < min_chars:
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


def _flatten_questions(
    batch: EvalQuestionBatch, types: tuple[str, ...] = QUESTION_TYPES
) -> list[tuple[str, QuestionCandidate]]:
    return [
        (question_type, item)
        for question_type in types
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
    if not question or not answer or not item.question_language:
        return False
    return (
        len(answer) <= ANSWER_MAX_CHARS
        and len(question) <= QUESTION_TARGET_CHARS[question_type] * 2
    )


def _messages(
    chunk: dict,
    target_language: str | None,
    types: tuple[str, ...] = QUESTION_TYPES,
) -> list[tuple[str, str]]:
    return [
        ("system", _system_prompt(chunk, target_language, types)),
        ("human", _human_prompt(chunk)),
    ]


# Per-type wording, so the prompt can name only the types being asked for. Kept
# apart from the prompt text because density mode requests a subset: a 256-token
# chunk is asked for one type and a 1,024-token chunk for four, and listing the
# other four anyway is how a model ends up filling them.
TYPE_INSTRUCTIONS = {
    "direct_short": f"direct wording, about {QUESTION_TARGET_CHARS['direct_short']} characters",
    "direct_long": f"direct wording with context, about {QUESTION_TARGET_CHARS['direct_long']} characters",
    "vague_short": f"indirect wording, about {QUESTION_TARGET_CHARS['vague_short']} characters",
    "vague_long": f"indirect wording with context, about {QUESTION_TARGET_CHARS['vague_long']} characters",
    "crosslingual": f"direct wording, about {QUESTION_TARGET_CHARS['crosslingual']} characters",
}


def _system_prompt(
    chunk: dict,
    target_language: str | None,
    types: tuple[str, ...] = QUESTION_TYPES,
) -> str:
    """Ask for `types` and nothing else.

    The shape of this prompt and its schema is the one combination measured
    reliable for this model: a multi-field object through `json_schema`. A single
    object holding a list of pairs, and one field per numbered slot, both failed
    on 4 real tok1024 chunks (0/4 and 0/4 through `json_schema`, 2/4 and 1/4
    through function calling), and one question per call failed outright. Density
    mode therefore changes *which* fields are requested rather than the schema.

    The spread instruction is what carries the fairness argument when more than
    one is asked for. Four questions taken from a 1,024-token chunk with no such
    constraint cluster in its opening, which probes the same text a single
    256-token question would and leaves three quarters of the chunk untested — the
    coverage difference density mode exists to remove, reintroduced inside the
    chunk.
    """
    source_language_name = LANGUAGE_NAMES.get(chunk["language"], chunk["language"])
    target_language = target_language or "en"
    target_language_name = LANGUAGE_NAMES[target_language]
    source_types = [name for name in types if name != "crosslingual"]
    language_lines = []
    if source_types:
        language_lines.append(
            f"Write {', '.join(source_types)} in {source_language_name}."
        )
    if "crosslingual" in types:
        language_lines.append(
            f'Write crosslingual in {target_language_name} and set '
            f'question_language to "{target_language}".'
        )
    targets = "\n".join(
        f"- {name}: {TYPE_INSTRUCTIONS[name]}." for name in types
    )
    count = len(types)
    spread = (
        ""
        if count == 1
        else (
            f"\nThe {count} questions must be about {count} different facts, spread "
            f"over the whole chunk: divide it into {count} consecutive stretches of "
            f"roughly equal length and take one question from each. Do not ask "
            f"{count} questions about the same sentence or paragraph.\n"
        )
    )
    return f"""
Generate one question and one answer for each of the {count} question types listed below. Use only facts in the chunk.

{chr(10).join(language_lines)}
Answers must use the same language as their questions.

Question targets:
{targets}
{spread}
Answer target: about {ANSWER_TARGET_CHARS} characters.
If a type is not supported by the chunk, leave it null.
Leave every field not listed above null.

Return one JSON object whose keys are those type names, each holding an object with "question" and "answer". Do not return a list.
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
    variant: str | None = None,
    min_chars: int = MIN_CHUNK_CHARS,
    density_tokens: int | None = None,
    density_provider: str = "qwen",
) -> None:
    """Fill the question slots a batch run left empty, one call per slot.

    `density_tokens` must match the value the variant was generated with. Without
    it this offers every type for every chunk, which is how the obvious remedy for
    a short density run — top up the gaps — would quietly restore the very
    per-type design density mode replaced, and by then the counts look right.
    """
    initialize_eval_db()
    load_local_env()
    # The same schema and method the batch pass uses, asked for one type instead
    # of several. The old path here had its own two-field schema through function
    # calling and produced nothing at all with this question model — measured 1 of
    # 28 slots, then 0 of 6 — which made the natural remedy for a short density run
    # unavailable exactly when it was needed. A one-type request through the shape
    # that works is what tok256 already runs at 49 of 49.
    structured_model = structured_local_model(
        _resolve_model_name(model_id or CONFIG["question_model"]),
        EvalQuestionBatch,
        reasoning=CONFIG["question_model_reasoning"],
        num_ctx=CONFIG["question_model_num_ctx"],
        num_predict=CONFIG["question_model_num_predict"],
    )
    tasks = _missing_question_tasks(
        limit, variant, min_chars, density_tokens, density_provider
    )
    scope = f" in {variant}" if variant else ""
    density = f" within the {density_tokens}-token density budget" if density_tokens else ""
    print(f"found {len(tasks)} missing question slots{scope}{density}", flush=True)

    inserted = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _chunk_questions, task, structured_model, (task["question_type"],)
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            completed += 1
            task, questions = future.result()
            if questions:
                with connect() as conn:
                    inserted += insert_questions(conn, task["id"], questions)
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"processed {completed}/{len(tasks)}, inserted {inserted}",
                    flush=True,
                )
    print(f"done, inserted {inserted} questions", flush=True)


def _missing_question_tasks(
    limit: int | None,
    variant: str | None = None,
    min_chars: int = MIN_CHUNK_CHARS,
    density_tokens: int | None = None,
    density_provider: str = "qwen",
) -> list[dict]:
    sql = """
        select c.id, c.text, c.heading_path, m.title,
               coalesce(m.language, s.language) as language,
               group_concat(q.question_type, ',') as question_types
        from page_chunks c
        join page_metadata m on m.id = c.page_id
        join page_sources s on s.source = m.source
        left join eval_relevant_chunks r on r.chunk_id = c.id
        left join eval_questions q on q.id = r.question_id and q.approved = 1
        """
    params: list = []
    if variant is not None:
        sql += "\n        where c.variant = ?"
        params.append(variant)
    # Group on the expression, not on the `language` alias: `page_metadata` and
    # `page_sources` both have that column, so SQLite calls the bare name
    # ambiguous and the query fails outright.
    sql += """
        group by c.id, c.text, c.heading_path, m.title,
                 coalesce(m.language, s.language)
        order by c.id
        """
    with connect() as conn:
        rows = [dict(row) for row in conn.execute(sql, params)]

    count_tokens = density_counter(density_provider) if density_tokens else None
    tasks = []
    for chunk in rows:
        if not _is_fact_dense(chunk["text"], min_chars):
            continue
        existing = set((chunk["question_types"] or "").split(",")) - {""}
        wanted = (
            slot_types(
                chunk["id"],
                density_budget(chunk["text"], density_tokens, count_tokens),
            )
            if density_tokens
            else QUESTION_TYPES
        )
        for question_type in wanted:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--fill-missing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Generate questions only from this chunk variant's chunks, so the "
            "variant owns its own question set instead of inheriting base's."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=MIN_CHUNK_CHARS,
        help=(
            "Skip chunks shorter than this. The default excludes a quarter of "
            "tok256's chunks, so lower it when generating for small variants."
        ),
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help=(
            "Scale each chunk's question count with its size, so every variant "
            "is probed at one question per --density-tokens tokens: 256 gets "
            "one question, 512 two, 1024 four. Without this a chunk of any size "
            "gets one question per type, which probes tok256 four times more "
            "densely than tok1024 over the same corpus."
        ),
    )
    parser.add_argument(
        "--density-tokens",
        type=int,
        default=CONFIG.get("question_density_tokens", QUESTION_DENSITY_TOKENS),
        help="Tokens of chunk per question in --density mode.",
    )
    parser.add_argument(
        "--density-provider",
        default=CONFIG.get("question_density_provider", "qwen"),
        help=(
            "Whose tokenizer counts the tokens. Defaults to the provider the "
            "token-sized variants were cut with, so the budget and the chunk "
            "boundaries are measured the same way."
        ),
    )
    args = parser.parse_args()
    if args.fill_missing:
        generate_missing_questions(
            args.limit,
            args.workers,
            args.model,
            args.variant,
            args.min_chars,
            density_tokens=args.density_tokens if args.density else None,
            density_provider=args.density_provider,
        )
    else:
        generate_dataset(
            args.limit,
            args.model,
            args.reasoning or None,
            args.variant,
            args.workers,
            args.min_chars,
            density_tokens=args.density_tokens if args.density else None,
            density_provider=args.density_provider,
        )


if __name__ == "__main__":
    main()
