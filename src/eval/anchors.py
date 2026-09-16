"""Locate the text that supports each answer, once, in page coordinates.

A chunk-size ablation needs to know which chunk holds the answer under every
cutting. Asking a model to pick among candidate chunks does that once per
variant, and each choice is an unverifiable judgement — the noise lands directly
in the numbers being compared, and it lands asymmetrically, because `base`'s own
labels were never judged at all (its questions were generated *from* their gold
chunk, so that chunk is gold by construction).

Anchoring removes the judgement instead of repeating it. One pass asks the model
to quote, verbatim, the sentences in the known gold chunk that support the
answer. That output is *checkable*: a quote either appears in the chunk or it
does not, so a bad extraction is rejected and retried rather than silently
recorded. The quote is then stored as a character span in the **page**, which no
chunking can invalidate, and labelling a variant becomes interval overlap.

Three consequences worth having:

- one model pass total, not one per variant;
- a quote split across a chunk boundary overlaps both chunks and labels both,
  which is the right answer rather than a failure case;
- every variant's labels, and base's, derive from the same anchor.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from src.db.pages import connect_pages as connect
from src.db.pages import initialize_page_artifacts_db
from src.preprocess.chunkers import normalize
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.shared.env import load_yaml
from src.shared.llm import (
    LocalOllamaStructuredLlm,
    StructuredLlm,
    run_structured_outputs,
)

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

# Not the question model. Extracting its own wording's source would be the more
# elegant choice, but gemma4 emits a bare string rather than the object for a
# single-field schema and only 15/100 quotes landed. Format reliability wins.
DEFAULT_ANCHOR_MODEL = CONFIG.get("anchor_model", "gpt-oss:20b")
DEFAULT_STRUCTURED_METHOD = CONFIG.get("anchor_structured_method", "function_calling")
DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 100

# Below this, a match carries no information: "895" occurs all over a page, so
# finding it says nothing about where the answer is. Above it, a repeated quote
# is disambiguated rather than refused — see `_place`.
MIN_QUOTE_CHARS = 8

# Long enough that two occurrences are the same passage appearing twice rather
# than two different facts, so the first is as good as the second.
UNAMBIGUOUS_CHARS = 24

# Models quote what a reader sees, not the Markdown source. Asked to copy
# "... je bilo v [zasavski regiji](https://sl.wikipedia.org/wiki/...)" they
# return "... je bilo v zasavski regiji", which is a faithful quote and was the
# largest single source of rejections. Comparing both sides with the markup
# removed accepts it; the offset map is what keeps the resulting span pointing at
# real page coordinates.
_MARKUP = re.compile(
    r"!?\[(?P<label>[^\]\n]*)\]\([^)\n]*\)"  # inline link or image
    r"|\*+|_{2,}|`+"  # emphasis and code markers
)


class AnswerQuote(BaseModel):
    """The verbatim source of an answer inside its chunk.

    Two fields on purpose. A single-field schema invites a model to return the
    bare value instead of the object, which is exactly how the first attempt
    failed; `found` also lets the model say the chunk does not state the answer
    rather than inventing a quote to fill the field.
    """

    found: bool = Field(
        description="True if the chunk states the answer, false if it does not.",
    )
    quote: str = Field(
        default="",
        description=(
            "The sentence or sentences from the chunk that state the answer, "
            "copied exactly, character for character, with nothing added. "
            "Empty when found is false."
        ),
    )


@dataclass
class AnchorSummary:
    questions: int = 0
    anchored: int = 0
    rejected: int = 0
    failed: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(self.reasons.items()))
        return (
            f"anchors: {self.anchored}/{self.questions} placed, "
            f"{self.rejected} rejected, {self.failed} model failures"
            + (f" ({detail})" if detail else "")
        )


def collapse_with_offsets(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text, plus each kept character's original index.

    Chunk text is re-joined and re-wrapped by the chunkers, so a quote matches
    the page on its words but not on its whitespace. Collapsing both sides makes
    the match work; keeping the index map is what lets the match be turned back
    into a real span in the original text.
    """
    collapsed: list[str] = []
    offsets: list[int] = []
    previous_was_space = True
    for index, character in enumerate(text):
        if character.isspace():
            if previous_was_space:
                continue
            collapsed.append(" ")
            offsets.append(index)
            previous_was_space = True
            continue
        collapsed.append(character)
        offsets.append(index)
        previous_was_space = False
    while collapsed and collapsed[-1] == " ":
        collapsed.pop()
        offsets.pop()
    return "".join(collapsed), offsets


def render_with_offsets(text: str) -> tuple[str, list[int]]:
    """Text as a reader sees it, plus each kept character's original index.

    Link and image syntax collapses to its label and emphasis markers disappear,
    so a quote of the rendered text can still be matched against the Markdown it
    came from. The offsets are what let the match be turned back into a span in
    the original source.
    """
    kept: list[str] = []
    offsets: list[int] = []

    def keep(start: int, stop: int) -> None:
        kept.extend(text[start:stop])
        offsets.extend(range(start, stop))

    cursor = 0
    for match in _MARKUP.finditer(text):
        keep(cursor, match.start())
        if match.group("label"):
            keep(match.start("label"), match.end("label"))
        cursor = match.end()
    keep(cursor, len(text))
    return "".join(kept), offsets


def rendered(text: str) -> str:
    """`text` with Markdown link and emphasis markup removed."""
    return render_with_offsets(text)[0]


def _searchable(text: str) -> tuple[str, list[int]]:
    """Comparable form of `text`: rendered, whitespace-collapsed, still mappable."""
    plain, plain_offsets = render_with_offsets(text)
    collapsed, collapsed_offsets = collapse_with_offsets(plain)
    return collapsed, [plain_offsets[index] for index in collapsed_offsets]


def find_spans(quote: str, text: str) -> list[tuple[int, int]]:
    """Every span of `quote` in `text`, ignoring whitespace and Markdown markup."""
    needle, _ = _searchable(normalize(quote))
    if len(needle) < MIN_QUOTE_CHARS:
        return []
    haystack, offsets = _searchable(text)
    spans: list[tuple[int, int]] = []
    position = haystack.find(needle)
    while position >= 0:
        # The end offset is exclusive, so take the last matched character's index + 1.
        spans.append((offsets[position], offsets[position + len(needle) - 1] + 1))
        position = haystack.find(needle, position + 1)
    return spans


def locate(quote: str, text: str, *, near: tuple[int, int] | None = None):
    """Span of `quote` inside `text`, or None when it cannot be placed."""
    return _place(quote, text, near)[0]


def _place(
    quote: str, text: str, near: tuple[int, int] | None = None
) -> tuple[tuple[int, int] | None, str | None]:
    """`locate` plus the reason it failed, so rejections can be counted apart.

    A repeated quote is not automatically unusable. When the chunk the question
    was written from is known, the occurrence inside it is the right one; a long
    quote's repeats are the same passage twice. Only a short quote with no window
    is genuinely ambiguous, which is the case worth refusing.
    """
    needle, _ = _searchable(normalize(quote))
    if len(needle) < MIN_QUOTE_CHARS:
        return None, "quote too short"
    spans = find_spans(quote, text)
    if not spans:
        return None, "not locatable in page"
    if len(spans) == 1:
        return spans[0], None
    if near is not None:
        inside = [span for span in spans if span[0] < near[1] and span[1] > near[0]]
        if inside:
            return inside[0], None
    if len(needle) >= UNAMBIGUOUS_CHARS:
        return spans[0], None
    return None, "ambiguous in page"


def extract_anchors(
    limit: int | None = None,
    llm: StructuredLlm | None = None,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> AnchorSummary:
    """Place every approved answer's supporting text in its page, once."""
    initialize_page_artifacts_db()
    pending = _pending(limit)
    summary = AnchorSummary(questions=len(pending))
    if not pending:
        return summary

    client = _client(llm)
    for start in range(0, len(pending), max(batch_size, 1)):
        batch = pending[start : start + max(batch_size, 1)]
        rows, counts = _anchor_batch(batch, client, workers)
        _store(rows)
        summary.anchored += len(rows)
        summary.rejected += counts["rejected"]
        summary.failed += counts["failed"]
        for reason, count in counts["reasons"].items():
            summary.reasons[reason] = summary.reasons.get(reason, 0) + count
        print(
            f"anchors: {min(start + len(batch), len(pending))}/{len(pending)} "
            f"questions, {summary.anchored} placed",
            flush=True,
        )
    return summary


def _anchor_batch(batch, client, workers) -> tuple[list[tuple], dict]:
    counts = {"rejected": 0, "failed": 0, "reasons": {}}

    def note(reason: str) -> None:
        counts["reasons"][reason] = counts["reasons"].get(reason, 0) + 1

    judge = _Tolerant(client)
    prompts = [_prompt(row) for row in batch]
    quotes = run_structured_outputs(judge, prompts, AnswerQuote, workers=workers)
    counts["failed"] = len(judge.failures)

    rows: list[tuple] = []
    for row, answer in zip(batch, quotes):
        quote = (answer.quote or "").strip()
        if not answer.found or not quote:
            counts["rejected"] += 1
            note("no quote returned")
            continue
        needle, _ = _searchable(normalize(quote))
        if len(needle) < MIN_QUOTE_CHARS:
            counts["rejected"] += 1
            note("quote too short")
            continue
        # Verifiable: the quote must really be in the gold chunk. This is the
        # step a chunk-choice cannot offer. Only presence matters here — the
        # span that gets stored is the one in the page.
        if not find_spans(quote, row["gold_text"]):
            counts["rejected"] += 1
            note("not in gold chunk")
            continue
        span, reason = _place(quote, normalize(row["markdown"]), _gold_window(row))
        if span is None:
            counts["rejected"] += 1
            note(reason or "not locatable in page")
            continue
        rows.append((row["id"], row["page_id"], quote, span[0], span[1]))
    return rows, counts


def label_variant(variant: str) -> tuple[int, int]:
    """Link questions to `variant` chunks overlapping their anchor span.

    Pure interval arithmetic. A quote split by a chunk boundary overlaps both
    chunks and labels both, which is the correct answer for that question rather
    than a case needing a tie-break.
    """
    if variant == BASE_CHUNK_VARIANT:
        raise ValueError(
            "The base variant already owns the generated labels; overwriting "
            "them would replace the ground truth the sweep is measured against."
        )
    initialize_page_artifacts_db()
    with connect() as conn:
        rows = conn.execute(
            """
            select a.question_id, c.id as chunk_id
            from eval_answer_anchors a
            join page_chunks c
              on c.page_id = a.page_id
             and c.variant = ?
             and c.start_char is not null
             and c.end_char > a.start_char
             and c.start_char < a.end_char
            """,
            (variant,),
        ).fetchall()
        # This function is the only authority for a variant's labels. Adding to
        # whatever was there before would leave links from an earlier labelling
        # method in place and quietly mix two label sets in one column.
        conn.execute(
            """
            delete from eval_relevant_chunks
            where chunk_id in (select id from page_chunks where variant = ?)
            """,
            (variant,),
        )
        conn.executemany(
            "insert or ignore into eval_relevant_chunks (question_id, chunk_id) "
            "values (?, ?)",
            [(row["question_id"], row["chunk_id"]) for row in rows],
        )
    return len({row["question_id"] for row in rows}), len(rows)


@dataclass
class _Tolerant:
    """One unanswerable question must not abort a 3,476-question pass."""

    inner: StructuredLlm
    failures: list[str] = field(default_factory=list)

    def structured_output(self, prompt, output_schema, *, retries: int = 3):
        try:
            return self.inner.structured_output(prompt, output_schema, retries=retries)
        except Exception as error:
            self.failures.append(f"{type(error).__name__}: {error}")
            # Never None: run_structured_outputs drops None and would shift every
            # later answer onto the wrong question.
            return output_schema(found=False, quote="")


def _client(llm: StructuredLlm | None) -> StructuredLlm:
    if llm is not None:
        return llm
    model = CONFIG["model_aliases"].get(DEFAULT_ANCHOR_MODEL, DEFAULT_ANCHOR_MODEL)
    return LocalOllamaStructuredLlm(
        model_id=model,
        reasoning=CONFIG.get("anchor_reasoning", "low"),
        num_ctx=CONFIG.get("anchor_num_ctx", 8192),
        num_predict=CONFIG.get("anchor_num_predict", 512),
        method=DEFAULT_STRUCTURED_METHOD,
    )


def _gold_window(row: dict) -> tuple[int, int] | None:
    """The gold chunk's own span, used to choose between repeated matches.

    A short quote can occur more than once in a page — an infobox and the prose
    beneath it — and the occurrence that matters is the one in the chunk the
    question was written from. Where that span is missing, length decides instead.
    """
    if row.get("gold_start") is None or row.get("gold_end") is None:
        return None
    return int(row["gold_start"]), int(row["gold_end"])


def _pending(limit: int | None) -> list[dict]:
    sql = """
        select q.id, q.question, q.answer, q.question_language,
               gold.page_id as page_id, gold.text as gold_text,
               gold.start_char as gold_start, gold.end_char as gold_end,
               m.markdown
        from eval_questions q
        join eval_relevant_chunks r on r.question_id = q.id
        join page_chunks gold on gold.id = r.chunk_id
        join page_markdown_content m on m.page_id = gold.page_id
        where q.approved = 1
          and gold.variant = ?
          and not exists (
            select 1 from eval_answer_anchors a where a.question_id = q.id
          )
        group by q.id
        order by q.id
        """
    params: list = [BASE_CHUNK_VARIANT]
    if limit is not None:
        sql += "\n        limit ?"
        params.append(limit)
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def _prompt(row: dict) -> list[tuple[str, str]]:
    return [
        (
            "system",
            "Copy the sentences from the chunk that state the given answer. "
            "Reproduce them exactly as they appear, character for character. "
            "Do not translate, summarize, correct or add anything. The question "
            "may be in a different language from the chunk; the quote must be in "
            "the chunk's language.",
        ),
        (
            "human",
            f"Question ({row['question_language']}): {row['question']}\n"
            f"Answer: {row['answer']}\n\n"
            # Rendered, not raw: asking for a verbatim copy of text containing
            # inline URLs asks for something no model returns.
            f"Chunk:\n{rendered(row['gold_text'])}",
        ),
    ]


def _store(rows: list[tuple]) -> None:
    if not rows:
        return
    with connect() as conn:
        conn.executemany(
            """
            insert or replace into eval_answer_anchors
              (question_id, page_id, quote, start_char, end_char)
            values (?, ?, ?, ?, ?)
            """,
            rows,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    print(
        extract_anchors(
            limit=args.limit, workers=args.workers, batch_size=args.batch_size
        )
    )


if __name__ == "__main__":
    main()
