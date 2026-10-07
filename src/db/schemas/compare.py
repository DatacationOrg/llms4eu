from __future__ import annotations

from typing import Any, Literal

from pydantic import Json

from src.db.schemas.base import Lang, Row, Split


class Compare(Row):
    """A question that needs two pages: same kind of place, language and country.
    Test on `ok` rows."""

    file = "qa/wiki_qa_compare.parquet"
    id: str
    """The first page, `pages[0]`."""
    pages: list[str]
    """The two pages (`Page.id`), A then B; both are needed to answer."""
    lang: Lang
    """Language of the question: both pages' language."""
    question: str
    """The question."""
    question_en: str
    """The question in English, for reading."""
    answer: str
    """Short answer."""
    qtype: Literal["number", "date", "attribute", "common"]
    """What is compared: a number, a date, an attribute, or what both share."""
    evidence_a: list[str]
    """Quotes from page A that support the answer."""
    evidence_b: list[str]
    """Quotes from page B that support the answer."""
    spans_a: list[list[int]]
    """`[start, end)` offsets of `evidence_a` in page A's `Page.text`."""
    spans_b: list[list[int]]
    """`[start, end)` offsets of `evidence_b` in page B's `Page.text`."""
    verbatim: bool
    """Whether every quote was found in its page as is."""
    check: Json[dict[str, Any]]
    """Checks by a second call: one_comparison, needs_both, answer_supported,
    identifiable, clean_text (bool each)."""
    ok: bool
    """Whether every check passed."""
    split: Split
    """Dev or test, by page."""
