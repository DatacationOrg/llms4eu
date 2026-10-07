from __future__ import annotations

from typing import Any, Literal

from pydantic import Json

from src.db.schemas.base import Lang, Qrels, Row, Split


class Unanswerable(Row, Qrels):
    """A question no page answers; a system should say so. Test on `ok` rows."""

    file = "qa/wiki_qa_unanswerable.parquet"
    id: str
    """The page it was written from (`Page.id`); that page does not answer it either."""
    lang: Lang
    """Language of the question: the page's language."""
    title: str
    """The page title."""
    type: Literal["false_premise", "not_covered"]
    """`false_premise`: assumes a fact the page contradicts; `not_covered`: asks
    something the page does not say."""
    question: str
    """The question."""
    question_en: str
    """The question in English, for reading."""
    why: str
    """Why it has no answer: what is false or missing."""
    check: Json[dict[str, Any]]
    """Checks by a second call: unanswerable, type_ok, natural (bool each)."""
    ok: bool
    """Whether every check passed and no other page answers it (`answering` empty)."""
    split: Split
    """Dev or test, by page."""
