from __future__ import annotations

from typing import Any, Literal

from pydantic import Json

from src.db.schemas.base import Qrels, Row, Split


class Unanswerable(Row, Qrels):
    """A question no page answers; `answering` is empty on ok rows."""

    file = "qa/wiki_qa_unanswerable.parquet"
    id: str  # the page it was written from
    lang: str
    title: str
    type: Literal["false_premise", "not_covered"]
    question: str
    question_en: str
    why: str  # what is false or missing
    check: Json[dict[str, Any]]
    ok: bool
    split: Split
