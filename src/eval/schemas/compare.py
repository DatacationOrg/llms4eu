from __future__ import annotations

from typing import Any

from pydantic import Json

from src.eval.schemas.base import Row, Split


class Compare(Row):
    """A question that needs two pages."""

    file = "compare"
    id: str  # = pages[0]
    pages: list[str]
    lang: str
    question: str
    question_en: str
    answer: str
    qtype: str  # number, date, attribute, common
    evidence_a: list[str]
    evidence_b: list[str]
    spans_a: list[list[int]]
    spans_b: list[list[int]]
    verbatim: bool
    check: Json[dict[str, Any]]
    ok: bool
    split: Split
