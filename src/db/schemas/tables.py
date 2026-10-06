from __future__ import annotations

from typing import Any

from pydantic import Json

from src.db.schemas.base import Row, Split


class Tables(Row):
    """An aggregation (max, mean, argmin, ...) over a table in one page."""

    file = "tables"
    id: str
    n: int
    title: str
    in_language: str
    table: list[str]  # header row
    n_rows: int
    axis: str  # col / row
    index: int
    exclude: list[int]  # totals skipped
    op: str
    unit: str
    question: str
    question_en: str
    answer_final: Json[dict[str, Any] | str] | None  # use this: value, label
    answer_source: str  # computed / sonnet / unverified
    answer: Json[dict[str, Any]] | None  # computed by code
    sonnet: Json[dict[str, Any] | str] | None
    sonnet_checked: bool
    ok: bool
    split: Split
