from __future__ import annotations

from typing import Any, Literal

from pydantic import Json

from src.db.schemas.base import Lang, Row, Split


class Tables(Row):
    """An aggregation (max, mean, argmin, ...) over a table in one page. Test on `ok`
    rows; the answer is `answer_final`."""

    file = "qa/wiki_qa_tables.parquet"
    id: str
    """The page (`Page.id`)."""
    n: int
    """Item number among the page's table questions, 0 to 2."""
    title: str
    """The page title."""
    in_language: Lang
    """Language of the question: the page's language."""
    table: list[str]
    """The table's header row."""
    n_rows: int
    """Data rows in the table."""
    axis: Literal["col", "row"]
    """Whether the values are a column or a row."""
    index: int
    """Which column or row, from 0."""
    exclude: list[int]
    """Row indices left out of the aggregation, e.g. a totals row."""
    op: Literal["max", "min", "argmax", "argmin", "mean", "sum", "count"]
    """The aggregation; argmax / argmin answer with the row's label."""
    unit: str
    """Unit of the values, e.g. `ha`; empty when none."""
    question: str
    """The question."""
    question_en: str
    """The question in English, for reading."""
    answer_final: Json[dict[str, Any] | str] | None
    """The answer to use: `value`, `label` (argmax / argmin), `values` (the inputs);
    null: no verified answer."""
    answer_source: Literal["computed", "sonnet", "unverified"]
    """Where `answer_final` comes from: code, Claude Sonnet's correction, or unchecked."""
    answer: Json[dict[str, Any]] | None
    """The answer computed by code, before any correction."""
    sonnet: Json[dict[str, Any] | str] | None
    """Claude Sonnet's check: answer_correct, question_ok, your_answer, note."""
    sonnet_checked: bool
    """Whether Claude Sonnet checked this row."""
    ok: bool
    """Whether the question is clear and its answer verified."""
    split: Split
    """Dev or test, by page."""
