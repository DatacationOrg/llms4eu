"""The wiki places QA test set: one pydantic model per Parquet file.

Each model lists its file's columns. Rows are questions about the pages in
`pages.jsonl` next to the folder; a page id looks like `svwiki/Q123`.
Test on `ok` rows (`answer_ok` in `Rag`), tune on `split == "dev"`.

    hard = load(Rag, ["id", "question"], answer_ok=True, kind="challenge").to_pandas()
    for q in read(Unanswerable, ok=True): print(q.question, q.why)
    texts = pages(hard.id)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, Json

ROOT = Path(os.getenv("WIKI_QA_DIR", "/data/llms4eu/wiki/qa"))  # or a copy of it
Split = Literal["dev", "test"]


class Row(BaseModel):
    file: ClassVar[str]


class Qrels(BaseModel):
    """Which other pages fit the question, judged over its BM25 top 20 (null = not judged)."""

    qrels_judged: bool
    gold_match: str | None = None  # the gold page's own judgement: yes / partly / no
    gold_answers: bool | None = None
    relevant: list[str] | None = None  # other pages it fits fully
    partial: list[str] | None = None  # other pages it fits partly
    answering: list[str] | None = None  # pages whose text answers it
    hard_negatives: list[str] | None = None  # similar pages that do not fit
    n_relevant: int | None = None  # gold + relevant
    hits: Literal["unique", "few", "many"] | None = None
    pool_saturated: bool | None = None  # likely more matches than judged


class Rag(Row, Qrels):
    """One question about one page: easy (names the place) or hard (describes it)."""

    file = "rag"
    id: str  # the gold page
    n: int  # item number on the page
    kind: Literal["corpus", "challenge"]  # easy / hard
    lang: str
    question: str
    answer: str
    qtype: str  # identify, location, number, date, name, description, reason
    evidence: list[str]  # quotes from the page
    spans: list[list[int]]  # their [start, end) offsets in the page text
    verbatim: bool  # every quote was found
    answer_ok: bool | None  # passed every answer check
    criteria: Json[dict[str, bool]] | None  # each answer check
    judges: int
    labels: Json[dict[str, Any]]  # question quality labels
    split: Split
    balanced: bool  # in the <= 400 pages per language subset
    # the question in another EU language
    x_lang: str
    question_x: str
    answer_x: str
    x_ok: bool | None
    # page
    wikidata_id: str
    url: str
    title: str
    country: str
    categories: list[str]
    latitude: float
    longitude: float
    sitelinks: int
    char_count: int
    machine_generated: bool
    stub: bool  # page < 2,000 chars
    page_tags: Json[dict[str, Any]]
    page_template_sim: float  # similarity to the nearest other page
    # tags
    title_in_question: bool
    lex_overlap: float  # share of the question's keywords found in the page, 0-1
    reasoning: str | None  # single_fact / multi_fact / inference
    time_sensitive: bool | None
    evidence_pos: float | None  # where the first quote sits in the page, 0-1
    evidence_in_table: bool
    rank_o: list[float | None] | None  # gold page [dense rank, dense margin, BM25 rank]
    rank_x: list[float | None] | None  # the same for question_x
    variants: Json[dict[str, Any]] | None  # keyword, typo, verbose, followup rewrites


class Unanswerable(Row, Qrels):
    """A question no page answers; `answering` is empty on ok rows."""

    file = "unanswerable"
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


class Meta(Row):
    """A list or geo question whose answer is a set of pages."""

    file = "meta"
    key: str
    kind: Literal["list", "geo"]
    lang: str
    spec: str  # the task in English
    question: str
    question_en: str
    gold_pages: list[str]
    gold: list[str]  # Wikidata ids
    n_gold: int
    anchor: str | None  # geo: the centre page
    radius_km: int | None
    category: str | None
    check: Json[dict[str, Any]]
    ok: bool
    split: Split


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


def path(model: type[Row]) -> Path:
    return ROOT / f"wiki_qa_{model.file}.parquet"


def load(model: type[Row], columns: list[str] | None = None, **where):
    """The file as a pyarrow Table (`.to_pandas()`), only `columns`, rows matching `where`."""
    filters = [(k, "==", v) for k, v in where.items()] or None
    return pq.read_table(path(model), columns=columns, filters=filters)


def read(model: type[Row], **where) -> Iterator[Row]:
    """The matching rows as validated models, JSON columns parsed."""
    for batch in load(model, **where).to_batches():
        yield from map(model.model_validate, batch.to_pylist())


def pages(ids: Iterable[str]) -> dict[str, dict]:
    """Page id -> page (title, text, metadata); one pass over pages.jsonl."""
    want = set(ids)
    with open(ROOT.parent / "pages.jsonl") as fh:
        return {p["id"]: p for p in map(json.loads, fh) if p["id"] in want}
