from __future__ import annotations

from typing import Any, Literal

from pydantic import Json

from src.db.schemas.base import Qrels, Row, Split


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
