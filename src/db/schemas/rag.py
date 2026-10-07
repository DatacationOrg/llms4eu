from __future__ import annotations

from typing import Any, Literal

from pydantic import Json

from src.db.schemas.base import Category, Country, Lang, Qrels, Row, Split


class Rag(Row, Qrels):
    """One question about one page: easy (names the place) or hard (describes it).
    Test on `answer_ok` rows."""

    file = "qa/wiki_qa_rag.parquet"
    id: str
    """The gold page (`Page.id`); a question's own id is `<id>:<kind>:<n>`."""
    n: int
    """Item number among the page's questions of this kind, 0 to 9."""
    kind: Literal["corpus", "challenge"]
    """`corpus`: easy, names the place; `challenge`: hard, describes it without naming."""
    lang: Lang
    """Language of the question and answer: the page's language."""
    question: str
    """The question."""
    answer: str
    """Short answer, from the page."""
    qtype: Literal[
        "identify", "location", "number", "date", "name", "description", "reason"
    ]
    """What the answer is."""
    evidence: list[str]
    """Quotes from the page that support the answer."""
    spans: list[list[int]]
    """`[start, end)` character offsets of each quote in `Page.text`."""
    verbatim: bool
    """Whether every quote was found in the page as is."""
    answer_ok: bool | None
    """Whether the answer passed every criterion of every judge; null: not judged."""
    criteria: Json[dict[str, bool]] | None
    """Per answer criterion, true only if every judge agrees: context_relevant,
    answerable, answer_relevant, answer_supported, answer_complete, language_match,
    answer_fluent, x_ok; null: not judged."""
    judges: int
    """How many judges scored the answer, 0 to 2."""
    labels: Json[dict[str, Any]]
    """Question quality labels: duplicate, language_ok, fluent, realistic,
    self_answering, unique_place, facts_supported, facts_answer, translation_ok,
    query_ok, answer_difficulty / retrieval_difficulty (easy, medium, hard), verdict
    (keep, fix, drop)."""
    split: Split
    """Dev or test, by page."""
    balanced: bool
    """Whether the page is in the language-balanced subset (at most 400 per language)."""
    x_lang: Lang
    """Language of the cross-lingual version: another EU language."""
    question_x: str
    """The question asked in `x_lang`; the gold page stays the same."""
    answer_x: str
    """The answer in `x_lang`."""
    x_ok: bool | None
    """Whether both judges passed the cross-lingual version; null: not judged."""
    # The gold page's metadata, as documented in `Page`:
    wikidata_id: str
    url: str
    title: str
    country: Country
    categories: list[Category]
    latitude: float
    longitude: float
    sitelinks: int
    char_count: int
    machine_generated: bool
    stub: bool
    """Whether the page has under 2,000 characters."""
    page_tags: Json[dict[str, Any]]
    """Page content tags (Gemma E4B classifier): article_type (descriptive,
    registry_stub, list_or_disambiguation), information_richness / tourist_appeal
    (low, medium, high), primary_focus (description, history, visiting, data,
    protection, other), visitor_info / history / nature / culture (bool)."""
    page_template_sim: float
    """Similarity of the page to its nearest other page, 0 to 1; high: templated stub."""
    title_in_question: bool
    """Whether the question names the page title."""
    lex_overlap: float
    """Share of the question's idf weight found in the page, 0 to 1: keyword ease."""
    reasoning: Literal["single_fact", "multi_fact", "inference"] | None
    """How the answer is reached; null: not tagged (mostly unbalanced pages)."""
    time_sensitive: bool | None
    """Whether the answer may change over time; null: not tagged."""
    evidence_pos: float | None
    """Where the first quote starts in the page, 0 to 1; null: no quote found."""
    evidence_in_table: bool
    """Whether a quote sits in a table row."""
    rank_o: list[float | None] | None
    """The gold page's [dense rank, dense margin, BM25 rank] for `question`; dense
    null: not computed; 101: not in the top 100."""
    rank_x: list[float | None] | None
    """The same for `question_x`."""
    variants: Json[dict[str, Any]] | None
    """Rewrites of the question: keyword, typo, verbose, followup (with history),
    `check` per rewrite, `followup_applicable`; null: none made (balanced pages only)."""
