"""Answer anchoring, and the deterministic labelling it enables.

Comparability across variants rests on the question set staying fixed while only
the gold links move, and on those links being derived the same way for every
variant. These pin the three things that would break that: a quote the model did
not actually copy, a span that does not survive re-chunking, and a labelling step
that is anything other than arithmetic.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.eval.anchors import (
    AnswerQuote,
    collapse_with_offsets,
    extract_anchors,
    label_variant,
    locate,
)
from src.eval.evaluate import EvalRun, add_span_scores, load_span_labels
from src.eval.relabel import relabel_variant

PAGE = (
    "# Grad Rajhenburg\n\n"
    "Grad Rajhenburg stoji nad reko Savo in je bil prvic omenjen leta 895.\n"
    "Grad so veckrat prezidali, danes pa v njem deluje muzej.\n\n"
    "## Odprtost\n\n"
    "Muzej je odprt od torka do nedelje, ob ponedeljkih je zaprt.\n"
)
GOLD = (
    "Grad Rajhenburg stoji nad reko Savo in je bil prvic omenjen leta 895.\n"
    "Grad so veckrat prezidali, danes pa v njem deluje muzej."
)
QUOTE = "Grad Rajhenburg stoji nad reko Savo in je bil prvic omenjen leta 895."


class StubModel:
    """Returns a fixed quote, and records what it was asked."""

    def __init__(self, quote: str) -> None:
        self.quote = quote
        self.prompts: list = []

    def structured_output(self, prompt, output_schema, *, retries: int = 3):
        self.prompts.append(prompt)
        return AnswerQuote(found=bool(self.quote), quote=self.quote)


class BrokenModel:
    def structured_output(self, prompt, output_schema, *, retries: int = 3):
        raise RuntimeError("model returned no structured output")


# --- span mechanics ---------------------------------------------------------


def test_collapse_maps_every_kept_character_back_to_its_original_index():
    collapsed, offsets = collapse_with_offsets("  a  b\n\nc  ")

    assert collapsed == "a b c"
    assert len(offsets) == len(collapsed)
    original = "  a  b\n\nc  "
    assert original[offsets[0]] == "a"
    assert original[offsets[-1]] == "c"


def test_locate_matches_across_different_whitespace():
    text = "Uvod.\n\nGrad je bil\n  omenjen leta 895 v listini.\n\nKonec."
    quote = "Grad je bil omenjen leta 895 v listini."

    span = locate(quote, text)

    assert span is not None
    start, end = span
    # The returned span is a real slice of the original, whitespace and all.
    assert text[start:end].split() == quote.split()


def test_locate_refuses_a_quote_too_short_to_place_unambiguously():
    # "leta 895" appears twice; anchoring an arbitrary one would be worse than
    # admitting the quote is unusable.
    text = "Prvic omenjen leta 895. Spet leta 895 v drugi listini."

    assert locate("leta 895", text) is None


def test_locate_returns_none_when_the_quote_is_absent():
    assert locate("Something never written in this page at all.", PAGE) is None


def test_locate_matches_a_quote_of_the_rendered_text_against_markdown_source():
    # The chunk holds an inline link; no model reproduces the URL, and the quote
    # of what a reader sees is the correct quote.
    text = "Delovnih migrantov je bilo v [zasavski regiji](https://sl.wikipedia.org/x) 54 %."
    quote = "Delovnih migrantov je bilo v zasavski regiji 54 %."

    span = locate(quote, text)

    assert span is not None
    # The span is in source coordinates, so it covers the markup it spans over.
    assert text[span[0] : span[1]] == text.strip(".") + "."


def test_locate_ignores_emphasis_markers():
    assert locate(
        "Grad je bil omenjen leta 895.", "**Grad** je bil *omenjen* leta 895."
    )


def test_a_short_quote_is_placed_when_it_occurs_only_once():
    # "Marko Kenig" is 11 characters and a complete answer. Refusing it on length
    # alone threw away questions whose answer is a name.
    text = "Grad je leta 1881 kupil Marko Kenig, trgovec iz Ljubljane."

    span = locate("Marko Kenig", text)

    assert span is not None and text[span[0] : span[1]] == "Marko Kenig"


def test_a_repeated_short_quote_is_placed_inside_the_chunk_it_came_from():
    # Ambiguous on its own, decided by the span of the chunk the question was
    # written from rather than refused.
    text = "Marko Kenig v uvodu. Vmesni odstavek. Lastnik je bil Marko Kenig."
    window = (text.index("Vmesni"), len(text))

    span = locate("Marko Kenig", text, near=window)

    assert span is not None and span[0] > window[0]


# --- extraction ------------------------------------------------------------


def test_a_verifiable_quote_is_anchored_in_page_coordinates(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    model = StubModel(QUOTE)

    summary = extract_anchors(llm=model, workers=1)

    assert summary.anchored == 1 and summary.rejected == 0
    anchor = _anchors(db_path)[0]
    assert anchor["quote"] == QUOTE
    # The span points into the page, not into a chunk, which is what makes it
    # survive any re-chunking.
    assert PAGE[anchor["start_char"] : anchor["end_char"]].split() == QUOTE.split()
    # The model is shown the gold chunk and the answer, not the whole page.
    human = model.prompts[0][1][1]
    assert "Answer: 895" in human
    assert "Muzej je odprt" not in human


def test_a_quote_the_model_did_not_copy_is_rejected(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    # Paraphrased rather than copied: checkable, and therefore refused.
    summary = extract_anchors(
        llm=StubModel("The castle was first mentioned in the year 895."),
        workers=1,
    )

    assert summary.anchored == 0
    assert summary.rejected == 1
    assert summary.reasons == {"not in gold chunk": 1}
    assert _anchors(db_path) == []


def test_a_quote_too_short_to_place_is_reported_as_such(monkeypatch, tmp_path):
    """Length and absence are different failures and were being conflated.

    Both made `locate` return None, so valid short answers were counted as
    quotes the model had invented — which is what hid the real rejection rate.
    """
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    summary = extract_anchors(llm=StubModel("895"), workers=1)

    assert summary.reasons == {"quote too short": 1}


def test_a_model_failure_is_counted_not_fatal(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    summary = extract_anchors(llm=BrokenModel(), workers=1)

    assert summary.failed == 1
    assert summary.anchored == 0


def test_extraction_is_skipped_for_already_anchored_questions(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    first = extract_anchors(llm=StubModel(QUOTE), workers=1)
    second_model = StubModel(QUOTE)
    second = extract_anchors(llm=second_model, workers=1)

    assert first.anchored == 1
    assert second.questions == 0, "one model pass serves every later variant"
    assert second_model.prompts == []


# --- deterministic labelling ------------------------------------------------


def test_labelling_a_variant_is_interval_overlap(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    extract_anchors(llm=StubModel(QUOTE), workers=1)

    questions, links = label_variant("narrow")

    assert questions == 1
    # Only the chunk covering the quote, not the later section of the same page.
    assert _labels(db_path, "narrow") == [("q-a", "page-1:narrow:0")]
    assert links == 1


def test_a_quote_split_by_a_chunk_boundary_labels_both_chunks(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    _add_split_variant(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    extract_anchors(llm=StubModel(QUOTE), workers=1)

    label_variant("split")

    # Both halves hold part of the answer, so both are relevant. Under a
    # pick-one judge this was a tie-break; here it is just the right answer.
    assert _labels(db_path, "split") == [
        ("q-a", "page-1:split:0"),
        ("q-a", "page-1:split:1"),
    ]


def test_labelling_is_idempotent(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    extract_anchors(llm=StubModel(QUOTE), workers=1)

    label_variant("narrow")
    label_variant("narrow")

    assert _labels(db_path, "narrow") == [("q-a", "page-1:narrow:0")]


def test_span_scores_are_read_from_the_anchors_and_the_variant_spans(
    monkeypatch, tmp_path
):
    """The anchors placed here are what the character-overlap metrics score on."""
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    extract_anchors(llm=StubModel(QUOTE), workers=1)

    anchors, chunk_spans = load_span_labels("narrow")
    assert anchors["q-a"][0] == "page-1"
    assert set(chunk_spans) == {"page-1:narrow:0", "page-1:narrow:1"}

    run = EvalRun(
        questions=[{"id": "q-a", "question_type": "direct_short"}],
        relevance=[],
        methods=["stub"],
        warmup_count=0,
        # The answer-bearing chunk first, then the unrelated later section.
        rankings={"stub": {"q-a": ["page-1:narrow:0", "page-1:narrow:1"]}},
        timings={},
        score_names=["hit@1"],
        scores={"stub": {"hit@1": 1.0}},
    )

    scored = add_span_scores(run, "narrow")

    assert scored.span_questions == 1
    assert scored.scores["stub"]["char_recall@1"] == 1.0
    # Adding the second chunk cannot help recall and must cost precision.
    assert (
        scored.scores["stub"]["char_precision@5"]
        < (scored.scores["stub"]["char_precision@1"])
    )
    assert "hit@1" in scored.score_names


def test_span_metrics_need_no_model_pass_by_default(monkeypatch, tmp_path):
    """The default target is the base chunk the question was generated from.

    That chunk is ground truth by construction, so every approved question has a
    target without anyone running a model. Anchoring only narrows the target from
    a paragraph to a sentence; it is not what makes the metric work.
    """
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    run = _stub_run()
    scored = add_span_scores(run, "narrow")

    # No anchor table exists at all in this fixture, let alone a placed anchor.
    assert load_span_labels("narrow", target="anchor") == ({}, {})
    assert scored.span_questions == 1
    assert scored.scores["stub"]["char_recall@1"] == 1.0


def test_the_anchor_target_is_skipped_when_nothing_is_anchored(monkeypatch, tmp_path):
    """A tighter target that was never placed reports nothing, not zeroes."""
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    run = _stub_run()

    assert add_span_scores(run, "narrow", target="anchor") is run


def test_an_unknown_span_target_is_refused(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    with pytest.raises(ValueError, match="Unknown span target"):
        load_span_labels("narrow", target="whatever")


def _stub_run() -> EvalRun:
    return EvalRun(
        questions=[{"id": "q-a", "question_type": "direct_short"}],
        relevance=[],
        methods=["stub"],
        warmup_count=0,
        rankings={"stub": {"q-a": ["page-1:narrow:0", "page-1:narrow:1"]}},
        timings={},
        score_names=["hit@1"],
        scores={"stub": {"hit@1": 1.0}},
    )


def test_labelling_the_base_variant_is_refused():
    with pytest.raises(ValueError, match="already owns the generated labels"):
        label_variant("base")


def test_relabel_variant_runs_both_stages(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    summary = relabel_variant("narrow", llm=StubModel(QUOTE), workers=1)

    assert summary.anchored == 1
    assert summary.questions == 1
    assert summary.links == 1
    assert "narrow" in str(summary)


def test_base_labels_are_never_disturbed(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    relabel_variant("narrow", llm=StubModel(QUOTE), workers=1)

    assert _labels(db_path, "base") == [("q-a", "page-1:0")]


# --- fixture ---------------------------------------------------------------


def _anchors(db_path) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("select * from eval_answer_anchors")]


def _labels(db_path, variant: str) -> list[tuple[str, str]]:
    with sqlite3.connect(db_path) as conn:
        return [
            tuple(row)
            for row in conn.execute(
                """
                select r.question_id, r.chunk_id
                from eval_relevant_chunks r
                join page_chunks c on c.id = r.chunk_id
                where c.variant = ?
                order by r.question_id, r.chunk_id
                """,
                (variant,),
            )
        ]


def _add_split_variant(path):
    """A variant whose boundary falls in the middle of the quoted sentence."""
    middle = PAGE.index("prvic omenjen")
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            insert into page_chunks (
              id, page_id, chunk_index, variant, heading_path, text, char_count,
              start_char, end_char
            ) values (?, 'page-1', ?, 'split', null, ?, length(?), ?, ?)
            """,
            [
                ("page-1:split:0", 0, PAGE[:middle], PAGE[:middle], 0, middle),
                (
                    "page-1:split:1",
                    1,
                    PAGE[middle:],
                    PAGE[middle:],
                    middle,
                    len(PAGE),
                ),
            ],
        )


def _write_fixture(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key, title text, source text not null,
              page_kind text not null, language text
            );
            create table page_markdown_content (
              page_id text primary key references page_metadata(id) on delete cascade,
              markdown text not null
            );
            create table page_sources (
              source text primary key, language text not null
            );
            create table page_chunks (
              id text primary key,
              page_id text not null references page_metadata(id) on delete cascade,
              chunk_index integer not null,
              variant text not null default 'base',
              heading_path text,
              text text not null,
              char_count integer not null,
              start_char integer,
              end_char integer,
              unique(page_id, variant, chunk_index)
            );
            create table eval_questions (
              id text primary key, question text not null, answer text not null,
              question_type text not null, question_language text not null,
              approved integer not null default 1
            );
            create table eval_relevant_chunks (
              question_id text not null references eval_questions(id) on delete cascade,
              chunk_id text not null references page_chunks(id) on delete cascade,
              primary key (question_id, chunk_id)
            );

            insert into page_metadata values
              ('page-1', 'Grad', 'fixture', 'prose', 'sl');
            insert into page_sources values ('fixture', 'sl');
            insert into eval_questions values
              ('q-a', 'Kdaj je bil grad prvic omenjen?', '895',
               'direct_short', 'sl', 1);
            """
        )
        conn.execute(
            "insert into page_markdown_content (page_id, markdown) values (?, ?)",
            ("page-1", PAGE),
        )
        split = PAGE.index("## Odprtost")
        conn.executemany(
            """
            insert into page_chunks (
              id, page_id, chunk_index, variant, heading_path, text, char_count,
              start_char, end_char
            ) values (?, 'page-1', ?, ?, null, ?, length(?), ?, ?)
            """,
            [
                # base: the gold chunk the question was generated from
                ("page-1:0", 0, "base", GOLD, GOLD, PAGE.index(GOLD[:20]), split),
                # narrow: the answer-bearing chunk plus a later, unrelated one
                (
                    "page-1:narrow:0",
                    0,
                    "narrow",
                    PAGE[:split],
                    PAGE[:split],
                    0,
                    split,
                ),
                (
                    "page-1:narrow:1",
                    1,
                    "narrow",
                    PAGE[split:],
                    PAGE[split:],
                    split,
                    len(PAGE),
                ),
            ],
        )
        conn.execute("insert into eval_relevant_chunks values ('q-a', 'page-1:0')")


def test_labelling_replaces_labels_from_an_earlier_method(monkeypatch, tmp_path):
    """A variant column must hold one label set, not an accumulation of them.

    The judge-based labeller wrote links that anchoring would not produce.
    Leaving them in place would mix two derivations in one column and make the
    variant look better labelled than it is.
    """
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    with sqlite3.connect(db_path) as conn:
        # A link the anchor does not support: the later, unrelated section.
        conn.execute(
            "insert into eval_relevant_chunks values ('q-a', 'page-1:narrow:1')"
        )

    extract_anchors(llm=StubModel(QUOTE), workers=1)
    label_variant("narrow")

    assert _labels(db_path, "narrow") == [("q-a", "page-1:narrow:0")]
