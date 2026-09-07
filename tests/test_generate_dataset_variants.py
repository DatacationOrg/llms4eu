"""Per-variant question generation.

A question generated from a `base` chunk is answerable from a `base` chunk by
construction, which flatters `base` and any cutting close to it. Giving each
variant its own questions removes that, and these pin the two things that make it
work without a schema change: chunk ids carry the variant, so question ids do too,
and each variant's gold links point only at its own chunks.
"""

from __future__ import annotations

import sqlite3

from src.eval.generate_dataset import (
    QUESTION_TYPES,
    EvalQuestionBatch,
    QuestionCandidate,
    _chunk_questions,
    _eligible_unprocessed_chunks,
    _is_fact_dense,
    _missing_question_tasks,
    _question_id,
    _system_prompt,
    _valid_question,
    density_budget,
    insert_questions,
    slot_types,
)

PROSE = (
    "Grad Rajhenburg stoji nad reko Savo in je bil prvic omenjen leta 895. "
    "Grad so veckrat prezidali, danes pa v njem deluje muzej trapistov. "
    "Muzej je odprt od torka do nedelje, ob ponedeljkih pa je zaprt za obiskovalce. "
    "Leta 1881 so ga kupili menihi trapisti iz Francije in uredili samostan. "
    "Med drugo svetovno vojno je grad sluzil kot zbirno taborisce za izgnance, "
    "danes pa je v njem urejena stalna razstava o tem obdobju."
)
SHORT = "Grad Rajhenburg stoji nad Savo. Zgrajen je bil v 12. stoletju."


def test_generation_is_scoped_to_one_variant(tmp_path, monkeypatch):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    chunks, _ = _eligible_unprocessed_chunks(None, variant="tok256")

    assert [chunk["id"] for chunk in chunks] == ["page-1:tok256:0"]


def test_every_variant_is_offered_the_same_page(tmp_path, monkeypatch):
    """No variant is skipped for having been covered under another cutting."""
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    for variant in ("base", "tok256", "tok512"):
        chunks, _ = _eligible_unprocessed_chunks(None, variant=variant)
        assert len(chunks) == 1, variant


def test_question_ids_and_gold_links_stay_inside_their_variant(tmp_path, monkeypatch):
    """The whole reason this needs no schema change.

    A question id is derived from the chunk id, and chunk ids carry the variant, so
    two variants asking the identical question still get distinct rows and each
    keeps its own gold link.
    """
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    candidate = QuestionCandidate(
        question="Kdaj je bil grad prvic omenjen?",
        answer="Leta 895.",
        question_language="sl",
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        insert_questions(conn, "page-1:tok256:0", [("direct_short", candidate)])
        insert_questions(conn, "page-1:tok512:0", [("direct_short", candidate)])

    assert _question_id("page-1:tok256:0", candidate) != _question_id(
        "page-1:tok512:0", candidate
    )
    assert _labels(db_path, "tok256") == [("page-1:tok256:0",)]
    assert _labels(db_path, "tok512") == [("page-1:tok512:0",)]


def test_a_generated_chunk_is_not_offered_twice(tmp_path, monkeypatch):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    candidate = QuestionCandidate(
        question="Kdaj je bil grad prvic omenjen?",
        answer="Leta 895.",
        question_language="sl",
    )
    with sqlite3.connect(db_path) as conn:
        insert_questions(conn, "page-1:tok256:0", [("direct_short", candidate)])

    chunks, _ = _eligible_unprocessed_chunks(None, variant="tok256")

    assert chunks == []


def test_short_chunks_are_excluded_and_the_exclusion_is_reported(tmp_path, monkeypatch):
    """Silence here would look like a variant with fewer questions, not a filter.

    The 300-character floor drops 17 of 726 base chunks but 612 of 2,609 tok256
    chunks, so it is a size-dependent filter and its count has to be visible.
    """
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    chunks, excluded = _eligible_unprocessed_chunks(None, variant="tiny")

    assert chunks == [] and excluded == 1
    # Lowering the floor is what makes a small variant usable.
    kept, excluded = _eligible_unprocessed_chunks(None, variant="tiny", min_chars=40)
    assert len(kept) == 1 and excluded == 0


def test_the_length_floor_is_a_parameter_not_a_constant():
    assert not _is_fact_dense(SHORT)
    assert _is_fact_dense(SHORT, min_chars=40)


def test_missing_question_slots_are_scoped_to_one_variant(tmp_path, monkeypatch):
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    tasks = _missing_question_tasks(None, variant="tok512")

    assert {task["id"] for task in tasks} == {"page-1:tok512:0"}
    # One slot per question type, none of them filled yet.
    assert len(tasks) == 5


# --- robustness over a multi-hour run ---------------------------------------


class _Model:
    def __init__(self, result) -> None:
        self.result = result

    def invoke(self, _messages):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


CHUNK = {
    "id": "page-1:tok256:0",
    "text": PROSE,
    "language": "sl",
    "title": "Grad",
    "heading_path": None,
}


def test_a_missing_question_language_is_filled_in_not_fatal():
    """The model routinely omits it, and it is not something it has to tell us.

    Requiring the field made the parser raise, and one omission aborted a run
    that had already generated nothing else.
    """
    batch = EvalQuestionBatch(
        direct_short=QuestionCandidate(question="Kdaj?", answer="Leta 895."),
        crosslingual=QuestionCandidate(question="When?", answer="In 895."),
    )

    _, questions = _chunk_questions(CHUNK, _Model(batch))

    languages = {
        question_type: item.question_language for question_type, item in questions
    }
    assert languages["direct_short"] == "sl", "source-language types follow the chunk"
    assert languages["crosslingual"] not in ("", "sl"), (
        "crosslingual follows the target"
    )


def test_a_stated_question_language_is_left_alone():
    batch = EvalQuestionBatch(
        direct_short=QuestionCandidate(
            question="Kdaj?", answer="Leta 895.", question_language="sl"
        ),
    )

    _, questions = _chunk_questions(CHUNK, _Model(batch))

    assert questions[0][1].question_language == "sl"


def test_a_batch_wrapped_in_a_list_is_accepted():
    """11 of 28 observed failures were this, holding perfectly usable questions."""
    batch = EvalQuestionBatch.model_validate(
        [
            {
                "direct_short": {
                    "question": "Kdaj?",
                    "answer": "895",
                    "question_language": "sl",
                }
            }
        ]
    )

    assert batch.direct_short is not None
    assert batch.direct_short.answer == "895"


def test_a_list_of_type_tagged_items_is_accepted():
    batch = EvalQuestionBatch.model_validate(
        [
            {"type": "direct_short", "question": "Kdaj?", "answer": "895"},
            {"type": "crosslingual", "question": "When?", "answer": "895"},
        ]
    )

    assert batch.direct_short is not None and batch.crosslingual is not None
    assert batch.vague_short is None


def test_a_well_formed_object_is_untouched():
    batch = EvalQuestionBatch.model_validate(
        {"direct_short": {"question": "Kdaj?", "answer": "895"}}
    )

    assert batch.direct_short is not None


def test_one_bad_chunk_does_not_end_the_run():
    """Whatever the model layer raises, three hours of work must survive it."""
    from langchain_core.exceptions import OutputParserException

    _, questions = _chunk_questions(
        CHUNK, _Model(OutputParserException("Failed to parse EvalQuestionBatch"))
    )

    assert questions == []


def test_a_question_with_no_resolvable_language_is_dropped():
    assert not _valid_question(
        "direct_short", QuestionCandidate(question="Kdaj?", answer="895")
    )


# --- density-normalised generation ------------------------------------------
#
# The fairness claim: over the same corpus, every variant ends up with the same
# number of questions, so no cutting is probed more densely than another. The
# legacy design asks one question per type per chunk, which probes a 256-token
# cut four times more densely than a 1,024-token one over identical pages — and
# reaches further down each chunk for facts to ask about, so the large cut gets
# the more salient questions as well as the smaller sample.


def _tokens(chars_per_token: int = 4):
    """A stand-in tokenizer, so these tests need no local model weights."""
    return lambda text: len(text) // chars_per_token


def test_the_budget_scales_with_the_chunk():
    count = _tokens()

    assert density_budget("x" * (256 * 4), 256, count) == 1
    assert density_budget("x" * (512 * 4), 256, count) == 2
    assert density_budget("x" * (1024 * 4), 256, count) == 4


def test_every_variant_gets_the_same_questions_for_the_same_corpus():
    """The claim itself, stated as arithmetic over one page of 1,024 tokens."""
    count = _tokens()
    page = "x" * (1024 * 4)
    quarters = [page[index * 1024 : (index + 1) * 1024] for index in range(4)]
    halves = [page[:2048], page[2048:]]

    as_tok256 = sum(density_budget(chunk, 256, count) for chunk in quarters)
    as_tok512 = sum(density_budget(chunk, 256, count) for chunk in halves)
    as_tok1024 = density_budget(page, 256, count)

    assert as_tok256 == as_tok512 == as_tok1024 == 4


def test_a_short_tail_chunk_is_still_asked_one_question():
    """Budgets come from the chunk's real length, not its variant's nominal size.

    The last chunk of a page is routinely a fraction of its target. Rounding it
    to zero would drop it from the eval entirely and quietly shrink whichever
    variant produced the most page tails.
    """
    assert density_budget("x" * 40, 256, _tokens()) == 1


def test_a_one_slot_chunk_does_not_make_the_whole_variant_one_type():
    """Rotation, not the head of the list: tok256 asks for exactly one type.

    Taking it from the front every time would make every tok256 question
    `direct_short` and delete the category breakdown for the smallest cut.
    """
    types = [slot_types(f"page-{index}:tok256:0", 1)[0] for index in range(200)]

    assert set(types) == set(QUESTION_TYPES)
    # Even enough that no type carries the variant on its own.
    assert max(types.count(name) for name in QUESTION_TYPES) < len(types) / 2


def test_a_four_slot_chunk_asks_for_four_different_types():
    assert len(set(slot_types("page-1:tok1024:0", 4))) == 4


def test_slots_beyond_the_type_vocabulary_cycle_rather_than_run_out():
    """Unreachable through `density_budget`, which caps at the type count.

    Kept because the cap and the rotation are separate decisions: if the schema
    ever grows past one field per type, this is the behaviour the budget would
    start relying on, and a helper that silently returned fewer slots than asked
    would break the count without failing.
    """
    assert len(slot_types("page-1:tok2048:0", 8)) == 8


def _labels(db_path, variant: str) -> list[tuple[str]]:
    with sqlite3.connect(db_path) as conn:
        return [
            tuple(row)
            for row in conn.execute(
                """
                select r.chunk_id
                from eval_relevant_chunks r
                join page_chunks c on c.id = r.chunk_id
                where c.variant = ?
                order by r.chunk_id
                """,
                (variant,),
            )
        ]


def _write_fixture(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key, title text, source text not null,
              page_kind text not null, language text
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
            """
        )
        conn.executemany(
            """
            insert into page_chunks (
              id, page_id, chunk_index, variant, heading_path, text, char_count,
              start_char, end_char
            ) values (?, 'page-1', 0, ?, null, ?, length(?), 0, length(?))
            """,
            [
                ("page-1:0", "base", PROSE, PROSE, PROSE),
                ("page-1:tok256:0", "tok256", PROSE, PROSE, PROSE),
                ("page-1:tok512:0", "tok512", PROSE, PROSE, PROSE),
                ("page-1:tiny:0", "tiny", SHORT, SHORT, SHORT),
            ],
        )


def test_only_the_budgeted_types_are_requested_and_kept():
    """The count is the claim, so a generous model must not inflate it.

    Density mode changes which of `EvalQuestionBatch`'s fields are asked for
    rather than changing the schema, because the multi-field object through
    `json_schema` is the one shape measured reliable for this model. A model that
    fills the other fields anyway is ignored rather than allowed to break the
    count the whole design rests on.
    """
    chunk = {**CHUNK, "id": "page-1:tok512:0"}
    requested = tuple(slot_types(chunk["id"], 2))
    batch = EvalQuestionBatch(
        **{
            name: QuestionCandidate(
                question=f"Vprasanje {name}?",
                answer="Leta 895.",
                question_language="sl",
            )
            for name in QUESTION_TYPES
        }
    )

    _, questions = _chunk_questions(chunk, _Model(batch), requested)

    assert tuple(question_type for question_type, _ in questions) == requested


def test_the_prompt_names_only_the_budgeted_types():
    """Listing the other four is how a model ends up filling them."""
    prompt = _system_prompt(CHUNK, "en", ("direct_short", "vague_long"))

    assert "direct_short" in prompt and "vague_long" in prompt
    assert "crosslingual" not in prompt
    assert "the 2 question types listed below" in prompt


def test_the_prompt_asks_for_an_object_rather_than_a_list():
    """Measured: naming a subset of the types made a list the usual reply.

    The parser normalises the list shapes anyway, but every one it has to repair
    is a reply whose types could have been mislabelled on the way through.
    """
    prompt = _system_prompt(CHUNK, "en", ("direct_short", "vague_long"))

    assert "Do not return a list" in prompt


def test_a_multi_question_prompt_asks_for_facts_spread_over_the_chunk():
    """Four questions clustered in the opening probe one quarter of the chunk.

    That is the coverage difference density mode exists to remove, reintroduced
    inside the chunk.
    """
    prompt = _system_prompt(CHUNK, "en", QUESTION_TYPES[:4])

    assert "spread over the whole chunk" in prompt
    assert "4 consecutive stretches" in prompt

    single = _system_prompt(CHUNK, "en", ("direct_short",))
    assert "spread over the whole chunk" not in single


def test_a_repeated_question_is_dropped_rather_than_silently_collapsed():
    """A chunk that delivered two of its four questions is a density failure.

    `insert or ignore` would hide it: two identical questions from one chunk share
    a question id, so the row count would come out short with nothing said.
    """
    batch = EvalQuestionBatch(
        direct_short=QuestionCandidate(
            question="Kdaj je bil grad prvic omenjen?",
            answer="Leta 895.",
            question_language="sl",
        ),
        vague_short=QuestionCandidate(
            question="  KDAJ je bil grad   prvic omenjen? ",
            answer="Leta 895.",
            question_language="sl",
        ),
    )

    _, questions = _chunk_questions(
        CHUNK, _Model(batch), ("direct_short", "vague_short")
    )

    assert len(questions) == 1


def test_the_budget_is_capped_at_the_number_of_question_types():
    """One field per type, so a type cannot be filled twice.

    A 2,048-token cut would want eight questions and can only be asked five, which
    makes it less densely probed than the rest of the grid — reported by the run
    rather than applied quietly.
    """
    assert density_budget("x" * (2048 * 4), 256, _tokens()) == len(QUESTION_TYPES)


# --- output shapes measured on real chunks ----------------------------------
#
# Every payload below is copied from a failing reply in a live tok1024 run. The
# model's questions were fine in all of them; the batch was thrown away over the
# shape it arrived in, and asking for a subset of the types made these the usual
# reply rather than the occasional one.


def test_a_list_of_one_field_items_with_a_sibling_answer_is_merged():
    """The commonest shape: type name holding the question, `answer` beside it."""
    batch = EvalQuestionBatch.model_validate(
        [
            {
                "vague_long": "Kje je Arko maturiral?",
                "answer": "Maturo je leta 1965 opravil v Celju.",
                "crosslingual": None,
                "direct_short": None,
            },
            {
                "vague_long": None,
                "crosslingual": "Kde Andrej Arko studoval jazyky?",
                "answer": "Studoval je anglescino in rioscino.",
            },
            {"direct_short": "Kje je Arko delal novinar?", "answer": "Pri Delu."},
        ]
    )

    assert batch.vague_long.question == "Kje je Arko maturiral?"
    assert batch.vague_long.answer == "Maturo je leta 1965 opravil v Celju."
    assert batch.crosslingual.question == "Kde Andrej Arko studoval jazyky?"
    assert batch.direct_short.answer == "Pri Delu."
    assert batch.direct_long is None and batch.vague_short is None


def test_a_list_of_items_holding_proper_pair_objects_is_merged():
    batch = EvalQuestionBatch.model_validate(
        [
            {
                "vague_short": {
                    "question": "Kdo so bili starsi?",
                    "answer": "Karl in Marija Vodnik.",
                },
                "vague_long": None,
            },
            {
                "crosslingual": {
                    "question": "Waar werkte Gustav Strnisa?",
                    "answer": "Pri notarju Slambergerju.",
                }
            },
        ]
    )

    assert batch.vague_short.answer == "Karl in Marija Vodnik."
    assert batch.crosslingual.question == "Waar werkte Gustav Strnisa?"


def test_items_tagged_with_question_type_are_merged():
    """`question_type` rather than the `type` key the old unwrap looked for."""
    batch = EvalQuestionBatch.model_validate(
        [
            {
                "question_type": "crosslingual",
                "question_language": "de",
                "question": "Wer ist der Mentor?",
                "answer": "Dubravko Lapaine.",
            },
            {
                "question_type": "direct_short",
                "question": "Kdo je mentorica delavnice?",
                "answer": "Tina Brinovar.",
            },
        ]
    )

    assert batch.crosslingual.question_language == "de"
    assert batch.direct_short.answer == "Tina Brinovar."


def test_a_type_declined_with_nulls_does_not_take_the_others_down_with_it():
    """The difference between a chunk yielding two questions and none."""
    batch = EvalQuestionBatch.model_validate(
        {
            "direct_short": {
                "question": "Kdaj in kje se je rodil Milan Jereb?",
                "answer": "26. avgusta 1900 v Ospu.",
            },
            "direct_long": {"question": None, "answer": None},
            "vague_short": {
                "question": "Kje je Milan Jereb maturiral?",
                "answer": "9. julija 1919 v Ljubljani.",
            },
            "vague_long": {"question": None, "answer": None},
        }
    )

    assert batch.direct_short is not None and batch.vague_short is not None
    assert batch.direct_long is None and batch.vague_long is None


def test_invented_keys_beyond_the_vocabulary_are_dropped():
    """Asked for more questions than there are types, it makes new type names."""
    batch = EvalQuestionBatch.model_validate(
        {
            "direct_short": {"question": "Kje?", "answer": "V Ospu."},
            "direct_short_2": {"question": "Kje v steklarni?", "answer": "V Paracinu."},
            "vague_short_2": {"question": "Katero revijo?", "answer": "Galeb."},
        }
    )

    assert batch.direct_short.answer == "V Ospu."
    assert all(
        getattr(batch, name) is None for name in QUESTION_TYPES if name != "direct_short"
    )


def test_prose_is_still_a_retry_rather_than_something_to_parse():
    """Normalising must not turn an unusable reply into an empty batch.

    An empty batch would be recorded as a chunk that supported no questions,
    which is a silent hole in the density count rather than a failure to retry.
    """
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvalQuestionBatch.model_validate("I could not find any facts in the chunk.")


def test_filling_gaps_stays_inside_the_density_budget(tmp_path, monkeypatch):
    """The obvious remedy for a short density run must not undo the design.

    `--fill-missing` offers every type for every chunk by default, so topping up
    a variant that was generated at one question per 256 tokens would quietly
    restore the per-type design — and the counts would look right afterwards.
    """
    db_path = tmp_path / "pages.db"
    _write_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    every_type = _missing_question_tasks(None, variant="tok256", min_chars=40)
    budgeted = _missing_question_tasks(
        None, variant="tok256", min_chars=40, density_tokens=256
    )

    assert len(every_type) == len(QUESTION_TYPES)
    # The fixture chunk is one 256-token slot's worth, so exactly one is offered.
    assert len(budgeted) == 1
    assert budgeted[0]["question_type"] in slot_types("page-1:tok256:0", 1)
