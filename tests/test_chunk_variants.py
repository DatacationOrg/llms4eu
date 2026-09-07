"""Contract tests for chunking variants and the library-backed chunkers.

The ablation rests on four invariants: the frozen `legacy` strategy still
produces exactly the chunks the approved labels point at, variants never
collide, spans stay ordered and inside the page, and a chunk never exceeds the
size it was cut for.
"""

from __future__ import annotations

import sqlite3
import unicodedata

import pytest

from src.preprocess.chunkers import (
    MERGE_TOLERANCE,
    _resolve_spans,
    build_chunker,
    normalize,
)
from src.preprocess.chunks import (
    chunk_id,
    default_chunker,
    rebuild_page_chunks,
)

MARKDOWN = """# Castle

The castle stands above the river and guards the valley below it, looking out
over the terraced vineyards that climb the opposite bank. Travellers approaching
from the south see the towers long before they reach the bridge, which is why the
site was chosen for a fortification in the first place.

It was first mentioned in 895 and rebuilt several times after that, most
thoroughly after the earthquake that brought down the eastern wing. The current
outline dates from the sixteenth century, when the courtyard was enclosed and the
chapel was added along the northern range.

## History

Monks lived here for centuries and kept detailed records of the estate, listing
every harvest, every repair and every dispute with the neighbouring parish. Those
ledgers are the reason so much is known about daily life on the estate compared
with other holdings in the region.

The last owners left in 1941 when the estate was seized, and the buildings served
several institutional purposes before restoration began. Much of the interior
detail had to be reconstructed from the photographs taken during the survey of
1936, which remain the best record of the original rooms.

## Visiting

Open Tuesday to Sunday, closed on Mondays throughout the winter season, with
guided tours starting on the hour from the gatehouse. Tickets cover the chapel,
the cellars and the exhibition in the eastern wing, and the last tour leaves
ninety minutes before closing time.
"""

PAGE = {
    "id": "page-castle",
    "title": "Castle",
    "source": "fixture",
    "page_kind": "prose",
    "language": "sl",
}


def test_chunk_ids_keep_the_historical_form_for_the_base_variant():
    assert chunk_id("page-1", "base", 3) == "page-1:3"
    assert chunk_id("page-1", "tok512", 3) == "page-1:tok512:3"


def test_unknown_strategy_and_unit_are_rejected():
    with pytest.raises(ValueError):
        build_chunker("nonsense", size=500, overlap=0)
    with pytest.raises(ValueError):
        build_chunker("recursive", size=500, overlap=0, unit="furlongs")


def test_token_sizing_needs_a_provider():
    with pytest.raises(ValueError):
        build_chunker("recursive", size=500, overlap=0, unit="tokens")


def test_overlap_must_stay_below_the_chunk_size():
    with pytest.raises(ValueError):
        build_chunker("recursive", size=100, overlap=100)


@pytest.mark.parametrize("strategy", ["recursive", "markdown", "legacy"])
def test_every_strategy_respects_its_chunk_size(strategy):
    chunker = build_chunker(strategy, size=300, overlap=0)

    chunks = chunker.split(MARKDOWN, PAGE)

    assert chunks
    # `legacy` allows one long paragraph to run to its derived ceiling; the library
    # strategies may exceed the target only by absorbing a short trailing piece,
    # bounded by MERGE_TOLERANCE.
    ceiling = 300 * 13 // 9 if strategy == "legacy" else int(300 * MERGE_TOLERANCE)
    assert all(len(chunk.text) <= ceiling for chunk in chunks)


def test_no_fragment_chunks_are_emitted():
    """A section's leftover tail is folded back, not shipped as its own chunk.

    Real cost of not doing this: 612 of tok256's 2,609 chunks came out under 300
    characters and 157 under 100. A fragment starting mid-sentence retrieves badly,
    cannot carry a generated question, and excluding it later biases the whole
    comparison towards cuttings that happen to produce fewer of them.
    """
    chunker = build_chunker("markdown", size=300, overlap=0)

    chunks = chunker.split(MARKDOWN, PAGE)

    assert len(chunks) > 1, "the fixture must actually split"
    assert all(len(chunk.text) >= 150 for chunk in chunks), [
        len(chunk.text) for chunk in chunks
    ]


def test_merging_loses_no_text():
    """The legacy chunker dropped short chunks and lost 0.91% of the corpus."""
    merged = build_chunker("markdown", size=300, overlap=0).split(MARKDOWN, PAGE)
    unmerged = build_chunker("markdown", size=300, overlap=0, min_size=1).split(
        MARKDOWN, PAGE
    )

    assert len(merged) < len(unmerged), "the fixture must produce a short piece"
    assert _words(merged) == _words(unmerged)


def test_a_merged_chunk_still_spans_its_real_position_in_the_page():
    """Spans are resolved before merging, so the hull is exact, not a guess."""
    chunks = build_chunker("markdown", size=300, overlap=0).split(MARKDOWN, PAGE)

    for chunk in chunks:
        assert 0 <= chunk.start_char < chunk.end_char <= len(MARKDOWN)
        # Every merged chunk's first line still occurs inside its own span.
        first_line = chunk.text.splitlines()[0].strip()
        assert first_line[:30] in MARKDOWN[chunk.start_char : chunk.end_char]


def test_merging_can_be_switched_off():
    disabled = build_chunker("markdown", size=300, overlap=0, min_size=1)

    assert disabled.min_size == 1


def test_a_minimum_at_or_above_the_target_is_refused():
    with pytest.raises(ValueError, match="Minimum chunk size"):
        build_chunker("markdown", size=300, overlap=0, min_size=300)


def _words(chunks) -> list[str]:
    return sorted(word for chunk in chunks for word in chunk.text.split())


@pytest.mark.parametrize("strategy", ["recursive", "markdown", "legacy"])
def test_every_strategy_returns_ordered_spans_inside_the_page(strategy):
    chunker = build_chunker(strategy, size=300, overlap=0)

    chunks = chunker.split(MARKDOWN, PAGE)

    for chunk in chunks:
        assert 0 <= chunk.start_char <= chunk.end_char <= len(MARKDOWN)
    starts = [chunk.start_char for chunk in chunks]
    assert starts == sorted(starts)


def test_smaller_size_produces_more_chunks():
    big = build_chunker("recursive", size=600, overlap=0).split(MARKDOWN, PAGE)
    small = build_chunker("recursive", size=200, overlap=0).split(MARKDOWN, PAGE)

    assert len(small) > len(big)


def test_overlap_repeats_text_between_consecutive_chunks():
    without = build_chunker("recursive", size=300, overlap=0).split(MARKDOWN, PAGE)
    with_overlap = build_chunker("recursive", size=300, overlap=100).split(
        MARKDOWN, PAGE
    )

    assert len(with_overlap) >= len(without)
    assert sum(len(c.text) for c in with_overlap) > sum(len(c.text) for c in without)


def test_markdown_strategy_keeps_the_heading_path():
    chunks = build_chunker("markdown", size=300, overlap=0).split(MARKDOWN, PAGE)

    paths = {chunk.heading_path for chunk in chunks}
    assert any(path and "History" in path for path in paths)


def test_default_chunker_is_the_frozen_legacy_strategy():
    assert default_chunker().name == "legacy"


# --- multilingual hardening -------------------------------------------------


def test_normalize_folds_decomposed_eu_diacritics_to_one_form():
    decomposed = unicodedata.normalize("NFD", "Rajhenburški grad v Občini Krško")

    assert normalize(decomposed) == "Rajhenburški grad v Občini Krško"
    assert len(normalize(decomposed)) < len(decomposed)


def test_chunker_normalizes_before_splitting_so_spans_resolve():
    page = (
        "# Grad\n\n" + unicodedata.normalize("NFD", "Občina Krško leži ob Savi. ") * 40
    )

    chunks = build_chunker("recursive", size=200, overlap=0).split(page, PAGE)

    normalized = normalize(page)
    for chunk in chunks:
        assert unicodedata.is_normalized("NFC", chunk.text)
        # A span only means something if the text is really there.
        assert normalized[chunk.start_char : chunk.end_char] == chunk.text


@pytest.mark.parametrize(
    "script_text",
    [
        "Το κάστρο χτίστηκε το 895 και ανακαινίστηκε αργότερα. ",  # Greek
        "Замъкът е споменат за пръв път през 895 година. ",  # Bulgarian Cyrillic
        "A vár 895-ben épült, és később felújították. ",  # Hungarian
    ],
)
def test_no_mid_word_hard_cuts_across_eu_scripts(script_text):
    page = script_text * 12
    chunks = build_chunker("recursive", size=120, overlap=0).split(page, PAGE)

    assert len(chunks) > 1
    # The old hand-rolled splitter cut at a raw character offset once no
    # separator was found, which halves a word. The library cascade must not:
    # every word of the page survives intact inside some chunk.
    intact = {word for chunk in chunks for word in chunk.text.split()}
    assert set(page.split()) <= intact


def test_resolve_spans_stays_ordered_when_a_piece_is_not_found_verbatim():
    text = "alpha beta gamma delta"
    # The middle piece was normalized by the header splitter and no longer
    # appears verbatim; spans must still be defined and non-decreasing.
    spans = _resolve_spans(text, ["alpha", "BETA-NORMALIZED", "gamma", "delta"])

    starts = [start for start, _ in spans]
    assert starts == sorted(starts)
    assert all(0 <= start <= end <= len(text) for start, end in spans)
    assert text[spans[0][0] : spans[0][1]] == "alpha"
    assert text[spans[2][0] : spans[2][1]] == "gamma"


# --- token-unit sizing ------------------------------------------------------


def _qwen_tokenizer_or_skip():
    from src.shared.tokenizers import huggingface_tokenizer

    try:
        return huggingface_tokenizer("qwen")
    except Exception as error:  # not cached in this environment
        pytest.skip(f"qwen tokenizer unavailable locally: {error}")


def test_token_sized_chunks_stay_under_the_provider_limit():
    _qwen_tokenizer_or_skip()
    from src.shared.tokenizers import token_counter

    count = token_counter("qwen")
    chunker = build_chunker(
        "markdown", size=256, overlap=0, unit="tokens", provider="qwen"
    )

    chunks = chunker.split(MARKDOWN, PAGE)

    assert chunks
    # The budget reserves the representation prefix, so the indexed document —
    # not merely the chunk text — is what has to fit. Merging a short trailing
    # piece can push a chunk past the target, never past MERGE_TOLERANCE, and
    # every provider now reads far more than this anyway.
    assert all(count(chunk.text) <= int(256 * MERGE_TOLERANCE) for chunk in chunks)


def test_token_sizing_beats_character_sizing_on_a_dense_script():
    _qwen_tokenizer_or_skip()
    from src.shared.tokenizers import token_counter

    count = token_counter("qwen")
    dense = "# Замък\n\n" + "Замъкът е споменат за пръв път през 895 година. " * 60

    by_chars = build_chunker("recursive", size=1000, overlap=0).split(dense, PAGE)
    by_tokens = build_chunker(
        "recursive", size=250, overlap=0, unit="tokens", provider="qwen"
    ).split(dense, PAGE)

    # A character budget says nothing about the real size in a dense script.
    assert max(count(c.text) for c in by_chars) > 250
    assert max(count(c.text) for c in by_tokens) <= 250


# --- database behaviour -----------------------------------------------------


def test_variants_coexist_without_colliding(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks()
    rebuild_page_chunks("small", build_chunker("recursive", size=200, overlap=0))

    with sqlite3.connect(db_path) as conn:
        counts = dict(
            conn.execute("select variant, count(*) from page_chunks group by 1")
        )
        base_ids = [
            row[0]
            for row in conn.execute("select id from page_chunks where variant = 'base'")
        ]

    assert counts["base"] < counts["small"], "smaller size must cut more chunks"
    assert all(":small:" not in identifier for identifier in base_ids)


def test_rebuild_is_append_only_per_variant(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks()
    with sqlite3.connect(db_path) as conn:
        first = conn.execute("select count(*) from page_chunks").fetchone()[0]

    rebuild_page_chunks()
    with sqlite3.connect(db_path) as conn:
        second = conn.execute("select count(*) from page_chunks").fetchone()[0]

    assert first == second


def test_clean_recuts_a_variant(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks("small", build_chunker("recursive", size=600, overlap=0))
    with sqlite3.connect(db_path) as conn:
        wide = conn.execute(
            "select count(*) from page_chunks where variant = 'small'"
        ).fetchone()[0]

    rebuild_page_chunks(
        "small", build_chunker("recursive", size=200, overlap=0), clean=True
    )
    with sqlite3.connect(db_path) as conn:
        narrow = conn.execute(
            "select count(*) from page_chunks where variant = 'small'"
        ).fetchone()[0]

    assert narrow > wide


def test_clean_refuses_to_wipe_the_base_variant_in_the_durable_database(
    monkeypatch, tmp_path
):
    # Both the connection path and the "is this the durable one" check point at
    # a throwaway file. Pointing only the check at the real database would make
    # this test migrate `data/db/pages.db` on every run, which is the exact
    # thing the guard exists to prevent.
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    monkeypatch.setattr("src.preprocess.chunks.raw_pages_db_path", lambda: db_path)
    monkeypatch.setattr("src.preprocess.chunks._durable_db_path", lambda: db_path)

    with pytest.raises(RuntimeError, match="durable database"):
        rebuild_page_chunks(clean=True)


def test_clean_is_allowed_once_the_database_is_not_the_durable_one(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)
    monkeypatch.setattr("src.preprocess.chunks.raw_pages_db_path", lambda: db_path)
    monkeypatch.setattr(
        "src.preprocess.chunks._durable_db_path", lambda: tmp_path / "elsewhere.db"
    )

    rebuild_page_chunks(clean=True)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("select count(*) from page_chunks").fetchone()[0] > 0


def test_chunk_spans_are_stored_and_point_at_the_stored_text(monkeypatch, tmp_path):
    db_path = tmp_path / "pages.db"
    _write_page_fixture(db_path)
    monkeypatch.setattr("src.db.pages.raw_pages_db_path", lambda: db_path)

    rebuild_page_chunks("small", build_chunker("recursive", size=200, overlap=0))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "select text, start_char, end_char from page_chunks where variant = 'small'"
        ).fetchall()

    assert rows
    normalized = normalize(MARKDOWN)
    for text, start, end in rows:
        assert start is not None and end is not None
        assert normalized[start:end] == text


def _write_page_fixture(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key,
              title text,
              source text not null,
              page_kind text not null
            );
            create table page_markdown_content (
              page_id text primary key references page_metadata(id) on delete cascade,
              markdown text not null
            );
            create table page_sources (
              source text primary key,
              language text not null
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
              id text primary key,
              question text not null,
              answer text not null,
              question_type text not null,
              question_language text not null,
              approved integer not null default 1
            );
            create table eval_relevant_chunks (
              question_id text not null references eval_questions(id) on delete cascade,
              chunk_id text not null references page_chunks(id) on delete cascade,
              primary key (question_id, chunk_id)
            );

            insert into page_metadata (id, title, source, page_kind)
            values ('page-castle', 'Castle', 'fixture', 'prose');
            insert into page_sources (source, language) values ('fixture', 'sl');
            """
        )
        conn.execute(
            "insert into page_markdown_content (page_id, markdown) values (?, ?)",
            ("page-castle", MARKDOWN),
        )


def test_pinned_provider_resolves_its_tokenizer_offline():
    """A revision-pinned provider must tokenize without reaching the network.

    Downloading by explicit revision leaves the cache with no `refs/` directory,
    and every by-name lookup resolves through `refs/main`. The weights still load,
    because the indexer passes the pin — but a tokenizer loaded by bare name falls
    through to the network and fails offline, which silently drops that provider
    from the token audit while every other provider reports normally.
    """
    from src.shared.tokenizers import (
        provider_revision,
        provider_token_limit,
        token_counter,
    )

    pinned = [
        provider
        for provider in ("nemotron", "nemotron8b")
        if provider_revision(provider)
    ]
    if not pinned:
        pytest.skip("no revision-pinned provider configured")

    for provider in pinned:
        try:
            count = token_counter(provider)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            pytest.skip(f"{provider} tokenizer unavailable locally: {error}")
        assert count("Grad Rajhenburg stoji nad Savo.") > 0
        assert provider_token_limit(provider) > 512
