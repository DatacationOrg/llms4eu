"""Tests for the whole-output structural inventory carried in every judge brief.

The inventory exists because the brief clips long outputs: at the old 10k budget a
judge saw 5.7% of a 178k-character extraction, so a table lost by one entrant almost
always fell inside an elided window -- and "did one output lose the page's tables"
is the rubric's top criterion. These tests pin the two properties that make the
inventory trustworthy: it counts the *whole* string, and it does not overstate what
it found.
"""

import re

from research.scrapers.judge import inventory, inventory_table


def test_counts_cover_the_whole_string_not_an_excerpt():
    """The point of the inventory is to see past the clip."""
    body = "\n".join(f"| {index} | value |" for index in range(500))
    text = "# Heading\n\n" + ("filler line\n" * 5000) + body

    counts = inventory(text)

    assert counts["pipe-rows total"] == 500
    assert counts["characters"] == len(text)


def test_pipe_rows_recover_a_real_table():
    text = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"
    counts = inventory(text)
    assert counts["pipe-rows total"] == 4
    assert counts["pipe-row runs"] == 1


def test_separated_rows_count_as_separate_runs():
    """Named "runs", not "tables", precisely because this happens.

    A table whose rows are separated by blank lines reads as one run per row. An
    earlier label of "pipe-table blocks" invited a judge to read "32 blocks" as 32
    tables on a page whose HTML has 4.
    """
    text = "| a | b |\n\n| c | d |\n\n| e | f |\n"
    counts = inventory(text)
    assert counts["pipe-rows total"] == 3
    assert counts["pipe-row runs"] == 3


def test_plain_text_output_reports_no_table_markup():
    """The floor case: html-text keeps the values, loses the structure."""
    counts = inventory("Berlin 3644826 Germany\nMadrid 3223334 Spain\n")
    assert counts["pipe-rows total"] == 0
    assert counts["pipe-row runs"] == 0
    assert counts["characters"] > 0, "the data is still there; only structure is gone"


def test_atx_and_setext_headings_are_distinguished():
    """markdownify defaults to setext; the arena aligns dialects, so both are counted."""
    counts = inventory("# Real heading\n\nTitle\n=====\n\nOther\n-----\n")
    assert counts["ATX headings (# ..)"] == 1
    assert counts["setext underlines"] == 2


def test_hash_without_a_space_is_not_a_heading():
    assert inventory("#hashtag\n#1 best\n")["ATX headings (# ..)"] == 0


def test_markdown_links_and_bare_urls_are_counted_separately():
    text = "See [docs](https://example.org/a) and https://example.org/b for more.\n"
    counts = inventory(text)
    assert counts["markdown links"] == 1
    assert counts["bare URLs"] == 1, "the linked URL must not also count as bare"


def test_noise_markers_are_counted():
    counts = inventory("Ada Lovelace[1] wrote notes[27] on the engine.\nsnake\\_case\n")
    assert counts["citation markers"] == 2
    assert counts["backslash escapes"] == 1


def test_empty_output_reports_zeroes_not_a_crash():
    counts = inventory("")
    assert counts["characters"] == 0
    assert counts["pipe-rows total"] == 0


def test_table_states_source_ground_truth_and_both_outputs():
    rendered = "\n".join(
        inventory_table(
            [("1", "| a | b |\n| 1 | 2 |\n"), ("2", "a b\n1 2\n")],
            {"tables": 4, "rows": 98},
        )
    )

    assert "4 `<table>` element(s)" in rendered
    assert "98 `<tr>` row(s)" in rendered
    assert "OUTPUT 1" in rendered and "OUTPUT 2" in rendered
    # Every measure must appear for both outputs, or a judge reads a partial row.
    for line in rendered.splitlines():
        if line.startswith("| ") and "---" not in line and "measure" not in line:
            assert line.count("|") == 4, f"ragged inventory row: {line!r}"


def test_table_carries_the_caveat_against_mechanical_reading():
    """Without this, a plain-text entrant looks like it lost data it actually kept."""
    rendered = "\n".join(
        inventory_table([("1", "x"), ("2", "y")], {"tables": 0, "rows": 0})
    )
    assert "does not by itself prove the data is" in rendered
    assert "criterion-1b" in rendered and "criterion-1a" in rendered


def test_thousands_separators_keep_large_counts_readable():
    rendered = "\n".join(
        inventory_table([("1", "x" * 12345), ("2", "y")], {"tables": 0, "rows": 0})
    )
    assert "12,345" in rendered


def test_rubric_states_the_measured_calibration_anchors():
    """The draw-rate and position-bias numbers are results, so they must not drift silently."""
    from research.scrapers.judge import RUBRIC

    assert "63%" in RUBRIC, "position-bias anchor missing"
    assert "23%" in RUBRIC and "11%" in RUBRIC, "draw / both-bad anchors missing"
    assert "both_bad" in RUBRIC.lower() or "BOTH_BAD" in RUBRIC
    # Criterion order must match the reviewer's stated hierarchy.
    order = [
        RUBRIC.index("DATA AND TABLES"),
        RUBRIC.index("MARKDOWN FORMATTING"),
        RUBRIC.index("NOISE"),
        RUBRIC.index("TOKEN ECONOMY"),
    ]
    assert order == sorted(order), "criteria are out of the reviewer's priority order"
    assert "NOT a lexicographic tiebreak" in RUBRIC, "trade-off rule missing"


def test_inventory_is_cheap_enough_to_run_on_every_brief():
    import time

    text = ("| a | b |\n" * 20_000) + (
        "prose line with https://example.org/x\n" * 20_000
    )
    started = time.perf_counter()
    inventory(text)
    elapsed = (time.perf_counter() - started) * 1000
    assert elapsed < 500, f"inventory took {elapsed:.0f}ms on a 1MB output"


def test_every_measure_is_an_integer():
    """The brief formats with `:,`, which raises on anything else."""
    counts = inventory("# x\n| a | b |\nhttps://example.org\n")
    assert all(isinstance(value, int) for value in counts.values())
    assert not any(re.search(r"\s$", key) for key in counts), (
        "measure labels must be clean"
    )


# --------------------------------------------------------------- challenger round


def test_challenger_sets_pin_one_entrant_to_every_pair():
    """Three newcomers must cost three pairs, not six.

    A plain four-entrant focus set is a round robin, which spends half its votes
    comparing the newcomers to each other -- a question nobody asked, and one the
    ~12 votes/pair budget cannot afford alongside the one that was asked.
    """
    from research.scrapers.arena.app import CHAMPION_OF, FOCUS_SETS

    for mode, champion in CHAMPION_OF.items():
        entrants = FOCUS_SETS[mode]
        assert champion in entrants, f"{mode}: champion not in its own set"
        assert len(entrants) >= 3, f"{mode}: pinning is pointless below three entrants"


def test_content_three_way_is_a_round_robin_and_wiki_only():
    """The last matchup votes can still move, and the only one worth 20 more of them.

    The similarity matrix separates every other pair in the field for free -- the two
    families share 44-71% of their words, so a cross-family vote records a preference
    between two different documents. These three sit at 88-97% word-cosine, which is
    where a judge is the only instrument that works. So: full round-robin (no incumbent
    to defend, and `ours` is itself under test), and restricted to the pinned wiki
    pages, which are the only ones whose live reference matches the snapshot judged.
    """
    from research.scrapers.arena.app import CHAMPION_OF, FOCUS_SETS, WIKI_ONLY

    entrants = FOCUS_SETS["content3"]
    assert set(entrants) == {"ours", "trafilatura", "resiliparse"}
    assert "content3" not in CHAMPION_OF, "a champion would drop resiliparse vs trafilatura"

    pairs = [(a, b) for i, a in enumerate(entrants) for b in entrants[i + 1 :]]
    assert len(pairs) == 3, "three entrants unpinned is three pairs"
    assert "content3" in WIKI_ONLY


def test_wiki_only_modes_are_the_exception_and_are_named():
    """Every other mode keeps the whole corpus; the flag must not spread silently.

    The unpinned half is unlabellable because of *our* reference pane, not because of
    anything about those pages, so the fix is to serve the saved snapshot there -- at
    which point this set should empty out rather than grow.
    """
    from research.scrapers.arena.app import FOCUS_SETS, WIKI_ONLY

    assert WIKI_ONLY <= set(FOCUS_SETS), "a wiki-only mode with no entrant set"
    # Grown once, deliberately: `floorcheck` exists to settle whether the automated
    # board's leader beats trafilatura, and that question is only answerable on the
    # pinned slice -- the board's leader won its record on the unpinned one. Any
    # further addition should have to argue for itself here too.
    assert WIKI_ONLY == {
        "content3",
        "floorcheck",
        "committed",
        "crossplay",
        "converters",
        "bottomhalf",
    }
    assert set(FOCUS_SETS) - WIKI_ONLY, "the whole-corpus modes must not disappear"


def test_champion_filter_keeps_only_champion_pairs():
    names = ["crawl4ai", "docling", "markitdown", "trafilatura"]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    assert len(pairs) == 6, "round robin baseline"

    kept = [pair for pair in pairs if "trafilatura" in pair]
    assert len(kept) == 3
    assert all("trafilatura" in pair for pair in kept)
    # and every newcomer still gets exactly one pair
    others = sorted(n for pair in kept for n in pair if n != "trafilatura")
    assert others == ["crawl4ai", "docling", "markitdown"]


def test_challengers_are_registered_and_out_of_process():
    """They must be in the registry (so votes can name them) but refuse to run here.

    crawl4ai pulls 97 packages and docling a document-AI stack; neither can join
    arena-env. Raising loudly beats a run_extractors pass silently recording empties.
    """
    import pytest

    from research.scrapers.extractors import BY_NAME, CHALLENGERS

    for name in CHALLENGERS:
        entrant = BY_NAME[name]
        assert entrant.layer == "challenger"
        assert entrant.variants == ("raw",), "must read the same saved HTML as the rest"
        with pytest.raises(RuntimeError, match="out of process"):
            entrant.run({"raw": "<html></html>"}, {"id": "x", "is_wiki": 0})


def test_cosine_reads_100_for_the_same_words_and_0_for_disjoint_ones():
    """The deck's two headline matrices are this function; a silent skew is invisible.

    The 100 on the fetch-layer matrix is what licenses "the scrapers agree", and
    the ~48 between trafilatura and markitdown is what licenses "two families".
    Both claims are only as good as this.
    """
    from research.scrapers.similarity import cosine, tokens

    text = "Amsterdam is the capital of the Netherlands."
    assert cosine(tokens(text), tokens(text)) == 1.0
    # Punctuation, case and markdown syntax must not register as a difference.
    assert cosine(tokens(text), tokens(f"## {text.upper()}")) == 1.0
    assert cosine(tokens("alpha beta"), tokens("gamma delta")) == 0.0
    assert cosine(tokens("alpha"), tokens("")) == 0.0


def test_similarity_matrices_are_symmetric_and_carry_the_pair_size():
    """Off-diagonal asymmetry would mean the page-set intersection leaked."""
    from research.scrapers.similarity import _summarise

    texts = {
        "a": {"p1": "one two three", "p2": "four five"},
        "b": {"p1": "one two three", "p2": "six seven"},
        "c": {"p1": "one two three"},  # handles fewer pages than the others
    }
    out = _summarise(texts, ("a", "b", "c"))

    assert out["common_pages"] == 1
    assert out["cosine"]["a|a"]["pct"] == 100.0
    assert out["cosine"]["a|b"]["n"] == 2, "both handled p1 and p2"
    assert out["cosine"]["a|c"]["n"] == 1, "c only handled p1"
    assert out["exact"]["a|c"]["pct"] == 100.0
    assert out["chars"]["c"]["pages"] == 1


def test_floorcheck_is_a_two_entrant_duel_needing_no_champion_pin():
    """Two entrants make one pair, so pinning would be a no-op with a config to drift."""
    from research.scrapers.arena.app import CHAMPION_OF, FOCUS_SETS

    names = FOCUS_SETS["floorcheck"]
    assert set(names) == {"trafilatura", "scrapling-md"}
    assert "floorcheck" not in CHAMPION_OF
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    assert len(pairs) == 1


def test_committed_mode_pits_the_branch_against_what_is_actually_committed():
    """The one comparison 135 votes never made.

    `ours` resolves `extract_markdown` from the working tree, so every vote in the
    study describes the five uncommitted fixes. Without a frozen copy of the
    committed extractor there is no way to ask a judge whether those fixes are an
    improvement -- only whether they are a difference. Stock trafilatura is the
    third leg so the answer does not rest on our wrapper: if committed loses to
    both, the fixes stand on evidence rather than on measurement alone.
    """
    from research.scrapers.arena.app import CHAMPION_OF, FOCUS_SETS, WIKI_ONLY
    from research.scrapers.extractors import BY_NAME

    entrants = FOCUS_SETS["committed"]
    assert set(entrants) == {"ours", "ours@committed", "trafilatura"}
    assert "committed" not in CHAMPION_OF, "the control pair must be sampled too"
    assert "committed" in WIKI_ONLY
    assert BY_NAME["ours@committed"].variants == BY_NAME["ours"].variants, (
        "the two must read the same snapshots, or the vote is about inputs"
    )


def test_the_committed_extractor_is_a_frozen_copy_not_a_live_import():
    """It must not follow further edits to the working tree.

    If `ours@committed` imported the settings from src/, then editing src/ would
    silently redefine what past votes meant -- the exact failure this entrant
    exists to expose.
    """
    import inspect

    from research.scrapers.extractors import legacy_ours

    source = inspect.getsource(legacy_ours.extract_markdown_as_committed)
    for flag in ("include_images=True", "include_links=True",
                 "deduplicate=True", "favor_precision=True"):
        assert flag in source, f"{flag} missing from the frozen copy"
    assert "_remove_common_junk" in source, "the bs4 pre-clean is part of the copy"


def test_crossplay_covers_exactly_the_pairs_the_board_is_missing():
    """A Bradley-Terry fit cannot order the leaves of a star graph.

    With `ours` off the board, resiliparse / scrapling-md / ours@committed connect only
    through trafilatura -- 15, 14 and 9 votes against it, and zero against each other.
    Their ratings are then implied by one shared opponent, which is how two of them
    landed on the same number. This mode is the three missing edges and nothing else:
    trafilatura is deliberately absent, because adding the hub again would spend votes
    on edges that already carry evidence.
    """
    from research.scrapers.arena.app import CHAMPION_OF, FOCUS_SETS, WIKI_ONLY

    entrants = FOCUS_SETS["crossplay"]
    assert set(entrants) == {"ours@committed", "resiliparse", "scrapling-md"}
    assert "trafilatura" not in entrants, "the hub's edges are already judged"
    assert "crossplay" not in CHAMPION_OF, "all three edges are the point"
    assert "crossplay" in WIKI_ONLY
    pairs = [(a, b) for i, a in enumerate(entrants) for b in entrants[i + 1 :]]
    assert len(pairs) == 3


def test_converters_mode_is_only_the_edges_between_them():
    """Their trafilatura edges already exist; re-judging them would buy nothing.

    Each of the three has two wiki votes, all against trafilatura, which rates them but
    leaves them unordered relative to each other -- three more leaves of the same star
    that `crossplay` was built to eliminate. So this set excludes trafilatura on purpose.
    """
    from research.scrapers.arena.app import CHAMPION_OF, FOCUS_SETS, WIKI_ONLY

    entrants = FOCUS_SETS["converters"]
    assert set(entrants) == {"crawl4ai", "markitdown", "docling"}
    assert "trafilatura" not in entrants
    assert "converters" not in CHAMPION_OF, "all three edges are the point"
    assert "converters" in WIKI_ONLY


def test_bottomhalf_pins_the_entrant_whose_position_is_unevidenced():
    """The board says ours@committed is last; nothing it played says so directly.

    Its rating comes from trafilatura, resiliparse and scrapling-md; the converters'
    ratings come from trafilatura and each other. The two groups meet nowhere, so the
    gap between them is inferred through shared opponents rather than measured. Pinning
    to ours@committed spends all three pairs on closing exactly that gap.
    """
    from research.scrapers.arena.app import CHAMPION_OF, FOCUS_SETS, WIKI_ONLY

    entrants = FOCUS_SETS["bottomhalf"]
    assert set(entrants) == {"ours@committed", "crawl4ai", "markitdown", "docling"}
    assert CHAMPION_OF["bottomhalf"] == "ours@committed"
    assert "bottomhalf" in WIKI_ONLY
    kept = [(a, b) for i, a in enumerate(entrants) for b in entrants[i + 1:]
            if "ours@committed" in (a, b)]
    assert len(kept) == 3, "three pairs, not the six a round robin would cost"
