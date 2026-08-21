"""Tests for whole-document diff marking in the page arena.

`marks.mark` keeps the whole output and marks which parts belong to only one side,
because the reviewer is judging the complete extraction, not one region of it.

Pure -- text in, spans out -- so no database and no HTTP.

The cases below pin what the UI depends on: reassembling spans must return the
original text exactly (otherwise the reviewer is voting on something we invented),
blank-line noise must not mark anything, and a one-character difference inside a
long line must mark that character rather than the whole paragraph -- which is the
case the corpus actually turns up (`[79]Dorothy`, `| 01 | Berlin |`).
"""

import pytest

from research.scrapers.arena import marks


def _text(spans, kind=None):
    return "".join(s.text for s in spans if kind is None or s.kind == kind)


# --- the contract that matters most: nothing is lost or invented ----------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("", ""),
        ("one line", "one line"),
        ("# T\n\nBody.\n", "# T\n\nOther.\n"),
        ("a\n\n\nb\n", "a\nb"),
        ("only left\n", ""),
        ("", "only right\n"),
        ("x\n\n", "x"),
        ("  leading\ttabs  \n", "leading tabs"),
        ("| 1 | Berlin |\n", "| 01 | Berlin |\n"),
        ("para one\n\npara two\n\npara three\n", "para one\n\npara three\n"),
    ],
)
def test_spans_reassemble_to_the_original_text(a, b):
    """Every character of both inputs survives, in order.

    This is the whole point of the page-arena view: the reviewer sees the complete
    output. A marking pass that silently dropped a blank line or reordered a block
    would make them vote on a document neither extractor produced.
    """
    marked = marks.mark(a, b)
    assert _text(marked.a) == a
    assert _text(marked.b) == b


def test_identical_text_marks_nothing():
    marked = marks.mark("# Title\n\nBody text.", "# Title\n\nBody text.")
    assert marked.regions == 0
    assert marked.identical
    assert _text(marked.a, "diff") == ""
    assert _text(marked.b, "diff") == ""
    assert marked.similarity == 1.0


def test_blank_line_differences_mark_nothing():
    """Same normalisation rule as `align`: blank-line count is not a difference.

    Extractors differ freely in how many newlines they put between blocks. Marking
    that would paint most of the document and hide the real differences.
    """
    marked = marks.mark("A\n\n\n\nB", "A\nB")
    assert marked.regions == 0
    assert _text(marked.a, "diff") == ""


def test_trailing_whitespace_marks_nothing():
    marked = marks.mark("A   \nB\t\n", "A\nB\n")
    assert marked.regions == 0


# --- one-sided content ---------------------------------------------------------


def test_line_only_on_the_left_is_marked_only_on_the_left():
    marked = marks.mark("keep\nLEFT ONLY\nkeep2", "keep\nkeep2")
    assert marked.regions == 1
    assert "LEFT ONLY" in _text(marked.a, "diff")
    assert _text(marked.b, "diff") == ""
    assert marked.a_only_lines == 1
    assert marked.b_only_lines == 0


def test_line_only_on_the_right_is_marked_only_on_the_right():
    marked = marks.mark("keep\nkeep2", "keep\nRIGHT ONLY\nkeep2")
    assert marked.regions == 1
    assert _text(marked.a, "diff") == ""
    assert "RIGHT ONLY" in _text(marked.b, "diff")
    assert marked.b_only_lines == 1


def test_dropped_infobox_marks_every_dropped_row_in_one_region():
    """The van Gogh case: one side loses a whole table, 34 rows.

    It must be one region -- one thing went wrong, so it is one thing to look at --
    not 34 separate marks the reviewer has to page through.
    """
    rows = "\n".join(f"| field {n} | value {n} |" for n in range(34))
    marked = marks.mark(f"# Van Gogh\n{rows}\nBody.", "# Van Gogh\nBody.")
    assert marked.regions == 1
    assert marked.a_only_lines == 34
    for n in range(34):
        assert f"value {n}" in _text(marked.a, "diff")


# --- intra-line marking: the defects this corpus actually produces --------------


def test_one_lost_space_marks_the_fused_word_not_the_paragraph():
    """The Scrapling defect: `notes.[79] Dorothy` -> `notes.[79]Dorothy`.

    Marking the whole 1,200-character paragraph would technically be correct and
    practically useless -- the reviewer cannot see which word moved. This is the
    case that decides whether the feature is worth having.
    """
    long_tail = " padding text that is identical on both sides." * 20
    a = f"Lovelace's notes.[79] Dorothy K. Stein regards it.{long_tail}"
    b = f"Lovelace's notes.[79]Dorothy K. Stein regards it.{long_tail}"
    marked = marks.mark(a, b)

    assert marked.regions == 1
    marked_a = _text(marked.a, "diff")
    marked_b = _text(marked.b, "diff")
    # The space *is* the entire difference, so that is all that may be marked.
    # Anything more means the reviewer is being pointed at text that did not change.
    assert marked_a == " "
    assert marked_b == ""
    assert "Dorothy" not in marked_a
    assert "padding text" not in marked_a


def test_zero_padded_cell_marks_just_the_digit():
    """`| 1 | Berlin |` vs `| 01 | Berlin |` -- the pre-clean defect."""
    marked = marks.mark("| 1 | Berlin | 3,600,000 |", "| 01 | Berlin | 3,600,000 |")
    assert marked.regions == 1
    assert "Berlin" not in _text(marked.a, "diff")
    assert "Berlin" not in _text(marked.b, "diff")
    assert "3,600,000" not in _text(marked.b, "diff")


def test_wholly_different_lines_are_marked_wholly():
    """When two lines share nothing, word-level marking would be noise."""
    marked = marks.mark("alpha beta gamma", "nothing alike here")
    assert marked.regions == 1
    assert _text(marked.a, "diff").strip() == "alpha beta gamma"
    assert _text(marked.b, "diff").strip() == "nothing alike here"


# --- navigation and reporting --------------------------------------------------


def test_regions_are_numbered_in_document_order_and_shared_across_sides():
    """Region ids let the UI jump A and B to the same difference together."""
    marked = marks.mark(
        "same1\nAAA\nsame2\nsame3\nBBB\nsame4",
        "same1\nXXX\nsame2\nsame3\nYYY\nsame4",
    )
    assert marked.regions == 2
    a_ids = sorted({s.region for s in marked.a if s.kind == "diff"})
    b_ids = sorted({s.region for s in marked.b if s.kind == "diff"})
    assert a_ids == [1, 2]
    assert b_ids == [1, 2]


def test_same_spans_carry_no_region():
    marked = marks.mark("keep\nDIFF A\nkeep2", "keep\nDIFF B\nkeep2")
    assert all(s.region == 0 for s in marked.a if s.kind == "same")


def test_similarity_falls_as_the_documents_diverge():
    close = marks.mark("a\nb\nc\nd\ne\nf\ng\nh\ni\nX", "a\nb\nc\nd\ne\nf\ng\nh\ni\nY")
    far = marks.mark("a\nb\nc\nd\ne", "v\nw\nx\ny\nz")
    assert close.similarity > 0.8
    assert far.similarity < 0.3


def test_empty_against_empty_is_identical():
    marked = marks.mark("", "")
    assert marked.identical
    assert marked.similarity == 1.0


def test_one_side_empty_marks_everything_on_the_other():
    marked = marks.mark("", "line one\nline two")
    assert marked.regions == 1
    assert marked.similarity == 0.0
    assert "line one" in _text(marked.b, "diff")
    assert "line two" in _text(marked.b, "diff")


def test_to_marked_json_round_trips_the_text():
    """The serialised form is what the client renders, so it carries the same guarantee."""
    a, b = "keep\nLEFT\nkeep2\n", "keep\nRIGHT\nkeep2\n"
    payload = marks.to_marked_json(marks.mark(a, b))
    assert "".join(s["text"] for s in payload["a"]) == a
    assert "".join(s["text"] for s in payload["b"]) == b
    assert payload["regions"] == 1
    assert payload["identical"] is False


def test_large_inputs_stay_bounded():
    """A 700k-char book against itself plus one edit must not blow up.

    Project Gutenberg's Pride and Prejudice is 716k chars in this corpus, and the
    page arena renders the whole thing. If marking were quadratic in characters it
    would hang the request.
    """
    body = "\n".join(f"Paragraph {n} with a reasonable amount of prose in it." for n in range(4000))
    marked = marks.mark(body, body.replace("Paragraph 2000", "Paragraph two-thousand"))
    assert marked.regions == 1
    assert _text(marked.a) == body


# --- section trails: getting from a diff back to the source page ----------------


def test_each_region_carries_the_heading_trail_above_it():
    """The reviewer has to check which side is right against the actual page.

    A region is useless for that without knowing where in the document it sits, so
    each one carries the heading path above it, outermost first.
    """
    a = "# Ada Lovelace\n## Early life\nShe was born.\n## Work\nOURS TEXT\n"
    b = "# Ada Lovelace\n## Early life\nShe was born.\n## Work\nTHEIRS TEXT\n"
    marked = marks.mark(a, b)
    assert marked.regions == 1
    assert marked.sections[1] == ["Ada Lovelace", "Work"]


def test_region_anchor_is_the_innermost_heading_as_a_url_fragment():
    """MediaWiki ids are the heading text with spaces as underscores.

    Verified against the saved corpus HTML, which carries `<h2 id="Adult_years">`.
    The same ids are present in the live page, so one anchor serves both the live
    iframe and the saved-DOM fallback.
    """
    a = "# T\n## Adult years\nOURS\n"
    b = "# T\n## Adult years\nTHEIRS\n"
    marked = marks.mark(a, b)
    assert marked.anchors[1] == "Adult_years"


def test_region_before_any_heading_has_an_empty_trail():
    marked = marks.mark("OURS\n# Later\nsame\n", "THEIRS\n# Later\nsame\n")
    assert marked.sections[1] == []
    assert marked.anchors[1] == ""


def test_trail_uses_the_heading_above_the_region_not_below_it():
    a = "## Before\nOURS\n## After\nshared\n"
    b = "## Before\nTHEIRS\n## After\nshared\n"
    marked = marks.mark(a, b)
    assert marked.sections[1] == ["Before"]


def test_sections_are_reported_for_every_region():
    a = "# T\n## One\nAAA\n## Two\nBBB\n"
    b = "# T\n## One\nXXX\n## Two\nYYY\n"
    marked = marks.mark(a, b)
    assert marked.regions == 2
    assert set(marked.sections) == {1, 2}
    assert marked.sections[1] == ["T", "One"]
    assert marked.sections[2] == ["T", "Two"]


def test_marked_json_carries_region_metadata():
    a = "# T\n## Work\nOURS\n"
    b = "# T\n## Work\nTHEIRS\n"
    payload = marks.to_marked_json(marks.mark(a, b))
    assert payload["sections"] == {"1": ["T", "Work"]}
    assert payload["anchors"] == {"1": "Work"}


def test_a_heading_that_is_itself_the_difference_still_gets_a_trail():
    """A dropped heading is a real defect we have already hit (`## Accessibility`)."""
    a = "# T\n## Accessibility\nbody\n"
    b = "# T\nbody\n"
    marked = marks.mark(a, b)
    assert marked.regions == 1
    assert marked.sections[1] == ["T"]


# --- coalescing: what makes collapsing actually work ---------------------------


def test_consecutive_shared_lines_become_one_span():
    """Otherwise the UI cannot collapse anything.

    `mark` walks the diff opcodes line by line, so a long identical run naturally
    comes out as hundreds of one-line spans. A collapse rule that fires on long
    spans then never fires at all -- which is exactly the bug this pins: the
    reviewer still faced the whole 90%-identical document.
    """
    body = "\n".join(f"identical line {n}" for n in range(500))
    marked = marks.mark(f"{body}\nLEFT\n", f"{body}\nRIGHT\n")
    same = [s for s in marked.a if s.kind == "same"]
    assert len(same) == 1, f"expected one merged span, got {len(same)}"
    assert same[0].text.count("identical line") == 500


def test_coalescing_preserves_the_text_exactly():
    body = "\n".join(f"line {n}" for n in range(200))
    a, b = f"{body}\nAAA\n{body}\n", f"{body}\nBBB\n{body}\n"
    marked = marks.mark(a, b)
    assert _text(marked.a) == a
    assert _text(marked.b) == b


def test_adjacent_diff_spans_in_one_region_merge():
    rows = "\n".join(f"| row {n} |" for n in range(30))
    marked = marks.mark(f"head\n{rows}\ntail", "head\ntail")
    diffs = [s for s in marked.a if s.kind == "diff"]
    assert len(diffs) == 1
    assert diffs[0].region == 1
    assert diffs[0].text.count("| row") == 30


def test_diff_spans_from_different_regions_do_not_merge():
    marked = marks.mark("s1\nAAA\ns2\ns3\nBBB\ns4", "s1\nXXX\ns2\ns3\nYYY\ns4")
    diffs = [s for s in marked.a if s.kind == "diff"]
    assert [s.region for s in diffs] == [1, 2]


def test_word_level_marks_inside_a_line_stay_separate():
    """Coalescing must not swallow the shared text between two intra-line marks."""
    marked = marks.mark("keep AAA keep BBB keep", "keep XXX keep YYY keep")
    kinds = [s.kind for s in marked.a]
    assert kinds.count("diff") == 2, kinds
    assert "same" in kinds


def test_span_count_stays_small_on_a_large_document():
    """Span count drives DOM nodes. 12,740 spans for one page was the old behaviour."""
    body = "\n".join(f"Paragraph {n} of prose." for n in range(4000))
    marked = marks.mark(body, body.replace("Paragraph 2000", "Paragraph two-thousand"))
    assert len(marked.a) < 20, f"{len(marked.a)} spans is too many to collapse usefully"


# --- one-sided regions must still be visible on the empty side -----------------


def test_a_region_missing_from_one_side_leaves_a_marker_there():
    """Otherwise the columns silently lose alignment.

    When B has content A lacks, A gets no span for that region at all -- so a
    collapsed column A renders as one "identical lines" stub with no indication that
    anything is missing, and the reviewer cannot see *where* the insertion belongs.
    A zero-length marker carries the position without adding text.
    """
    marked = marks.mark("keep\nkeep2", "keep\nRIGHT ONLY\nkeep2")
    a_markers = [s for s in marked.a if s.kind == "none"]
    assert len(a_markers) == 1
    assert a_markers[0].region == 1
    assert a_markers[0].text == ""
    # and the guarantee still holds
    assert _text(marked.a) == "keep\nkeep2"


def test_marker_side_is_the_empty_one():
    marked = marks.mark("keep\nLEFT ONLY\nkeep2", "keep\nkeep2")
    assert [s.kind for s in marked.b].count("none") == 1
    assert [s.kind for s in marked.a].count("none") == 0


def test_two_sided_regions_get_no_marker():
    marked = marks.mark("keep\nAAA\nkeep2", "keep\nBBB\nkeep2")
    assert not [s for s in marked.a if s.kind == "none"]
    assert not [s for s in marked.b if s.kind == "none"]


def test_markers_do_not_break_reassembly_anywhere():
    a = "shared\nonly-left-1\nonly-left-2\nshared2\n"
    b = "shared\nshared2\nonly-right\n"
    marked = marks.mark(a, b)
    assert _text(marked.a) == a
    assert _text(marked.b) == b


# --- section trails for entrants that emit no headings -------------------------


def test_trail_falls_back_to_the_other_side_when_this_side_has_no_headings():
    """resiliparse, justext, goose3 and html-text emit plain text, no ATX headings.

    Both columns are the same source document, so the side that *does* have headings
    can locate the region for the side that does not. Without this, every matchup
    involving a plain-text entrant loses source navigation entirely -- measured at 0
    of 13 regions anchored on `resiliparse` vs `resiliparse@rendered`.
    """
    a = "Some Heading\nbody text\nAAA\n"                 # no markdown headings
    b = "# Some Heading\n\nbody text\n\n## Details\nBBB\n"
    marked = marks.mark(a, b)
    assert marked.regions >= 1
    assert any(trail for trail in marked.sections.values()), marked.sections


# --- the trail index must agree with a naive backwards scan -----------------------


@pytest.mark.parametrize(
    "lines",
    [
        ["# A", "## B", "text", "### C", "more"],
        ["# T", "## A", "x", "## B", "y"],
        ["no headings", "at all", "here"],
        ["### deep first", "body", "# shallow later", "body2"],
        ["# A", "# B", "# C"],
        ["## only h2", "body"],
        [],
    ],
)
def test_trail_index_matches_section_trail_at_every_position(lines):
    """`mark` uses a one-pass index instead of rescanning from the top per region.

    Rescanning made marking quadratic in regions x lines: the slowest pair in the
    corpus went from 59ms to 1319ms, which is not acceptable inside a request. The
    fast index has to give the same answers as the obvious slow way, so this pins it
    against a naive backwards scan written independently here.
    """

    def naive_trail(upto: int) -> list[str]:
        """Walk backwards collecting the first heading of each shallower level."""
        trail: list[str] = []
        level = 7
        for line in reversed(lines[:upto]):
            match = marks.HEADING.match(line)
            if not match:
                continue
            found = len(match.group(1))
            if found < level:
                trail.append(match.group(2))
                level = found
        return list(reversed(trail))

    index = marks._trail_index(lines)
    assert len(index) == len(lines) + 1
    for position in range(len(lines) + 1):
        assert list(index[position]) == naive_trail(position), position


def test_marking_a_many_region_document_stays_fast():
    import time

    body = []
    for section in range(40):
        body.append(f"## Section {section}")
        body += [f"line {section}-{n} of shared prose" for n in range(100)]
    a = "\n".join(body)
    b = a.replace("of shared prose", "of altered prose")
    # Best of three. A single wall-clock sample fails on a machine that happens to be
    # busy, which says nothing about the algorithm; the floor is what we are pinning.
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        marked = marks.mark(a, b)
        samples.append((time.perf_counter() - started) * 1000)
    assert marked.regions >= 40
    assert min(samples) < 500, f"{min(samples):.0f}ms"


def test_lines_that_differ_throughout_are_not_word_diffed():
    """Two long lines with nothing in common must be marked whole, and fast.

    This is both the correct output and the affordable one. Word-diffing a pair of
    ~1,000-token lines that share only their spaces and punctuation is
    SequenceMatcher's worst case -- 59 such pairs on one corpus page cost 5.7
    seconds -- and the answer it produces is "almost everything differs", which a
    whole-line mark says more clearly.
    """
    import time

    a = "\n".join(
        " ".join(f"alpha{n}-{w}" for w in range(150)) for n in range(40)
    )
    b = "\n".join(
        " ".join(f"omega{n}-{w}" for w in range(150)) for n in range(40)
    )
    started = time.perf_counter()
    marked = marks.mark(a, b)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 400, f"{elapsed_ms:.0f}ms"
    assert _text(marked.a) == a
    assert _text(marked.b) == b


def test_a_small_edit_in_a_very_long_line_is_still_refined():
    """The cap applies to the *trimmed middle*, not the line length.

    Otherwise lowering it would break the case the feature exists for: one changed
    character inside a 3,000-character Wikipedia paragraph.
    """
    filler = " ".join(f"word{n}" for n in range(1200))
    marked = marks.mark(f"{filler} ONE {filler}", f"{filler} TWO {filler}")
    assert marked.regions == 1
    assert _text(marked.a, "diff").strip() == "ONE"
    assert _text(marked.b, "diff").strip() == "TWO"
