"""Whole-document diff marking between two extractor outputs.

The page arena asks "which of these two extractions is better", so the reviewer is
judging the complete output. They need the whole document, with the one-sided parts
marked so the differences are findable without reading both columns in full.

Two properties are load-bearing:

  1. Reassembling the spans returns the input byte-for-byte. The reviewer must be
     voting on what the extractor produced, not on our reconstruction of it.
  2. Differences are marked at word level inside a line when the two lines are
     related. Every defect this corpus has turned up is sub-line -- a lost space
     after a citation marker, a zero-padded sort key -- and marking the enclosing
     1,200-character Wikipedia paragraph would be correct and useless.

This module is deliberately pure: text in, marks out, no store and no HTTP. That
keeps it testable without a database and reusable by any caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# ATX headings in the extracted markdown. Used to tell the reviewer which section
# of the source page a marked region sits in -- without it, locating a difference
# in a long article means scrolling the whole thing looking for a match.
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Below this token-level similarity two lines are treated as unrelated and marked
# whole. Word-diffing "alpha beta gamma" against "nothing alike here" produces
# alternating specks that read as damage rather than as a difference.
MIN_LINE_SIMILARITY = 0.4

# Cap on the *trimmed middle* of a line pair -- what is left after the common prefix
# and suffix are removed. Above this the line is marked whole.
#
# The cap is low on purpose, and it is not a compromise. A large middle means the two
# lines differ throughout, so "the whole line differs" is the honest answer *and* the
# cheap one: word-diffing two ~1,000-token lines that share only spaces and punctuation
# is SequenceMatcher's worst case, and 59 such pairs on one corpus page cost 5.7
# seconds. Because the cap applies after trimming, the case this refinement exists for
# -- one changed character inside a 3,000-character paragraph -- is unaffected.
MAX_REFINE_TOKENS = 400

# Tokens for intra-line diffing: runs of word characters, or single non-word chars.
# Splitting on whitespace alone would hide the defect we most care about, because
# `notes.[79] Dorothy` and `notes.[79]Dorothy` differ only in whether a boundary
# exists -- so whitespace has to be its own token, not a separator.
_TOKEN = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Span:
    """A run of text in one side's output.

    `region` is 0 for shared text and 1..N for a differing region, numbered in
    document order and *shared between the two sides*, so the UI can scroll both
    columns to the same difference.
    """

    # "same" | "diff" | "none". A "none" span is a zero-length marker for a region
    # this side has nothing in, so the UI can show *where* the other side's content
    # would go. Without it, a collapsed column renders one "identical lines" stub and
    # the reviewer cannot see that anything is missing, or where.
    kind: str
    text: str
    region: int = 0


@dataclass
class Marked:
    """Both outputs in full, with their one-sided parts marked."""

    a: list[Span] = field(default_factory=list)
    b: list[Span] = field(default_factory=list)
    regions: int = 0
    same_lines: int = 0
    a_only_lines: int = 0
    b_only_lines: int = 0
    a_total_lines: int = 0
    b_total_lines: int = 0
    # region id -> heading trail above it, outermost first. Deciding *which side is
    # right* means checking the source page, and a region is useless for that without
    # a location. Keyed by region so the UI can label both columns identically.
    sections: dict[int, list[str]] = field(default_factory=dict)
    # region id -> URL fragment for the innermost heading. MediaWiki ids are the
    # heading text with spaces as underscores, and the saved snapshots carry the same
    # ids as the live page, so one anchor scrolls either reference.
    anchors: dict[int, str] = field(default_factory=dict)

    @property
    def identical(self) -> bool:
        return self.regions == 0

    @property
    def similarity(self) -> float:
        span = max(self.a_total_lines, self.b_total_lines)
        return (self.same_lines / span) if span else 1.0


@dataclass(frozen=True)
class _Block:
    """One comparable line plus the exact source text it stands for.

    `key` is the normalised line used for matching; `text` is everything the line
    occupies in the original, including any blank lines that preceded it. Keeping
    them separate is what lets blank-line noise be ignored for *matching* while the
    displayed document stays byte-identical to the extractor's output.
    """

    key: str
    text: str


def _blocks(text: str) -> list[_Block]:
    """Split into matchable blocks without losing a character.

    Blank and whitespace-only lines carry no key -- they attach to the next real
    line, or to a final keyless block when the document ends with them -- so a side
    that emitted two newlines where the other emitted one produces no difference.
    """
    blocks: list[_Block] = []
    pending = ""
    for line in text.splitlines(keepends=True):
        if line.strip():
            blocks.append(_Block(key=line.strip(), text=pending + line))
            pending = ""
        else:
            pending += line
    if pending:
        blocks.append(_Block(key="", text=pending))
    return blocks


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _refine(a_text: str, b_text: str) -> tuple[list[Span], list[Span]] | None:
    """Word-level marks for one related line pair, or None to mark it whole.

    Returns None when the two lines share too little to align meaningfully, or when
    the differing middle is large enough that refining costs more than it explains --
    in both cases the caller marks the whole line.

    Common prefix and suffix tokens are trimmed before diffing. This is what makes the
    function affordable: the case it exists for is a small edit inside a long line (one
    absent space in a 1,200-character Wikipedia paragraph), and a full token diff there
    is quadratic in the whole line for a two-token answer. Trimming is exact -- equal
    prefixes and suffixes are equal by construction, not by heuristic -- and it took the
    slowest corpus pair from 1,364ms to well under 100ms.
    """
    a_tokens, b_tokens = _tokens(a_text), _tokens(b_text)

    limit = min(len(a_tokens), len(b_tokens))
    head = 0
    while head < limit and a_tokens[head] == b_tokens[head]:
        head += 1
    tail = 0
    while tail < limit - head and a_tokens[-1 - tail] == b_tokens[-1 - tail]:
        tail += 1

    a_mid = a_tokens[head: len(a_tokens) - tail]
    b_mid = b_tokens[head: len(b_tokens) - tail]
    if max(len(a_mid), len(b_mid)) > MAX_REFINE_TOKENS:
        return None

    # Relatedness. When the shared prefix and suffix already account for most of the
    # line the two are plainly related and no comparison is needed -- which is both
    # cheaper and the overwhelmingly common case. Only when they share little at the
    # edges is the (now small) middle compared, and then on non-whitespace tokens
    # only: including whitespace let two entirely unrelated lines pass the threshold
    # on their shared spaces alone.
    # Counted on non-whitespace tokens only, for the same reason the fallback check
    # below is: a shared trailing newline is not evidence of relatedness. Counting it
    # made "LEFT\n" and "RIGHT\n" look 50% related, so they were word-diffed into a
    # mark plus a stray shared newline instead of being marked as two whole lines.
    def words(tokens: list[str]) -> int:
        return sum(1 for token in tokens if token.strip())

    shared_words = words(a_tokens[:head]) + words(a_tokens[len(a_tokens) - tail:])
    longest = max(words(a_tokens), words(b_tokens))
    if longest and shared_words / longest < MIN_LINE_SIMILARITY:
        a_words = [token for token in a_mid if token.strip()]
        b_words = [token for token in b_mid if token.strip()]
        if not a_words and not b_words:
            return None
        if SequenceMatcher(None, a_words, b_words, autojunk=False).ratio() < (
            MIN_LINE_SIMILARITY
        ):
            return None

    a_spans: list[Span] = []
    b_spans: list[Span] = []
    if head:
        a_spans.append(Span("same", "".join(a_tokens[:head])))
        b_spans.append(Span("same", "".join(b_tokens[:head])))
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, a_mid, b_mid, autojunk=False
    ).get_opcodes():
        a_part = "".join(a_mid[i1:i2])
        b_part = "".join(b_mid[j1:j2])
        if tag == "equal":
            if a_part:
                a_spans.append(Span("same", a_part))
                b_spans.append(Span("same", b_part))
            continue
        if a_part:
            a_spans.append(Span("diff", a_part))
        if b_part:
            b_spans.append(Span("diff", b_part))
    if tail:
        a_spans.append(Span("same", "".join(a_tokens[len(a_tokens) - tail:])))
        b_spans.append(Span("same", "".join(b_tokens[len(b_tokens) - tail:])))
    return a_spans, b_spans


def _trail_index(lines: list[str]) -> list[tuple[str, ...]]:
    """Heading trail above every position, in one forward pass.

    Walking backwards from each position to find its heading is quadratic for
    `mark`, which asks per region over a whole document: the slowest corpus pair
    went from 59ms to 1319ms that way -- and it is asked twice per region, once per
    side, for the plain-text fallback.

    Maintaining a stack forward gives the same answer in O(lines). Index `i` holds
    the trail above line `i`, so the list is one longer than `lines` and the last
    entry is the trail at end-of-document.
    """
    trails: list[tuple[str, ...]] = []
    stack: list[tuple[int, str]] = []
    for line in lines:
        trails.append(tuple(title for _, title in stack))
        match = HEADING.match(line)
        if not match:
            continue
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2)))
    trails.append(tuple(title for _, title in stack))
    return trails


def _coalesce(spans: list[Span]) -> list[Span]:
    """Merge neighbouring spans that carry the same kind and region.

    Not cosmetic. `mark` walks diff opcodes line by line, so an identical run comes
    out as one span *per line* -- 4,002 spans for a 4,000-line document. The UI's
    collapse rule fires on spans longer than a context budget, so against one-line
    spans it never fired at all and the reviewer still faced the whole
    90%-identical document. Merging first is what makes collapsing possible, and it
    cuts the DOM node count by three orders of magnitude on a large page.

    Word-level marks inside a single line are *not* affected: the shared text
    between two intra-line differences is a `same` span sitting between two `diff`
    spans, so there is nothing adjacent to merge it with.
    """
    merged: list[Span] = []
    for span in spans:
        if merged and merged[-1].kind == span.kind and merged[-1].region == span.region:
            merged[-1] = Span(span.kind, merged[-1].text + span.text, span.region)
        else:
            merged.append(span)
    return merged


def _stamp(spans: list[Span], region: int) -> list[Span]:
    """Assign a region id to the diff spans produced by `_refine`."""
    return [
        Span(s.kind, s.text, region if s.kind == "diff" else 0) for s in spans
    ]


def mark(text_a: str, text_b: str) -> Marked:
    """Both documents in full, with one-sided parts marked.

    Line-level alignment first, then word-level refinement inside `replace` regions
    whose two sides pair up one-to-one. Pairing is only attempted when the region
    has the same number of lines on both sides: that covers the reformatting case
    the corpus actually produces (same blocks, altered text) without inventing
    correspondences between, say, 3 lines and 11.
    """
    a_blocks, b_blocks = _blocks(text_a), _blocks(text_b)
    a_keys = [block.key for block in a_blocks]
    b_keys = [block.key for block in b_blocks]
    a_trails, b_trails = _trail_index(a_keys), _trail_index(b_keys)

    marked = Marked(
        a_total_lines=sum(1 for key in a_keys if key),
        b_total_lines=sum(1 for key in b_keys if key),
    )
    region = 0

    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, a_keys, b_keys, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                marked.a.append(Span("same", a_blocks[i1 + offset].text))
                marked.b.append(Span("same", b_blocks[j1 + offset].text))
            marked.same_lines += sum(1 for key in a_keys[i1:i2] if key)
            continue

        region += 1
        # Trail from A's keys: the heading above the region is shared by both sides
        # whenever there is one, and when the dropped text *is* the heading, A is the
        # side that still has the parent.
        # Prefer A's trail, fall back to B's. The plain-text entrants (resiliparse,
        # justext, goose3, html-text) emit no ATX headings at all, so a matchup between
        # two of them would lose source navigation entirely -- measured at 0 of 13
        # regions anchored. Both columns are the same source document, so whichever
        # side has headings can locate the region for the side that does not.
        trail = list(a_trails[i1] or b_trails[j1])
        marked.sections[region] = trail
        marked.anchors[region] = trail[-1].replace(" ", "_") if trail else ""
        a_side, b_side = a_blocks[i1:i2], b_blocks[j1:j2]
        marked.a_only_lines += sum(1 for block in a_side if block.key)
        marked.b_only_lines += sum(1 for block in b_side if block.key)

        if not a_side:
            marked.a.append(Span("none", "", region))
        if not b_side:
            marked.b.append(Span("none", "", region))

        paired = len(a_side) == len(b_side) and bool(a_side)
        for index in range(max(len(a_side), len(b_side))):
            a_block = a_side[index] if index < len(a_side) else None
            b_block = b_side[index] if index < len(b_side) else None

            refined = (
                _refine(a_block.text, b_block.text)
                if paired and a_block and b_block
                else None
            )
            if refined is not None:
                marked.a.extend(_stamp(refined[0], region))
                marked.b.extend(_stamp(refined[1], region))
                continue
            if a_block:
                marked.a.append(Span("diff", a_block.text, region))
            if b_block:
                marked.b.append(Span("diff", b_block.text, region))

    marked.regions = region
    marked.a = _coalesce(marked.a)
    marked.b = _coalesce(marked.b)
    return marked


def to_marked_json(marked: Marked) -> dict:
    """Serialise for the client. Entrant names are the caller's job to withhold."""
    return {
        "identical": marked.identical,
        "regions": marked.regions,
        "similarity": round(marked.similarity, 4),
        "same_lines": marked.same_lines,
        "a_only_lines": marked.a_only_lines,
        "b_only_lines": marked.b_only_lines,
        # JSON object keys are strings; the client indexes these by String(region).
        "sections": {str(k): v for k, v in marked.sections.items()},
        "anchors": {str(k): v for k, v in marked.anchors.items()},
        "a": [{"kind": s.kind, "text": s.text, "region": s.region} for s in marked.a],
        "b": [{"kind": s.kind, "text": s.text, "region": s.region} for s in marked.b],
    }
