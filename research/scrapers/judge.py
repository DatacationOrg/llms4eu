"""Automated pairwise judging by LLM subagents.

Why this exists: the arena needs hundreds of comparisons before Bradley-Terry
intervals tighten enough to separate the middle of the field, and a human has a
few hours. An LLM panel can produce a second, independent rating pool cheaply.

Three properties make the automated pool trustworthy enough to report:

* **Separate pool.** Votes are stored with `judge='llm'` and are never blended
  into the human ratings. An LLM panel and a human reviewer are different
  instruments; averaging them yields a number that describes neither. The report
  shows both leaderboards plus their rank correlation, which is a real result:
  it says how far an automated panel can substitute for a human here.
* **Non-competing sampling.** Each pool samples against only its own vote
  history, so automated judging never consumes matchups the human has not seen.
* **Blind by construction.** The brief handed to a judge contains the page, a
  neutral reference rendering, and two outputs labelled `1` and `2` in randomised
  order. Entrant identities are written to a separate keys directory that the
  judge is never pointed at, and are resolved only when the verdict is recorded.

The reference rendering is deliberately *not* produced by any entrant: using one
entrant's output as the yardstick would hand that entrant a perfect score on
coverage. It is a plain tag-strip of the same snapshot the entrants were given,
which is noisy on purpose -- it shows what is on the page, including boilerplate,
and leaves judging what belongs in the output to the judge.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from research.scrapers import rating, stats, store
from research.scrapers.extractors import BY_NAME, REGISTRY
from research.scrapers.sampler import Sampler


CONFIG = store.config()
RATING = CONFIG["rating"]
SAMPLING = CONFIG["sampling"]

# Character budgets for the brief. Raised from 6k/10k once the top criterion became
# "what happened to the page's tables": at a 10k budget a judge saw 5.7% of a 178k
# readability output, so a missing table almost always fell inside an elided window
# and the criterion was effectively unjudgeable. The judge pool runs as local Sonnet
# subagents, so a bigger brief costs context, not money -- and 30k covers a median
# output (40k chars) nearly whole. The structural inventory below covers what is
# still elided, so the stated totals are backed by real counts rather than guesswork.
REFERENCE_BUDGET = 8000
OUTPUT_BUDGET = 30000

BLOCK_TAGS = frozenset(
    """address article aside blockquote br caption dd div dl dt fieldset figcaption
    figure footer form h1 h2 h3 h4 h5 h6 header hr li main nav ol option p pre
    section table tbody td tfoot th thead tr ul""".split()
)


def judge_dir() -> Path:
    path = store.data_dir() / "judge"
    path.mkdir(parents=True, exist_ok=True)
    return path


def keys_dir() -> Path:
    """Entrant identities, kept out of the brief directory on purpose."""
    path = judge_dir() / "keys"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ reference


def visible_text(html: str) -> str:
    """Neutral tag-strip of a page: what a reader would see, boilerplate included.

    Not an entrant's output -- see the module docstring for why that matters.
    """
    import lxml.html

    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, 20000))
    try:
        document = lxml.html.document_fromstring(html)
        for node in document.xpath(
            "//script|//style|//noscript|//svg|//template|//iframe"
        ):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)
        parts: list[str] = []
        _walk(document, parts)
    except Exception:
        # A tag-strip failure must not take down a whole batch; the judge can
        # still compare the two outputs against each other.
        return ""
    finally:
        sys.setrecursionlimit(previous)

    text = "".join(parts)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines)).strip()


def _walk(element, parts: list[str]) -> None:
    tag = element.tag if isinstance(element.tag, str) else ""
    block = tag in BLOCK_TAGS
    if block:
        parts.append("\n")
    if element.text:
        parts.append(element.text)
    for child in element:
        if isinstance(child.tag, str):
            _walk(child, parts)
        if child.tail:
            parts.append(child.tail)
    if block:
        parts.append("\n")


def clip(text: str, budget: int, label: str) -> str:
    """Three-window excerpt: head, middle, tail.

    Head-and-tail alone is not good enough for the top criterion. readability's
    output on the Wikipedia Python article is 178k characters, so a 10k budget
    shows 5.7% of it; with only the ends visible a judge sees an infobox and a
    reference list and cannot tell whether the article body survived at all --
    which is precisely the coverage question it is being asked to rule on. A
    middle window costs nothing extra and makes that judgeable.

    The windows are labelled with their position so a judge can tell an elision
    from a truncation defect in the output itself.
    """
    if len(text) <= budget:
        return text

    head_size = int(budget * 0.45)
    middle_size = int(budget * 0.30)
    tail_size = budget - head_size - middle_size
    middle_start = max(head_size, (len(text) - middle_size) // 2)

    def gap(start: int, end: int) -> str:
        return (
            f"\n\n[... {end - start:,} characters of {label} elided "
            f"({100.0 * start / len(text):.0f}%-{100.0 * end / len(text):.0f}% "
            f"of the way through) ...]\n\n"
        )

    return (
        text[:head_size]
        + gap(head_size, middle_start)
        + text[middle_start : middle_start + middle_size]
        + gap(middle_start + middle_size, len(text) - tail_size)
        + text[len(text) - tail_size :]
    )


# --------------------------------------------------------------------- brief


RUBRIC = """\
You are judging two candidate text extractions of the same web page, produced by
two different (unnamed) HTML-to-text tools. Decide which extraction is better for
use as training/retrieval corpus text.

What a good extraction is: the page's real content, structured, with the
surrounding furniture of the website removed.

Criteria, in priority order:

1. DATA AND TABLES. Ask what happened to the page's data first -- tables,
   infoboxes, statistics, key-value blocks, figure captions -- in these two steps,
   in this order:
   a. Did one output FIND the data while the other lost it? A large table or
      infobox that the page contains and an output does not is the single worst
      failure on this list.
   b. If both found it, did one preserve its STRUCTURE? Rows that read as rows and
      columns that stay separated beat the same values flattened into a prose
      run-on. A table melted into a paragraph has lost real information even when
      every character survives.
   Then the same two questions for the body text: losing whole sections or the
   bulk of the article is a coverage failure of the same kind.
2. MARKDOWN FORMATTING. Output that is already usable markdown -- ATX headings
   that read as headings, lists as lists, tables as pipe tables -- earns real
   credit, because the alternative is a conversion step downstream that has to be
   written and maintained. An unstructured wall of text is worse than clean
   markdown carrying the same content.
3. NOISE. Leftover link dumps and bare URLs, navigation, cookie and consent
   banners, language switchers, login prompts, "edit" links, "share this" widgets,
   repeated headers and footers, stray escape characters, setext underlines,
   citation markers and footnote lists. A little is cheap to tolerate; bulk is not.
4. TOKEN ECONOMY. Given comparable content, fewer characters is better.

This is a priority order, NOT a lexicographic tiebreak. A deficit high on the list
can be paid for by a large enough advantage lower down. The reviewer whose labels
this rubric encodes gives this worked example: one output missed a single table,
but the document was long and that output's markdown was so much more usable
across the whole of it that the correct verdict was a DRAW. So weigh how much of
the document each difference touches, not merely which criterion it lands on.

Two failure modes, and they are NOT symmetric in the way you might assume:

* Under-extraction: dropping real content. Judge this harshly.
* Over-extraction: dumping the whole page, furniture included. Also judge this
  harshly. An output that simply reproduces everything on the page trivially has
  perfect coverage and is still a poor extraction -- that is the problem being
  solved here, not the solution. Do not reward it for completeness.

So "kept more text" is not the same as "better". Ask which output better isolates
the content from the page around it.

Rules:

* Judge on the criteria above only. Ignore presentation order -- position carries
  no information. This warning is not decorative: an earlier panel run under this
  rubric picked the first-shown output in 63% of its decisive verdicts while the
  human reviewer split 50/50 on the same kind of matchups. If you find yourself
  leaning toward OUTPUT 1, re-read OUTPUT 2 before committing.
* DRAW and BOTH_BAD are legitimate, expected verdicts, and the panel has
  historically under-used them. On this corpus the human reviewer drew on 23% of
  matchups and called both-bad on a further 11%; an earlier panel drew on 15% and
  never once said both-bad. Do not invent a distinction to force a winner.
  - DRAW: the two are genuinely close, differ only in trivial noise, or trade one
    real advantage against another of comparable weight.
  - BOTH_BAD: both failed badly in absolute terms -- both lost most of the page,
    or both are mostly furniture. Prefer this over DRAW when neither output would
    be acceptable to use, because "equally good" and "equally useless" are
    different findings.
* The REFERENCE RENDERING is a raw tag-strip of the page. It is deliberately noisy
  and includes all the navigation and boilerplate. It shows you WHAT IS ON THE PAGE
  so you can check coverage. It is NOT a target to imitate: an output resembling it
  closely has failed to remove anything. Note that a tag-strip cannot show you
  table structure -- the markup is exactly what it discarded -- so use the
  structural inventory for that.
* Long excerpts are shown as three windows -- beginning, middle and end -- with the
  true character total stated, so you can reason about the parts you cannot see.
  The structural inventory is computed over the WHOLE output, not the excerpt, so
  it is the reliable source for anything you cannot see.
"""


_shape_cache: dict[str, dict[str, int]] = {}


def source_shape(page_id: str) -> dict[str, int]:
    """Table counts taken from the snapshot HTML, as ground truth for criterion 1a.

    The reference rendering cannot supply this: it is a tag-strip, and the markup is
    precisely what it threw away. Counting `<table>` and `<tr>` in the saved HTML is
    the one number that lets a judge notice a table both the excerpt windows and the
    reference are blind to.
    """
    if page_id in _shape_cache:
        return _shape_cache[page_id]

    shape = {"tables": 0, "rows": 0}
    for variant in ("rendered", "raw"):
        html = store.snapshot_text(page_id, variant)
        if html:
            shape = {
                "tables": len(re.findall(r"<table[\s>]", html, re.IGNORECASE)),
                "rows": len(re.findall(r"<tr[\s>]", html, re.IGNORECASE)),
            }
            break
    _shape_cache[page_id] = shape
    return shape


def inventory(text: str) -> dict[str, int]:
    """Structural fingerprint of one whole output.

    Every count here is over the full string, never the excerpt. That is the point:
    the brief clips long outputs, so a lost table can sit entirely inside an elided
    window, and the top criterion would be decided on text the judge never saw.
    """
    lines = text.split("\n")

    # Counted as runs of consecutive pipe rows, not as tables: a table whose rows are
    # separated by blank lines reads as one run per row. Naming it "blocks" invited a
    # judge to read "32 blocks" as 32 tables on a page that has 4.
    blocks = 0
    previous_row = False
    for line in lines:
        row = line.count("|") >= 2
        if row and not previous_row:
            blocks += 1
        previous_row = row

    return {
        "characters": len(text),
        "lines": len(lines),
        "ATX headings (# ..)": sum(1 for line in lines if re.match(r"^#{1,6} ", line)),
        "setext underlines": sum(
            1 for line in lines if re.fullmatch(r"[=-]{3,}", line.strip())
        ),
        "pipe-row runs": blocks,
        "pipe-rows total": sum(1 for line in lines if line.count("|") >= 2),
        "list items": sum(
            1 for line in lines if re.match(r"^\s*([-*+]|\d+\.)\s", line)
        ),
        "markdown links": len(re.findall(r"\]\(\s*https?://", text)),
        "bare URLs": len(re.findall(r"(?<!\()\bhttps?://", text)),
        "citation markers": len(re.findall(r"\[\d+\]", text)),
        "backslash escapes": len(re.findall(r"\\[_*\[\]()#]", text)),
    }


def inventory_table(outputs: list[tuple[str, str]], shape: dict[str, int]) -> list[str]:
    """The inventory rendered for the brief, with the caveat that stops misreading it."""
    counts = [(label, inventory(text)) for label, text in outputs]
    measures = list(counts[0][1])

    lines = [
        "## Structural inventory (whole output, not just the excerpt below)",
        "",
        f"The page's own HTML holds {shape['tables']:,} `<table>` element(s) "
        f"and {shape['rows']:,} `<tr>` row(s).",
        "",
        "| measure | " + " | ".join(f"OUTPUT {label}" for label, _ in counts) + " |",
        "| --- | " + " | ".join("---:" for _ in counts) + " |",
    ]
    for measure in measures:
        cells = " | ".join(f"{values[measure]:,}" for _, values in counts)
        lines += [f"| {measure} | {cells} |"]
    lines += [
        "",
        "Read this carefully rather than mechanically. Zero pipe-table rows means the",
        "output carries no table *markup*; it does not by itself prove the data is",
        "gone, because a plain-text tool may still have kept every value as running",
        "text. That is a criterion-1b failure (structure lost), which is milder than a",
        "criterion-1a failure (data lost). Check the excerpt before deciding which one",
        "you are looking at. Equally, a high heading or table count is not a win on its",
        "own -- it can just as easily mean navigation furniture was formatted, not that",
        "content was preserved.",
        "",
    ]
    return lines


def write_brief(
    path: Path, page: dict, reference: str, outputs: list[tuple[str, str]]
) -> None:
    lines = [
        "# Extraction comparison",
        "",
        RUBRIC,
        "",
        "---",
        "",
        "## Page",
        "",
        f"- URL: {page['url']}",
        f"- Title: {page.get('title') or '(unknown)'}",
        f"- Language: {page['lang']}",
        f"- Page type: {page.get('shape') or 'unknown'}",
        "",
        *inventory_table(outputs, source_shape(page["id"])),
        f"## Reference rendering (noisy tag-strip, {len(reference):,} chars total)",
        "",
        "```",
        clip(reference, REFERENCE_BUDGET, "the reference") or "(unavailable)",
        "```",
        "",
    ]
    for label, text in outputs:
        lines += [
            f"## OUTPUT {label} ({len(text):,} chars total)",
            "",
            "```",
            clip(text, OUTPUT_BUDGET, f"output {label}") or "(empty)",
            "```",
            "",
        ]
    lines += [
        "---",
        "",
        "## Your verdict",
        "",
        "Reply with exactly two lines, nothing else:",
        "",
        "```",
        "VERDICT: 1 | 2 | draw | both_bad",
        "REASON: <one sentence, max 25 words>",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------- prepare


_reference_cache: dict[str, str] = {}


def _reference_for(page_id: str) -> str:
    """Prefer the rendered DOM: it is the superset, and it is what the human sees.

    The human reviewer's reference pane serves the rendered page, so using the
    rendered DOM here keeps the two judging setups looking at the same document.
    """
    # Cached: a batch often draws several matchups from the same page, and parsing a
    # 600 KB document repeatedly is pure waste.
    if page_id in _reference_cache:
        return _reference_cache[page_id]

    reference = ""
    for variant in ("rendered", "raw"):
        html = store.snapshot_text(page_id, variant)
        if html:
            text = visible_text(html)
            if text:
                reference = text
                break
    _reference_cache[page_id] = reference
    return reference


def prepare(count: int, seed: int | None = None) -> Path:
    store.initialize()
    votes = store.load_votes(judge=store.LLM)
    names = [entrant.name for entrant in REGISTRY]
    elo, _, _ = rating.online_elo(
        votes, names, float(RATING["initial"]), float(RATING["k_factor"])
    )

    sampler = Sampler(
        pages=store.load_pages(),
        run_index=store.load_run_index(),
        epsilon=float(SAMPLING["epsilon"]),
        candidate_pairs=int(SAMPLING["candidate_pairs"]),
        seed=seed,
    )

    existing = sorted(
        int(match.group(1))
        for path in judge_dir().glob("batch-*")
        if (match := re.fullmatch(r"batch-(\d+)", path.name))
    )
    batch = judge_dir() / f"batch-{(existing[-1] if existing else 0) + 1:03d}"
    batch.mkdir(parents=True, exist_ok=True)

    generator = random.Random(seed)
    # Matchups chosen in this batch are not in the database yet, so they are fed
    # back to the sampler as pending votes. Otherwise a batch of 20 would happily
    # pick the same (page, pair) twenty times. They inform the seen-set and the
    # per-page/per-bucket balance only; ELO still comes from real votes.
    pending: list[dict] = []
    written: list[Path] = []
    auto_total = 0

    for index in range(1, count + 1):
        selected, auto_draws = sampler.next_matchup(votes + pending, elo)
        for draw in auto_draws:
            store.append_vote(
                draw.page_id,
                draw.entrant_a,
                draw.entrant_b,
                "draw",
                auto=True,
                judge=store.LLM,
                reason="byte-identical output",
            )
            # Also append to `pending`. `votes` was read once before the loop, so a
            # pair auto-drawn on iteration 3 is invisible to the sampler's seen-set
            # on iteration 4 and would be selected and recorded a second time,
            # double-counting draws in the ratings.
            pending.append(
                {
                    "page_id": draw.page_id,
                    "entrant_a": draw.entrant_a,
                    "entrant_b": draw.entrant_b,
                    "winner": "draw",
                    "auto": 1,
                }
            )
            auto_total += 1
        if selected is None:
            print(f"matchups exhausted after {index - 1}")
            break

        page = selected.page
        text_a = store.run_output(page["id"], selected.entrant_a) or ""
        text_b = store.run_output(page["id"], selected.entrant_b) or ""

        # Randomise the brief's 1/2 labelling independently of the sampler's
        # left/right choice, so the stored entrant_a stays a faithful record of
        # what was shown as OUTPUT 1 and position bias remains measurable.
        flip = generator.random() < 0.5
        first, second = (
            (selected.entrant_b, selected.entrant_a)
            if flip
            else (
                selected.entrant_a,
                selected.entrant_b,
            )
        )
        first_text, second_text = (text_b, text_a) if flip else (text_a, text_b)

        name = f"m{index:03d}"
        write_brief(
            batch / f"{name}.md",
            page,
            _reference_for(page["id"]),
            [("1", first_text), ("2", second_text)],
        )
        (keys_dir() / f"{batch.name}-{name}.json").write_text(
            json.dumps(
                {
                    "batch": batch.name,
                    "matchup": name,
                    "page_id": page["id"],
                    "url": page["url"],
                    "bucket": page["bucket"],
                    "output_1": first,
                    "output_2": second,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(batch / f"{name}.md")
        pending.append(
            {
                "page_id": page["id"],
                "entrant_a": first,
                "entrant_b": second,
                "winner": "draw",
                "auto": 0,
            }
        )

    print(
        f"batch {batch.name}: {len(written)} briefs, {auto_total} auto-draws recorded"
    )
    for path in written:
        print(path)
    return batch


# -------------------------------------------------------------------- record

VERDICTS = {"1": "a", "2": "b", "draw": "draw", "both_bad": "both_bad"}


def record(matchup: str, verdict: str, reason: str | None) -> None:
    verdict = verdict.strip().lower()
    if verdict not in VERDICTS:
        raise SystemExit(f"bad verdict {verdict!r}; expected one of {sorted(VERDICTS)}")

    path = Path(matchup)
    stem = path.stem if path.suffix else path.name
    if path.parent.name.startswith("batch-"):
        key_path = keys_dir() / f"{path.parent.name}-{stem}.json"
    else:
        key_path = keys_dir() / f"{stem}.json"

    if not key_path.exists():
        # A judge that passed a bare matchup name rather than a full path would
        # otherwise lose its vote. Fall back to matching on the matchup name, and
        # only refuse if that is ambiguous.
        candidates = sorted(keys_dir().glob(f"*-{stem}.json"))
        if len(candidates) == 1:
            key_path = candidates[0]
        elif len(candidates) > 1:
            raise SystemExit(
                f"{stem!r} is ambiguous across batches "
                f"({', '.join(c.name for c in candidates)}); pass the full path"
            )
        else:
            raise SystemExit(f"no key for {matchup!r} (looked for {key_path})")

    key = json.loads(key_path.read_text(encoding="utf-8"))
    entrant_1, entrant_2 = key["output_1"], key["output_2"]
    for name in (entrant_1, entrant_2):
        if name not in BY_NAME:
            raise SystemExit(f"unknown entrant in key: {name}")

    # Defence in depth behind `pending`: a brief whose page has left the corpus
    # (repinning changes the URL, and the id is uuid5 of the URL) must not be able
    # to record a vote, however its path reached this function.
    if key["page_id"] not in {page["id"] for page in store.load_pages()}:
        raise SystemExit(
            f"{key['batch']}/{key['matchup']} is for page {key['page_id']}, "
            "which is no longer in the corpus; the brief is stale, not judgeable"
        )

    # `output_1` was shown first, so it is stored as entrant_a. Position bias for
    # the LLM pool is then measurable exactly as it is for the human pool.
    already = [
        vote
        for vote in store.load_votes(judge=store.LLM)
        if vote["page_id"] == key["page_id"]
        and {vote["entrant_a"], vote["entrant_b"]} == {entrant_1, entrant_2}
    ]
    if already:
        print(f"already recorded ({key['batch']}/{key['matchup']}); ignoring duplicate")
        return

    vote_id = store.append_vote(
        key["page_id"],
        entrant_1,
        entrant_2,
        VERDICTS[verdict],
        auto=False,
        judge=store.LLM,
        reason=(reason or "").strip()[:300] or None,
    )
    winner = {"a": entrant_1, "b": entrant_2}.get(VERDICTS[verdict], "draw")
    judged, auto = store.vote_count(judge=store.LLM)
    print(
        f"vote {vote_id} recorded: {key['matchup']} -> {verdict} ({winner}); "
        f"llm pool now {judged} judged + {auto} auto"
    )


def pending(limit: int | None = None, echo: bool = False) -> list[Path]:
    """Briefs that exist on disk but have no recorded vote yet.

    Makes the dispatch loop resumable: the database is the source of truth for what
    has been judged, so an interrupted run (or a fresh session with no memory of
    which batch it reached) can pick up exactly where it left off without
    re-judging anything or losing a brief.

    Briefs whose page has left the corpus are dropped rather than listed. Page ids
    are `uuid5` of the URL, so repinning the wiki set to `?oldid=` permalinks gave
    every one of those pages a new id and orphaned 149 staged briefs. `refresh`
    already skipped them -- it cannot rebuild a brief for a page it cannot load --
    so they would have sat here forever as un-refreshable work, and a judge
    dispatched at one would have recorded a vote against a page no longer in the
    corpus. Listing only what can actually be judged keeps the two in step.
    """
    pages = {page["id"] for page in store.load_pages()}
    voted = {
        (vote["page_id"], *sorted((vote["entrant_a"], vote["entrant_b"])))
        for vote in store.load_votes(judge=store.LLM)
    }
    outstanding: list[Path] = []
    for key_path in sorted(keys_dir().glob("*.json")):
        key = json.loads(key_path.read_text(encoding="utf-8"))
        if key["page_id"] not in pages:
            continue
        signature = (key["page_id"], *sorted((key["output_1"], key["output_2"])))
        if signature in voted:
            continue
        brief = judge_dir() / key["batch"] / f"{key['matchup']}.md"
        if brief.exists():
            outstanding.append(brief)
    if limit:
        outstanding = outstanding[:limit]
    if echo:
        for path in outstanding:
            print(path)
    return outstanding


def refresh() -> int:
    """Rewrite every unjudged brief from current database state.

    An adapter fix regenerates output, which silently invalidates any brief
    already staged on disk: the file still shows the old string. That is how a
    judge once described a wikiextractor output as "just a redirect stub" long
    after the redirect handling had been fixed. Rewriting is unconditional rather
    than change-detecting -- it is cheap, idempotent, and leaves no room for a
    stale brief to survive because the diffing logic missed it.

    Only unjudged briefs are touched. A brief that has already been voted on is
    the evidence for that vote and must keep saying what the judge actually saw;
    if its output has changed, the vote needs `store.invalidate_votes`, not a
    quietly edited brief.
    """
    pages = {page["id"]: page for page in store.load_pages()}
    rewritten = 0
    for brief in pending():
        stem = brief.stem
        key_path = keys_dir() / f"{brief.parent.name}-{stem}.json"
        if not key_path.exists():
            continue
        key = json.loads(key_path.read_text(encoding="utf-8"))
        page = pages.get(key["page_id"])
        if page is None:
            continue
        outputs = [
            (label, store.run_output(key["page_id"], key[f"output_{label}"]) or "")
            for label in ("1", "2")
        ]
        write_brief(brief, page, _reference_for(key["page_id"]), outputs)
        rewritten += 1
    print(f"rewrote {rewritten} unjudged briefs from current output")
    return rewritten


def status() -> None:
    pools = store.judge_pools()
    print("vote pools:", pools or "(none)")
    for pool in sorted(pools):
        judged, auto = store.vote_count(judge=pool)
        print(f"  {pool:8s} judged={judged:5d} auto={auto:5d}")


def agreement() -> None:
    """Print human-vs-panel agreement on the cells both pools judged.

    The command-line view of the report's calibration section, so a judging session
    can check whether the overlap is growing without opening the HTML.
    """

    result = stats.judge_cell_agreement()
    print(f"human judged : {result['human_judged']}")
    print(f"panel judged : {result['llm_judged']}")
    print(f"shared cells : {result['cells']}")
    if not result["cells"]:
        print(
            f"\nNo shared comparison yet, so agreement is not measurable. "
            f"{result['candidates']} panel cells are available to re-check -- run the "
            f"arena with calibration mode on."
        )
        return

    kappa = result["kappa"]
    print(f"exact        : {result['exact']}%")
    print(f"equivalent   : {result['equivalent']}%   (draw == both_bad)")
    print(f"decisive     : {result['decisive']}%   of {result['decisive_n']}")
    print(f"kappa        : {kappa if kappa is not None else 'undefined'}")

    def rate(value, count):
        return f"{value}% of {count}" if count else "n/a"

    print(
        f"order split  : same {rate(result['same_order'], result['same_order_n'])}, "
        f"flipped {rate(result['flipped_order'], result['flipped_order_n'])}"
    )
    print(f"un-rechecked : {result['candidates']} panel cells")
    if result["disagreements"]:
        print("\ndisagreements:")
        for row in result["disagreements"]:
            print(
                f"  {row['pair']:44s} human={row['human']:22s} panel={row['llm']:22s}"
                f" {row['page'][:40]}"
            )


def focus(entrants: str, pool: str | None = None) -> None:
    """Print closed-set standings: the readout for a duel or a 3-way."""

    names = tuple(name.strip() for name in entrants.split(",") if name.strip())
    if len(set(names)) < 2:
        print("need at least two entrant names, comma-separated")
        return

    result = stats.subset_standings(names, pool)
    print(f"closed set: {', '.join(result['entrants'])}")
    print(
        f"pool={result['judge']}  pages where all produced output={result['pages']}  "
        f"votes={result['votes']} ({result['focus_votes']} cast in focus mode)"
    )
    if not result["votes"]:
        print("\nNo votes on this set yet -- pick it in the arena's mode selector.")
        return

    print()
    header = f"{'entrant':22s} {'BT':>6s} {'95% CI':>13s} {'W':>3s} {'L':>3s} {'D':>3s}"
    print(header)
    for row in result["rows"]:
        interval = f"{row['ci_low']}-{row['ci_high']}"
        print(
            f"{row['entrant']:22s} {row['bt']:6d} {interval:>13s} "
            f"{row['wins']:3g} {row['losses']:3g} {row['draws']:3d}"
        )

    print("\nhead to head:")
    matrix = result["matrix"]
    for a in result["entrants"]:
        for b in result["entrants"]:
            if a >= b:
                continue
            cell = matrix.get(a, {}).get(b, {})
            if not cell.get("games"):
                print(f"  {a} vs {b}: no votes yet")
                continue
            print(
                f"  {a} {cell['wins']:g} - {cell['losses']:g} {b}"
                f"   draws {cell['draws']}   ({cell['games']} votes)"
            )

    if result["per_language"]:
        print("\nby language (the SIGIR multilingual question):")
        for label, entry in result["per_language"].items():
            ranked = sorted(entry["bt"].items(), key=lambda kv: -kv[1])
            listing = "  ".join(f"{name} {value}" for name, value in ranked)
            print(f"  {label:6s} {entry['votes']:3d} votes   {listing}")


def duel(a: str, b: str, pool: str | None = None) -> None:
    """Print one pair's head-to-head record, split by page shape."""

    result = stats.head_to_head_by_shape(a, b, pool)
    print(f"{result['a']} vs {result['b']}   pool={result['judge']}")
    print(f"{result['games']} votes ({result['duel_votes']} cast in duel mode)\n")
    if not result["games"]:
        print("No votes on this pair yet -- pick 'duel' in the arena's mode selector.")
        return

    for kind, entry in result["split"].items():
        if not entry["games"]:
            print(f"{kind:12s} no votes yet")
            continue
        rate = entry["win_rate_a"]
        print(
            f"{kind:12s} {entry['games']:2d} votes   "
            f"{result['a']} {entry['wins_a']} - {entry['wins_b']} {result['b']}"
            f"   draws {entry['draws']}"
            + (f"   {result['a']} win rate {rate}%" if rate is not None else "")
        )
        for row in entry["pages"]:
            mark = " [duel]" if row["duel"] else ""
            print(
                f"    {row['winner']:22s} {str(row['chars_a'] or 0):>8s} vs "
                f"{str(row['chars_b'] or 0):>8s} chars  "
                f"{(row['shape'] or '?'):22s} {row['url'][:44]}{mark}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("prepare", help="write a batch of blinded briefs")
    make.add_argument("--count", type=int, default=10)
    make.add_argument("--seed", type=int, default=None)

    put = sub.add_parser("record", help="record one verdict")
    put.add_argument("--matchup", required=True, help="path to the brief, or its name")
    put.add_argument("--verdict", required=True, help="1 | 2 | draw")
    put.add_argument("--reason", default=None)

    sub.add_parser("status", help="vote counts per judge pool")
    sub.add_parser("refresh", help="rewrite unjudged briefs from current output")
    sub.add_parser("agreement", help="human-vs-panel agreement on shared cells")

    focus_parser = sub.add_parser("focus", help="closed-set standings (2+ entrants)")
    focus_parser.add_argument("--entrants", default="ours,trafilatura,wikiextractor-v2")
    focus_parser.add_argument("--pool", default=None, help="human (default) or llm")

    duel_parser = sub.add_parser("duel", help="one pair's record, split by page shape")
    duel_parser.add_argument("--a", default="ours")
    duel_parser.add_argument("--b", default="trafilatura")
    duel_parser.add_argument("--pool", default=None, help="human (default) or llm")

    left = sub.add_parser("pending", help="briefs with no recorded vote yet")
    left.add_argument("--limit", type=int, default=None)

    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.count, arguments.seed)
    elif arguments.command == "record":
        record(arguments.matchup, arguments.verdict, arguments.reason)
    elif arguments.command == "pending":
        pending(arguments.limit, echo=True)
    elif arguments.command == "refresh":
        refresh()
    elif arguments.command == "agreement":
        agreement()
    elif arguments.command == "duel":
        duel(arguments.a, arguments.b, arguments.pool)
    elif arguments.command == "focus":
        focus(arguments.entrants, arguments.pool)
    else:
        status()


if __name__ == "__main__":
    main()
