---
name: scraper-arena-judge
description: Run automated blind A/B judging for the scraper arena in research/scrapers, using local Sonnet subagents as an independent rating pool. Use when asked to auto-judge, add LLM votes, or grow the arena's vote count without a human reviewer.
---

# Scraper Arena — automated judging

Adds votes to the **`llm` pool** of the arena in `research/scrapers`. Never touches
the `human` pool, and never consumes matchups the human reviewer has not seen: each
pool samples against only its own vote history.

Judging must run as **local `Agent` subagent calls with `model: sonnet`** — never a
paid API call, never a workflow. Keep each call small and targeted: one brief in,
two lines out.

## The loop

**1. Prepare a batch.** From the repo root:

```bash
uv run python -m research.scrapers.judge prepare --count 12
```

This prints the batch directory and one path per brief. Each brief is a
self-contained markdown file holding the rubric, the page metadata, a neutral
tag-strip reference rendering, and the two outputs labelled `OUTPUT 1` / `OUTPUT 2`
in randomised order. Entrant identities go to a separate `keys/` directory that
judges are never pointed at.

Byte-identical pairs are auto-drawn during preparation and never become briefs.

**2. Dispatch one subagent per brief**, all in a single message so they run
concurrently. Keep the prompt to essentially this, substituting the path:

> Read `<brief path>`. It contains a rubric and two candidate text extractions of a
> web page. Follow the rubric and decide which is better.
>
> Then record it by running exactly:
> `uv run python -m research.scrapers.judge record --matchup <brief path> --verdict <1|2|draw> --reason "<one sentence>"`
>
> Reply with only the two lines `VERDICT: ...` and `REASON: ...`.

Do not paste brief contents into the subagent prompt — that defeats the point of
keeping the orchestrator's context small. Let the subagent read the file.

**3. Verify, don't assume.** After a batch:

```bash
uv run python -m research.scrapers.judge status
```

Confirm the `llm` pool grew by the number of briefs dispatched. A subagent that
mangled the command records nothing and fails silently, so this check is required.
Re-dispatch any brief whose vote is missing.

Resuming later? `judge pending --limit N` lists briefs on disk with no recorded
vote, so the database decides what is left rather than your memory of which batch
you reached.

**Before dispatching, if any extractor adapter has changed since the briefs were
written**, run:

```bash
just arena-judge-refresh
```

Briefs are snapshots of output. Fix an adapter and a staged brief still shows the old
string, so a judge ends up ruling on output that no longer exists — this happened
once, with a judge describing a wikiextractor output as "just a redirect stub" long
after the redirect handling was fixed. `refresh` rewrites unjudged briefs only; votes
already cast on changed output need `store.invalidate_votes`, not a rewritten brief.

**4. Repeat.** Batches of 10–15 are a good size. Regenerate the report when done:

```bash
just arena-report
```

## Rules that matter

- **Blind.** Never tell a judge which tool produced which output, and never reveal
  entrant names in the prompt. The rubric already warns against position bias.
- **Draws are real verdicts.** The rubric instructs judges to draw when outputs are
  close or both bad. A pool with almost no draws is a sign the rubric was ignored.
- **Pools stay separate in the report.** Present the human and LLM leaderboards
  side by side with their rank correlation. Do not average them, and do not present
  an LLM rating as if a human produced it.
- **One vote per (page, pair) per pool.** `record` refuses duplicates.

## Judging criteria (encoded in the brief's rubric)

Priority order, rewritten in session 4 to match the human reviewer's stated
hierarchy: **data and tables**, then **markdown formatting**, then **noise**, then
token economy.

Criterion 1 asks two questions in order — did one output *find* the data the other
lost (worst failure on the list), and if both found it, did one preserve its
*structure*? A table melted into a paragraph has lost real information even when
every character survives.

The list is explicitly **not** a lexicographic tiebreak. The rubric carries the
reviewer's own worked example: an output that missed one table but was far more
usable markdown across a long document was correctly a **draw**. Judges are told to
weigh how much of the document a difference touches, not just which criterion it
lands on.

The rubric still treats **both** failure directions as serious. Under-extraction
(dropping real content) is obvious. Over-extraction matters just as much: an
entrant that dumps the whole page trivially has perfect coverage, and if the rubric
ranked coverage first without saying so, the no-boilerplate-removal floor
(`html-text`) would win every matchup by default — inverting the whole result. The
rubric states outright that "kept more text" is not "better".

### Calibration anchors carried in the rubric

Both are measured from this corpus, and both correct a real defect in the v1 pool:

- **Position bias.** The v1 panel picked the first-shown output in 55 of 87
  decisive verdicts (63%) while the human split 20/20. Brief order is randomised,
  so this cost precision rather than biasing any entrant — but the rubric now
  states the number and tells judges to re-read OUTPUT 2 before committing.
- **Draw rate.** The human draws on 23% of matchups and calls both-bad on a further
  11%. The v1 panel drew on 15% and *never* said both-bad, because the brief only
  ever offered `1 | 2 | draw`. `both_bad` was already a valid verdict in
  `VERDICTS`; it is now offered in the brief, with guidance to prefer it over a
  draw when neither output is usable.

### The structural inventory

Each brief carries a whole-output inventory table — characters, ATX headings,
setext underlines, pipe-table blocks and rows, list items, markdown links, bare
URLs, citation markers, backslash escapes — plus the `<table>`/`<tr>` counts of the
page's own HTML as ground truth.

This exists because criterion 1 was otherwise unjudgeable on long pages: at the old
10k budget a judge saw 5.7% of a 178k-character output, so a missing table almost
always fell inside an elided window. The inventory is computed over the *whole*
string, never the excerpt.

It comes with a caveat the rubric repeats: zero pipe-table rows means no table
*markup*, which does not prove the data is gone — a plain-text tool may have kept
every value as running text. That is the milder criterion-1b failure, and judges
are told to check the excerpt before deciding which one they are looking at.

## Rubric versions

The rubric is an instrument. Changing it changes what the pool measures, so votes
are not blended across versions.

- **v1** — coverage-first rubric, 6k/10k budgets, no inventory, no `both_bad`.
  102 votes, archived via `store.invalidate_votes` to the JSONL mirror on
  2026-08-19 with the reason recorded. Recoverable; `arena.db.bak-rubric-v1` is a
  full snapshot from just before.
- **v2** — current. Data/tables-first, 8k/30k budgets, structural inventory,
  `both_bad` offered, calibration anchors stated.

The 10 byte-identical auto-draws were kept across the change: they are decided by
string equality, not by any rubric.

Because refreshed briefs preserve their matchups, re-judging them under v2 gives
the same (page, pair) cells under both rubrics — so the archived v1 votes are a
usable measurement of what the rubric change did, not just dead weight.
