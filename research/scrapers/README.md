# Scraper Arena

Blind pairwise benchmark of HTML→text extractors, rated by ELO / Bradley-Terry.

## Why

`src/scraping/extract_markdown.py` is the single content-extraction path for this
whole project — a BeautifulSoup pre-clean feeding
`trafilatura.extract(..., favor_precision=True)`, wrapped in the
prose→listing→document→render ladder in `src/scraping/page_extract.py`. It
determines the quality of the chunk RAG corpus, the OKF bundles and the eval set
in `data/db/pages.db`, and it had never been compared against anything. There is
no automatic metric for extraction quality that does not need gold annotations,
so the metric here is human judgement over blind pairwise comparisons.

Three questions:

1. **Does our wrapper beat stock trafilatura, or hurt it?** The pre-clean deletes
   any element whose `id`/`class`/`role`/`aria-label` contains `accessibility` —
   plausibly destructive, never verified. `ours` vs `trafilatura` isolates exactly
   this.
2. **Is trafilatura the right engine?** Resiliparse is ~8× faster and much
   stronger on tables ([WCXB](https://arxiv.org/abs/2605.21097): WikiTQ 11.9 vs
   3.7). DCLM and Dolma chose it; FineWeb and HPLT chose trafilatura.
3. **Does it hold up outside English?** The only genuinely multilingual study
   ([SIGIR 2025](https://dl.acm.org/doi/10.1145/3726302.3730234)) puts
   trafilatura's macro-F1 at ~0.77 on non-English. Every other published
   comparison is English-dominant. Half this corpus is non-English for that
   reason.

## Running

### Entrant libraries

The entrants are the libraries under test, so they are **not** project dependencies
and are deliberately absent from `pyproject.toml` — only `trafilatura` (the incumbent,
used in production) and `docling` are declared there. Everything else has to be
installed before `just arena-extract` will produce a full board:

```bash
uv pip install resiliparse justext readability-lxml markdownify goose3 \
               html-text scrapling markitdown crawl4ai scrapy
```

`wikiextractor-v2` is not on PyPI in a usable form and only imports under Python
3.10; `just arena-wiki` runs it out of a vendored checkout in
`.local/scraper-arena/vendor/` (see the wikiextractor note below).

A missing library is safe but not silent. Adapters import lazily, `run_extractors.py`
records the `ImportError` as `status: "error"`, and `sampler.py` refuses to serve any
run whose status is not `ok` — so an absent entrant shows up as a coverage gap in the
report, never as an empty output that loses votes it should not have played for. It
does mean the recorded results in `.local/scraper-arena/` cannot be regenerated
without reinstalling the list above.

```bash
just arena-snapshot   # fetch the corpus once — network, ~5 min
just arena-extract    # every entrant × every page, timed — ~25 min
just arena-wiki       # the Python 3.10 wikiextractor pass
just arena-fetchers   # fetch-layer comparison (optional)
just arena            # serve at http://127.0.0.1:8090
just arena-report     # dump the results page standalone
just arena-reset-votes  # clear votes, keep snapshots and runs
```

Automated judging adds a second, independent vote pool without a human:

```bash
just arena-judge-prepare 12   # write 12 blinded briefs
just arena-judge-status       # vote counts per pool
just arena-judge-refresh      # rewrite unjudged briefs from current output
just arena-agreement          # human-vs-panel agreement on shared comparisons
just arena-duel               # one pair's record, split by page shape
just arena-wiki3              # ours / trafilatura / wikiextractor-v2 standings
just arena-focus a,b,c        # standings for any closed set of 2+ entrants
just arena-reset-llm-votes    # clear the automated pool only
```

Dispatch one Sonnet subagent per brief — the `scraper-arena-judge` skill has the
loop. Requires network, so it is a before-departure activity, not an in-flight one.

Steps 1–4 need network. `just arena` and the report do not: **the arena is
designed to run with networking fully disabled.** Nothing is fetched from a CDN,
and the source-page reference is a live iframe of the pinned URL, falling back to
the saved DOM off local disk where a site refuses framing.

Judging: `←` A better, `→` B better, `↓` draw, `X` both bad, `S` skip, `L`
leaderboard. Vote bars are duplicated at top and bottom so neither requires
scrolling.

## Protocol

**Fetch once, extract many.** Each URL is fetched exactly once and snapshotted;
every entrant runs against those identical bytes. Re-fetching per entrant gives
each a different document (A/B tests, rotating banners, edits) and any quality
difference would be confounded by snapshot luck. Variants per page:
`raw.html` (httpx), `rendered.html` (Playwright), `wikitext.txt` (wiki pages).

**An entrant is a (input variant, extractor) pair**, which makes the
raw-vs-rendered axis votable instead of a hidden constant. `trafilatura` and
`trafilatura@rendered` are separate entrants; on a static page they produce
identical output and the auto-draw rule hides the matchup, so clicks only land
where rendering changed something.

**Blinding.** Columns are A/B, sides randomised per matchup, identity revealed
only after the vote commits. `entrant_a` is stored as the *left* column, so
position bias stays measurable — it is reported.

**Judged as raw text** in a `<pre>`, because that string is what enters a corpus:
stray `](#)` links, duplicated nav and broken tables must be visible, not styled
away. A markdown-render toggle exists for convenience.

**Active sampling.** `weight = p(1-p)/sqrt(1+n_ij)`. `p(1-p)` peaks for
evenly-matched pairs and decays as one dominates, so first-vs-last stops being
sampled once settled; the `n_ij` term spreads coverage. Weighted draw (not
argmax) plus a 10% uniform floor so early noise cannot freeze the order. Pages
are chosen round-robin across buckets, keeping per-bucket data balanced at any
stopping point.

**Timing.** One discarded warmup, then median of 3, single-threaded, one machine,
one run. `input_ms` (raw fetch vs Playwright render) is charged to the entrant,
because needing a browser is a real cost. Reported alongside `extract_ms` and
`total_ms`; never collapsed into a single quality-per-ms score (see
`experiments/indexing/README.md` on composite scores).

**Ratings.** Online ELO drives the live board. The report leads with
Bradley-Terry MLE + bootstrap CIs, because sequential ELO is order-dependent and
that matters at 100–2000 votes. BT is regularized with a small symmetric prior
(`PRIOR_GAMES`); without it an undefeated entrant has an unbounded MLE and a
single 1-0 record rated 2817. Votes are append-only (SQLite + JSONL mirror), so
every rating recomputes from scratch.

## Corpus

100 pages, four buckets of 25: `en-wiki`, `en-other`, `eu-wiki`, `eu-other`.
Non-English spans **nl, de, sl, hu, es, pl**. Page *shape* is varied deliberately
— biography, infobox, table-heavy, list, glossary, science-with-math, news,
gov, docs, blog, recipe, forum, SPA, PDF-link — because extractors diverge on
shape far more than on topic; 25 lookalike prose articles would produce 25 draws.
`eu-other` includes the project's real target domains from `data/brestanica.json`.

`corpus/sites.yaml` holds ~140 candidates; `snapshot.py` validates them and keeps
the first 25 per bucket that fetch cleanly, dropping 403s, robots-disallowed URLs
and dead links. Pool order is curation order, so the shape spread survives
selection. `robots.txt` is honoured throughout and fetches are throttled
per-domain via `DomainThrottle` from `src/scraping/page_fetch.py`.

## Entrants

| Entrant | Input | Notes |
|---|---|---|
| `ours` | raw→rendered | our ladder, driven from cached HTML |
| `ours@raw-only` | raw | same, render fallback disabled |
| `trafilatura` | raw | stock defaults, Markdown output |
| `trafilatura@rendered` | rendered | |
| `resiliparse` | raw | `main_content=True` |
| `resiliparse@rendered` | rendered | |
| `justext` | raw | per-page-language stoplist |
| `readability-lxml` | raw | summary HTML → markdownify |
| `goose3` | raw | high precision, low recall |
| `html-text` | raw | **floor:** no boilerplate removal |
| `scrapling-md` | raw | **floor:** markdownify over `<body>` |
| `wikiextractor-v2` | wikitext | wiki slice only — see caveats |

Stock trafilatura is set to Markdown output on purpose, so `ours` vs
`trafilatura` measures our wrapper rather than a Markdown-vs-plaintext
presentation difference.

**Scrapy and Scrapling's fetchers are not voting entrants.** They retrieve HTML
and have no main-content model, so `scrapy + html-text` would emit output
byte-identical to `html-text` and voting on it would burn clicks for zero
information. `fetch_layer.py` measures them instead — success rate, anti-bot
blocks, charset handling, wall time — and hashes the normalised text of every
response so *divergence is measured rather than assumed*. Pages where the
fetchers genuinely disagree are worth promoting into the arena.

## Environment quirks

Three things here are non-obvious and will bite anyone reproducing this.

**1. Extractor libraries are layered with `uv run --with`, not added to
`pyproject.toml`.** `scrapling` pins Playwright while the project env already
pins `chromadb`, `sentence-transformers` and `docling`; a naive `uv sync` risks
failing to resolve or silently downgrading something the RAG pipeline needs. The
overlay leaves `uv.lock` and the project env untouched. See `arena-env` in the
justfile.

**2. Chromium needs a vendored `libasound.so.2`.** This WSL2 box lacks it and
installing it system-wide needs sudo, so it is extracted into
`.local/scraper-arena/lib/` and put on the loader path by
`research/scrapers/browser.py` before any browser launches. To recreate:

```bash
cd $(mktemp -d) && apt-get download libasound2t64 && dpkg-deb -x libasound2t64_*.deb x
cp -P x/usr/lib/x86_64-linux-gnu/libasound.so* /path/to/.local/scraper-arena/lib/
```

**3. Wikiextractor-V2 runs on its own Python 3.10.** It uses inline regex global
flags (`(?i)` mid-pattern) that Python 3.11+ rejects, so it does not import on
the project's 3.13. It runs unmodified under an older interpreter rather than
being patched, since patching a third-party tool would change what is measured.
Clone it (AGPL-3.0, deliberately not vendored into this repo):

```bash
git clone --depth 1 https://github.com/langtech-bsc/Wikiextractor-V2.git \
  .local/scraper-arena/vendor/Wikiextractor-V2
```

Further Wikiextractor caveats, all stated in the report: **template expansion —
its actual differentiator — is unavailable here**, because template definitions
live in a dump header and this drives it from single-page `action=raw` wikitext;
its `--generator` mode is broken upstream (`collect_pages` yields 5-tuples, its
own caller unpacks 4); it is AGPL-3.0 against Apache/MIT/BSD for everything else;
and its timings come from a different interpreter, so they are indicative only.
A faithful Wikipedia comparison would need the Wikimedia Enterprise HTML dumps.

## Two vote pools

Votes carry a `judge` column: `human` or `llm`. The pools are **never merged into
one rating.**

* They measure different things. An LLM judge reading a text brief and a human
  reading the rendered page are different instruments; a blended number describes
  neither.
* Scale would swamp judgement. Automated votes are cheap and human votes are not,
  so merging lets a few hundred of the former bury a few dozen of the latter.
* Each pool samples against **only its own vote history**, so automated judging
  never consumes matchups the human reviewer has not seen. This is why
  `store.load_votes()` defaults to the human pool rather than to everything — an
  accidental unscoped call that silently blended pools would be very hard to spot
  by eye.

### Calibration: making the pools comparable

The separation above has a cost that took a while to surface. Because cells are never
repeated within a pool and each pool samples against only its own history, the two
pools **drift into disjoint corners of the corpus**. At 17 human and 195 automated
votes they had judged exactly *one* comparison in common — so there was no evidence
either way about whether the automated ranking reflected the reviewer's judgement.
An unvalidated panel leaderboard that looks confident is worse than no leaderboard.

Calibration mode fixes that by deliberately re-serving cells the *other* pool has
already judged:

```bash
just arena            # tick "calibration mode" in the header
just arena-agreement  # the resulting numbers
```

Design points that matter for the number to mean anything:

* **Still blind.** The panel's verdict is never sent to the client. Agreement is
  computed afterwards from the two independently stored votes.
* **Stratified by ELO gap**, not information-weighted. Agreement measured only on
  near-ties understates it and only on lopsided pairs flatters it, so the sample is
  spread across gap bands, preferring whichever band is least represented so far.
* **Side randomised independently** of the order the other pool saw. Reusing their
  order would let a position bias shared by both instruments inflate agreement;
  inverting it would deflate it. The report splits agreement by whether the orders
  happened to match, so a shared bias is visible rather than baked in.
* **Three rates, because they answer different questions.** `exact` (same verdict,
  same side), `equivalent` (`draw` and `both_bad` collapsed, as `rating.py` scores
  them), and `decisive` (both named a winner). Reporting only the first would
  penalise a vocabulary difference; only the last would hide the pools' differing
  draw habits.
* **Cohen's kappa alongside**, because raw agreement on an unbalanced task is
  inflated by chance. Kappa is `None` rather than 1.0 when every shared cell drew the
  same label — that is a degenerate sample, not perfect agreement.

Calibration votes are ordinary human votes: they count toward the human leaderboard
too, so the clicks are not spent solely on measuring the panel.

One measured asymmetry is worth knowing before reading any automated result: the
panel has **never once used `both_bad`** (0 of 195, with the option available in its
verdict set), against 2 of 17 for the human reviewer. An instrument that always names
a winner manufactures a signal on pages where both outputs are unusable — precisely
the pages where the floor entrants should be exposed.

### Focus mode: a closed arena over a named set

Same idea at any size: fix the entrant set, sample only within it. Two entrants is a
duel; three or more is a round-robin. `just arena-focus a,b,c` reads out any set,
`arena-duel` and `arena-wiki3` are shorthands for the two that matter here.

The load-bearing rule is that pages are restricted to those where **every** named
entrant produced output. That is what makes the resulting ratings comparable to one
another — a Bradley-Terry fit in which one entrant played an easier subset of pages is
not a ranking, it is an artifact. It is also the mechanism that lets
`wikiextractor-v2` be rated against the others at all: it consumes wikitext, so its
shared population with `ours` and `trafilatura` is the **47 wiki pages** all three
handle, and the standings describe exactly those 47 and nothing else. Contrast the main
leaderboard, which cannot include it for precisely this reason and keeps it in a
separate wiki slice.

`subset_standings` and the sampler's progress counter apply the same restriction, and
must keep doing so. When only the sampler was scoped loosely, a stray explore-mode vote
on a non-wiki page made the header read `ours vs trafilatura: 1 judged, 47 of 47
remaining` and pushed the round-robin off that pair on the strength of a vote no
standings table counts.

**The stratification axis is chosen from the population, not fixed.** Splitting
`ours vs trafilatura` by page shape is exactly right; applying that same split to the
three-way set does nothing, because those 47 wiki articles contain *zero* index-like
pages. So `_focus_strata` picks page shape when both classes are present, else English
vs other language (25 English against 22 across de/es/hu/nl/sl — which is the SIGIR
multilingual question), else a single bucket. A fixed axis would have silently balanced
nothing.

### Duel mode: settling one question instead of building a board

The information-weighted sampler is right for producing a leaderboard and wrong for
settling a specific claim. Spread over 55 pairs, a hundred clicks leave every
individual matchup with a handful of votes — enough to order the field, not enough to
resolve `ours vs trafilatura`, which is the pair with a code change attached to it.

Duel mode fixes the pair and alternates the *page shape*:

```bash
just arena            # pick "duel: ours vs trafilatura" in the mode selector
just arena-duel       # the resulting record, split by shape
just arena-duel resiliparse trafilatura   # any other pair
```

Shape, not bucket, is the stratification axis here, and that is the whole point.
Measured over the corpus with a cold dedup cache, the single flag that separates our
wrapper from stock trafilatura is `favor_precision=True`: it costs >20% of the text on
14 of 100 pages, and takes six pages to near-zero (Hacker News 3,908 → 0,
tagesschau.de 7,475 → 46, bbc.com/news 7,327 → 168). Every one is a link index or news
front page — `favor_precision` keeps only text it is confident is article prose, and
such a page contains none. On article pages our output is a little *longer* than
stock's (median +7.3%, mostly from `include_images=True`).

So the two populations move in opposite directions, and a single win rate averages the
effect to nothing. Stratifying by `is_index_like(shape)` and reporting the halves
separately is what makes the mechanism visible rather than inferred.

Focus mode of any size is only **semi-blind**: naming the set tells the reviewer who is
involved, though not which column is which. The UI also names the pair currently on
screen, deliberately built from the *canonical* (sorted) pair rather than the display
order, so it cannot leak the side — there is a test pinning that. These votes are
weaker evidence than the fully blind pool's, so they are tagged `reason='focus'`
(historically `'duel'`), and both `head_to_head_by_shape` and `subset_standings` report
how many of a set's votes came from focus mode. Do not quietly pool the two when
reporting a result.

One mode selector caveat worth remembering: a `<select>` resets to its first option on
page load, so "refresh and pick duel" silently put the reviewer back in explore mode
and nine votes went through the wrong sampler. The mode now persists in
`localStorage` and is restored before the first matchup fetch.

Neither the report nor the arena leaderboard ever leads with an **empty** pool.
Rendering the human pool before the human has voted puts every entrant at the 1500
start value, which reads as "all extractors are equal" rather than "no votes yet" —
so an empty primary pool falls through to whichever pool has evidence, labelled as
such. The pools are still never blended.

## When an adapter changes after judging starts

A vote compares two specific strings. Fix an extractor adapter and any vote cast on
its old output becomes evidence about output that no longer exists, so it has to go:
`store.invalidate_votes(ids, reason)` deletes the rows and appends an `invalidated`
event to the JSONL mirror, keeping the audit trail append-only. Briefs already
staged on disk are the opposite case — they are unjudged, so
`just arena-judge-refresh` rewrites them from current output. A judged brief is
never rewritten: it is the record of what the judge actually saw.

The report shows both leaderboards with their rank correlation, which is itself a
result: how far an automated panel can substitute for a human on this task.

## Traps

Five things here are counter-intuitive enough to have caused real bugs during the
build. All are fixed; they are recorded because they will bite a re-implementation.

**1. `deduplicate=True` makes trafilatura stateful across calls.** It drives a
process-global LRU cache, so calling it repeatedly on the same HTML returns *less*
text each time once a segment has been seen more than twice. Measuring with N
repeats and storing the last result silently gutted our own output — 17,028 chars
to 618 on one page — and leaked between entrants that ran back-to-back.
`reset_extractor_state()` clears it before every call, and the runner stores the
first timed result. Any entrant with hidden cross-call state must be reset there
too.

**2. Anti-bot pages are valid HTML.** A Cloudflare challenge or a LiteSpeed 403
parses fine and extracts to ~190 plausible characters. Both fetch paths must screen
for them; screening only the raw path let three interstitials through and made
rendering look valuable exactly where it had failed.

**3. Don't time the politeness delay.** `DomainThrottle.wait()` must be called
before the timer starts, or `fetch_ms` measures 1.5 s of sleep and multi-candidate
domains read 20× slower than single-candidate ones.

**4. Load average lies on WSL2.** It read 6.7 with nothing running, because I/O wait
inflates it and it decays slowly. The timing pass gates on sampled `/proc/stat` CPU
utilisation instead, and re-checks periodically rather than only at startup — a job
starting mid-run would otherwise quietly contaminate the remaining pages.

**5. An empty-output count split by language will mislead you.** jusText returns
nothing on 18 of 100 pages, 12 of them non-English, which reads as a multilingual
weakness and lines up neatly with the published SIGIR 2025 finding — a very easy
wrong conclusion to publish. The real variable is page *shape*: 64% empty on
index-like pages against 5% on article pages, with English listing pages (Hacker
News, BBC, gov.uk) failing identically. Before attributing anything to language,
control for shape; the `eu-other` bucket holds 56% index-like pages against
`en-other`'s 32%. Verified separately: every stoplist loads (Slovenian 2160 words,
Hungarian 3253, English 453) and every page's declared language is correct.

Two smaller ones. jusText re-sniffs the encoding from the document's own
`<meta charset>` even when handed already-decoded UTF-8 bytes, so a page declaring
iso-8859-15 gets mojibake and then fails stoplist matching; pass `encoding="utf-8"`
explicitly. No page in this corpus triggers it, but a future one would, silently.
And `action=raw` returns redirect stubs rather than following them,
and the `#REDIRECT` keyword is localised (`#ÁTIRÁNYÍTÁS`, `#PREUSMERITEV`), so match
the link target rather than the keyword.

## Layout

```
config.yaml        entrants, paths, port, ELO and sampling parameters
corpus/sites.yaml  the candidate pool, with bucket/language/shape labels
store.py           SQLite schema and accessors
snapshot.py        fetch-once: raw + rendered + wikitext
extractors/        one adapter per entrant, plus the registry
run_extractors.py  every entrant × every page, timed
wiki_pass.py       the Python 3.10 wikiextractor pass
fetch_layer.py     ours vs Scrapy vs Scrapling
rating.py          online ELO + Bradley-Terry + bootstrap CIs
sampler.py         information-weighted matchup selection
judge.py           blinded briefs + verdict recording for the automated pool
similarity.py      word-cosine and byte-exact matrices for both layers
arena/             FastAPI app + vendored CSS/JS (no CDN)
arena/marks.py     whole-document diff marking for the review panes
stats.py           every aggregate over votes/runs/snapshots -- pure, no HTML
report.py          renders those aggregates; live at /report
methods.html       the report's methodology section, as prose
deck.src.html      the presentation source; `just arena-deck` builds deck.html
deck_data.json     similarity output; build_deck.py checks the deck against it
```

Bulky regenerable artifacts live in `.local/scraper-arena/` (gitignored):
`arena.db`, `votes.jsonl`, `snapshots/`, `vendor/`, `lib/`, `report.html`, and
`judge/` (briefs and their answer keys). Only code, `config.yaml` and
`corpus/sites.yaml` are committed — no page content.

The dispatch loop for automated judging lives outside this folder, in the
`scraper-arena-judge` skill, because it is a procedure for an agent rather than a
script.

## Reading the results

Read the leaderboard **next to the coverage table**. An entrant that wins on the
pages it handles but returns nothing on a third of them is not a winner, and BT
ratings say nothing about the pages an entrant skipped. Where confidence
intervals overlap, the entrants are not distinguishable at that vote count —
which is the honest reading, and the reason the report leads with intervals
rather than ranks.
