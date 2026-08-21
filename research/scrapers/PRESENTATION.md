---
marp: true
paginate: true
style: |
  section { font-size: 30px; }
  h1 { color: #1a3d5c; font-size: 54px; }
  h2 { color: #1a3d5c; font-size: 40px; border-bottom: 2px solid #d8e2ea; padding-bottom: .15em; }
  .row { display: flex; gap: 20px; }
  .box { flex: 1; border: 2px solid #c9d6e0; border-radius: 10px; padding: 14px 18px; background: #f7fafc; }
  .box.win { border-color: #2e8b57; background: #f0f9f4; }
  .box.lose { border-color: #c0554d; background: #fdf3f2; }
  .box h3 { margin: 0 0 6px 0; font-size: 26px; }
  .lab { font-size: 20px; color: #5a6b78; }
  .box .lab { display: block; }
  .flow { display: flex; align-items: center; gap: 12px; justify-content: center; margin: 26px 0; }
  .node { border: 2px solid #1a3d5c; border-radius: 10px; padding: 12px 22px; background: #fff; font-weight: 700; }
  .node.out { background: #1a3d5c; color: #fff; }
  .arrow { font-size: 32px; color: #1a3d5c; }
  .bar { height: 30px; border-radius: 4px; display: inline-block; vertical-align: middle; }
  .b-fetch { width: 620px; background: #1a3d5c; }
  .b-ours  { width: 100px; background: #4a90a4; }
  .b-traf  { width:  65px; background: #7aaebd; }
  .b-resi  { width:   5px; background: #a8c8d4; }
  .box h3.tight { margin: 0; font-size: 46px; }
  .big { font-size: 52px; font-weight: 700; color: #1a3d5c; }
  table { font-size: 26px; }
  .note { font-size: 22px; color: #5a6b78; }
  section.cols { display: grid; grid-template-columns: 390px 1fr; column-gap: 34px;
                 grid-template-rows: auto auto 1fr; align-content: start; }
  section.wide { grid-template-columns: 560px 1fr; }
  section.cols h2, section.cols .full { grid-column: 1 / -1; }
  section.cols pre { font-size: 15px !important; line-height: 1.4; margin: 0; }
  section.cols p { margin: 0; }
---

# Scraper + markdown converter

Two stages, A/B tested.

<p class="note">Handout version. The presented deck is <code>deck.html</code> —
same evidence, the reviewer's slide order.</p>

<div class="flow">
  <div class="node">httpx</div>
  <div class="arrow">→</div>
  <div class="node">trafilatura</div>
  <div class="arrow">→</div>
  <div class="node out">markdown</div>
</div>

---

## ① We compared scrapers

Same URL, four clients. Do they return the **same text**?

<div class="row">
<div class="box win"><h3>httpx</h3>reference</div>
<div class="box win"><h3>Scrapy</h3><span class="big">100%</span><span class="lab">byte-identical</span></div>
<div class="box lose"><h3>Scrapling</h3><span class="big">54%</span><span class="lab">byte-identical</span></div>
<div class="box lose"><h3>+ stealth</h3><span class="big">45%</span><span class="lab">byte-identical</span></div>
</div>

<p class="note">Byte-identical markdown. On <em>content</em> — token cosine, i.e.
whether a converter would see the same document — Scrapling is 100% and stealth
98.2%, which is the figure the presented deck shows. Only the browser fetch moves
the content; Scrapling only moves the bytes.</p>

---

## How Scrapling differs

It drops whitespace-only nodes, so citations **fuse into words**:

<div class="row">
<div class="box win"><h3>httpx</h3><code>…engine.[1][2] She was…</code></div>
<div class="box lose"><h3>Scrapling</h3><code>…engine.[1][2]She was…</code></div>
</div>

<br>

<div class="row">
<div class="box"><h3 class="tight">299</h3>fused boundaries</div>
<div class="box"><h3 class="tight">36</h3>pages affected</div>
<div class="box"><h3 class="tight">2</h3>infoboxes destroyed</div>
</div>

---

## ② So we picked the fetcher

<div class="row">
<div class="box lose"><h3>Rejected</h3>Scrapling — changes the text<br>Stealth — changes it more</div>
<div class="box"><h3>Dropped as redundant</h3>Scrapy — <b>100% byte match</b><br><span class="lab">nothing left to test</span></div>
<div class="box win"><h3>Chosen</h3><b>httpx</b><br><span class="lab">what we already ship</span></div>
</div>

<br>

Byte-identical candidates tell you nothing. We stopped testing them.

---

## Before any votes: how different are they?

Word-for-word similarity, every pair. **The fetchers barely differ at all.**

| | avg chars/page | same words as httpx | byte-identical |
|---|---:|---:|---:|
| httpx | 61 454 | — | — |
| Scrapy | 61 454 | 100% | **100%** |
| Scrapling | 61 432 | 100% | 54% |
| + stealth | 61 538 | 98% | 45% |

<p class="note">Same words, different bytes. Cosine alone would have called all four
interchangeable; byte-identity alone would have called them all different. The gap
between the two columns <i>is</i> the citation defect.</p>

---

## The converters split into two families

Same HTML in. **13× spread** in output length, and the similarity matrix has two blocks.

<div class="row">
<div class="box win"><h3>Content models</h3>
<code>ours · trafilatura · resiliparse<br>justext · html-text · goose3</code><br>
13k–70k chars<br><span class="lab">agree with each other 73–97%</span></div>
<div class="box lose"><h3>Document converters</h3>
<code>crawl4ai · scrapling-md<br>markitdown · docling</code><br>
112k–180k chars<br><span class="lab">agree with each other 83–99%</span></div>
</div>

<br>

Across the divide they share **44–71%** of their words — not a ranking difference,
a different document.

<p class="note"><b>markitdown is 98.6% the same text as scrapling-md</b>, the
"keep everything" floor we planted in the field on purpose. That verdict needed no votes.</p>

---

## ③ Then, fetcher fixed, we A/B tested converters

Every converter got **byte-identical HTML**. Blind pairwise votes.

<div class="flow">
  <div class="node">one page</div>
  <div class="arrow">→</div>
  <div class="node">A</div>
  <div class="node">B</div>
  <div class="arrow">→</div>
  <div class="node out">vote</div>
</div>

**9 converters · 100 pages · 161 votes · 2 juries** (human + LLM panel, never mixed)

---

## What we valued

<div class="row">
<div class="box win"><h3>① Data & tables</h3>Did it <b>find</b> the data?<br>Did the table stay a <b>table</b>?</div>
<div class="box"><h3>② Markdown</h3>Headings, lists, tables<br><span class="lab">so nothing else has to convert</span></div>
<div class="box"><h3>③ No junk</h3>No menus, cookie banners,<br>bare URLs, stray characters</div>
</div>

<br>

Not a strict order — a missing table can be outweighed by markdown
that helps across a whole long document.

---

## ④ The leaderboard

| # | converter | Elo | 95% CI |
|---|---|---:|---|
| 1 | readability-lxml | 1652 | 1450–1834 |
| **2** | **trafilatura** | **1612** | **1489–1732** |
| 3 | resiliparse@rendered | 1606 | 1420–1778 |
| 4 | ours@raw-only | 1581 | 1490–1673 |

<span class="note">Human jury. The LLM panel also put trafilatura 2nd. <b>All intervals overlap</b> — the top group is not separated.</span>

---

## Why trafilatura, not the #1

<div class="row">
<div class="box lose"><h3>readability-lxml</h3>emits <b>HTML</b><br>markdown would be <b>ours</b> to build and own</div>
<div class="box win"><h3>trafilatura</h3>emits <b>markdown natively</b><br>statistically tied, one less layer</div>
</div>

<br>

Nothing in the votes separates them. We broke the tie on **ownership**.

---

## Speed, per page

<div>
<span class="bar b-fetch"></span> <b>fetch — 1058 ms</b><br><br>
<span class="bar b-ours"></span> ours — 269 ms<br>
<span class="bar b-traf"></span> trafilatura — 174 ms<br>
<span class="bar b-resi"></span> resiliparse — 9 ms
</div>

<br>

Fetching dominates 4×. **The converter is not a speed decision.**

The one that is: **don't render.** Playwright costs **3178 ms**, and never won a vote.

---

## Our wrapper vs stock trafilatura

<div class="row">
<div class="box"><h3>89 of 97 pages</h3>byte-identical<br><span class="lab">the fallback never fires</span></div>
<div class="box win"><h3>8 pages differ</h3>7 are <b>listing pages</b><br><span class="lab">homepages, news indexes</span></div>
</div>

<br>

<div class="row">
<div class="box lose"><h3>stock — 311 chars</h3>one unrelated radio blurb</div>
<div class="box win"><h3>ours — 7 479 chars</h3>title + the whole linked index</div>
</div>

<span class="note">rtve.es/noticias/ — and on two pages stock emitted <b>more</b> characters, all of it furniture. Length is not coverage.</span>

---
<!-- _class: cols -->

## Example ① — the infobox

![w:300](figures/infobox.png)

<pre>
| Country     | Netherlands             |
| Province    | North Holland           |
| Region      | Metropolitan Region A.  |
| Founded     | c. 1275                 |
| City Hall   | Stopera                 |
| • Body      | Municipal council       |
| • Mayor     | Femke Halsema (GL)      |
| • Land      | 165.76 km2 (64.00 sq mi)|
| • Water     | 53.56 km2 (20.68 sq mi) |
| • Metro     | 2,580.26 km2            |
| Elevation   | −2 m (−6.6 ft)          |
</pre>

<p class="full note">The floating side box becomes a key/value table. 22 of 24 rows well-formed.</p>

---
<!-- _class: cols wide -->

## Example ② — a data table

![w:540](figures/wikitable.png)

<pre>
| Location      | Population    | %     | Date        |
|---|---|---|---|
| World         | 8,232,000,000 | 100%  | 13 Jun 2025 |
| India         | 1,429,404,000 | 17.3% | 1 Jul 2026  |
| China         | 1,404,890,000 | 17.0% | 31 Dec 2025 |
| United States |   341,784,857 |  4.1% | 1 Jul 2025  |
| Indonesia     |   288,315,089 |  3.5% | 31 Dec 2025 |
| Pakistan      |   241,499,431 |  2.9% | 1 Mar 2023  |
| Nigeria       |   223,800,000 |  2.7% | 1 Jul 2023  |
</pre>

<p class="full note">243 rows survive as rows. Plain-text converters flatten this into prose.</p>

---
<!-- _class: cols -->

## Example ③ — not a wiki page

![w:370](figures/pydocs.png)

<pre>
# 3. An Informal Introduction to Python

Comments in Python start with the hash
character, `#`, and extend to the end of
the physical line.

Some examples:

    # this is the first comment
    spam = 1  # and this is the second
    # ... and now a third!

## 3.1. Using Python as a Calculator
</pre>

<p class="full note">docs.python.org — headings, inline code and code blocks all intact.</p>

---

## Where we cut corners ✂️

- **100 pages**, not thousands
- **161 votes over 66 possible pairs** — ~1.5 each.
  No two converters are separated at 95% confidence
- **Never measured our own consistency** — no re-labelling replicate
- **Speed on one machine, single-threaded** — Scrapy's real strength untested
- **crawl4ai, markitdown, docling** — never benchmarked

<br>

Enough to make a defensible choice. Not enough to certify one.

---

## Locked in

<div class="flow">
  <div class="node">httpx<br><span class="lab">no browser</span></div>
  <div class="arrow">→</div>
  <div class="node">trafilatura<br><span class="lab">+ fallback ladder</span></div>
  <div class="arrow">→</div>
  <div class="node out">markdown</div>
</div>

<div class="row">
<div class="box"><h3>1328 ms</h3><span class="lab">per page</span></div>
<div class="box"><h3>native markdown</h3><span class="lab">tables · headings · lists</span></div>
<div class="box"><h3>+95 ms</h3><span class="lab">buys the listing pages</span></div>
</div>
