# Location: how much can resolving where a question is about do for retrieval?

*This file: one experiment, the source of truth for its findings and decisions. Cross-experiment summary and ratings: [overview](../README.md).*

## In short (2026-10-07)

**Insights**

1. **A region leaves few candidates for castles, many for lakes.** Within 10 km of the gold page, the median number of other pages of the same category is 1 for castles and castle ruins (5 or fewer for 84-95% of them), but 12 for lakes and 9 for mountains, with long tails (90th percentile ~100 within 10 km). Finnish and Swedish pages are the densest (median 63 and 44 within 10 km). So resolving the area could nearly settle a castle question but not a lake question. *Evidence: `region_density.py`, 702 gold pages of dev hard questions on balanced pages. 10 km is an optimistic resolution: "western Bohemia" is far wider, and at 25 km the share with 5 or fewer candidates drops from 63% to 36%.*
2. **A location filter undoes BM25's language locking, and more.** With an oracle filter (built from the gold page), translated hard questions go from 2% hit@10 to **45%** when only pages in the gold page's Wikipedia language edition (about its country) are searched, and to **70%** within 25 km; translated easy questions 45% -> 84% / 86%. Within the same language, 25 km lifts hard questions 71% -> 89%; the language edition changes nothing there. With the language-edition filter the anchor query adds nothing: the locking is gone. *Evidence: `oracle_filter.py`, stratified 929 hard + 1,678 easy, 95% intervals. Upper bound: a real resolver must infer the country or region from the question, and the language edition is the easiest level to infer.*

**Next:** a realistic resolver: infer the country / language edition and the region from the question (place names like "Västra Götaland", "in Aragón"; the pipeline's geo resolver), and measure how much of the oracle's gain it keeps. The language edition is the cheapest and recovers the most across languages (2% -> 45%).

## Decisions

- **Three formulations, by question type** (discussion 2026-10-07):
  - *Containment* ("in region R", hierarchy country > province > region > place; back off to the finest level the question supports) suits described places ("which castle in western Bohemia"), but cuts at borders, which a buffer around the region can soften.
  - *Radius* ("within N km of X") suits explicit or implied ranges; N depends on the question and on density (castles vs lakes).
  - *Distance score* (the pipeline's `*_geo` reranker: 0.7 text + 0.3 geo) suits recommendation, but does not remove distractors: a query-language page with a high text score can still win.
- **Two facts about the current setup:** every page in this corpus has coordinates (the "score, not filter" choice came from the Slovenian proxy corpus, where only half the pages were located); and the geo score gives an unlocated page 1.0, the same as a page at the requested spot (`src/retrieval/README.md`), so "unknown = neutral" acts as "unknown = closest".
- **Location resolution helps with the language problem, not the within-region problem.** With many lakes in a region, picking the right one needs specific details or interaction (a shortlist, asking back). Ambiguity is normal, not an error.
- **User location is out of scope:** privacy, and tourists usually plan from home; at most for on-site "near me" questions, with consent.

## Parked, and why

| task | why parked |
|---|---|
| cross-border geo questions | only ~47 in dev span two countries ([audit](../../docs/reports/wiki-audit/audit-2026-10-06.md)): too few |
| a realistic location resolver test | needs the oracle upper bound first |

---

## Details

### Candidates per region

`region_density.py`: other pages sharing a category with the gold page, within 10 / 25 km (haversine on page coordinates), for the 702 gold pages of dev hard questions on balanced pages. Metadata only; runs in seconds.

| category | pages | median within 10 km | share with <= 5 | median within 25 km | 90th pct within 25 km |
|---|---|---|---|---|---|
| lake | 168 | 11.5 | 40% | 40.5 | 448 |
| castle | 166 | 1 | 84% | 8 | 28 |
| mountain | 146 | 8.5 | 42% | 45.5 | 159 |
| nature reserve | 56 | 0 | 91% | 3 | 25 |
| cave | 50 | 6.5 | 50% | 12 | 271 |
| castle ruin | 42 | 1 | 95% | 5 | 21 |
| all | 702 | 3 | 63% | 12 | 208 |

By language, the densest are fi (median 63 within 10 km), sv (44), lt (25.5), sk (21); the sparsest el, pt, hr, bg (0-1).

```
uv run python -m experiments.location.region_density
```

### Plan

1. ~~Candidates per region~~ (done).
2. ~~Oracle location filter~~ (done; see *Oracle filter* below).

### Oracle filter

`oracle_filter.py --per-lang 100` (DuckDB BM25, gentle, ~70 min): full question, realistic anchors and both fused (RRF), each searched without a filter, within the gold page's language edition (prefix `svwiki/`; a stand-in for its country, since each place's article is in its country's language, except Belgium, Finland's Swedish pages and Ireland), and within 25 km of the gold page (any kind of place; median 59 pages). hit@10 [95% interval].

| | none | language edition | 25 km |
|---|---|---|---|
| hard, original, full | 71% [68-73] | 71% [68-74] | 89% [87-91] |
| hard, translated, full | 2% [1-3] | 45% [42-49] | 70% [67-73] |
| hard, translated, fused with anchors | 25% [22-28] | 46% [43-49] | 71% [68-74] |
| easy, original, full | 92% [91-93] | 92% [91-93] | 96% [95-97] |
| easy, translated, full | 45% [42-47] | 84% [82-85] | 86% [85-88] |

By page language, hard, translated, full: Swedish 1% -> 46% (edition) -> 70% (25 km); German 2% -> 47% -> 70%; other Latin 2% -> 45% -> 71%; Greek/Bulgarian 0% -> 50% -> 68%. Same language, 25 km: Swedish 40% -> 68%, other Latin 72% -> 90%.
3. **Realistic resolver:** the same with the region the pipeline's geo resolver extracts from the question.
