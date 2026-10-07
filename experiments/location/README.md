# Location: how much can resolving where a question is about do for retrieval?

*This file: one experiment, the source of truth for its findings and decisions. Cross-experiment summary and ratings: [overview](../README.md).*

## In short (2026-10-07)

**Insights**

1. **A region leaves few candidates for castles, many for lakes.** Within 10 km of the gold page, the median number of other pages of the same category is 1 for castles and castle ruins (5 or fewer for 84-95% of them), but 12 for lakes and 9 for mountains, with long tails (90th percentile ~100 within 10 km). Finnish and Swedish pages are the densest (median 63 and 44 within 10 km). So resolving the area could nearly settle a castle question but not a lake question. *Evidence: `region_density.py`, 702 gold pages of dev hard questions on balanced pages. 10 km is an optimistic resolution: "western Bohemia" is far wider, and at 25 km the share with 5 or fewer candidates drops from 63% to 36%.*
2. **Location filters are language-independent, so they could undo BM25's language locking** (inference, untested). A translated question returns 99% pages in the query's language ([language anchors](../language_anchors/README.md)); restricting to the right area removes those distractors whatever their language. It does not solve picking the right place within a dense area (insight 1).

**Next:** an oracle location filter on the translated questions (only pages within 25 km of the gold page, or in its country): the upper bound of what location can recover across languages. Then the same with the region the pipeline's resolver extracts from the question.

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
2. **Oracle location filter** on translated questions: BM25 (DuckDB engine, [`../language_anchors/fts.py`](../language_anchors/fts.py)) restricted to pages within 25 km of the gold page, and to its country; compare with the unfiltered translated question and the same-language question.
3. **Realistic resolver:** the same with the region the pipeline's geo resolver extracts from the question.
