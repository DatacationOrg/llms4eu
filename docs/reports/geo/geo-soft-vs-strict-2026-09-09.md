# Soft Versus Strict Geography, 2026-09-09

The WP1 gate of the [improvement plan](../../architecture/geo-improvement-plan.md):
does geography stop hurting once it re-ranks instead of filtering? Same 500
questions, `base` variant, qwen embedder and reranker, Azure resolver, as the
[2026-09-08 run](../retrieval/geo-run-2026-09-08.md); the resolver cache was
warm, so exactly the same 72 questions carry a scope (53 radius, 10 country,
6 NUTS-3, 3 NUTS-2). Per-cell tables and sidecars:
[geo-run-2026-09-09.md](../retrieval/geo-run-2026-09-09.md); the corrected
strict cell: [geo-run-2026-09-09-strict.md](../retrieval/geo-run-2026-09-09-strict.md).
Zero judge failures in either run.

## Result

hit@5, 500 questions; "scoped" is the 72 the resolver anchored to a place.
Wins and losses are per question against `qwen_hybrid_rerank` on the scoped 72.

| method | shape | all | scoped (72) | unscoped (428) | wins | losses |
|---|---|---:|---:|---:|---:|---:|
| qwen_hybrid_rerank | text baseline | 0.884 | 0.917 | 0.879 | | |
| qwen_hybrid_rerank_geo (2026-09-08) | hard filter, widen on count, boost | 0.856 | 0.722 | 0.879 | 1 | 15 |
| **qwen_hybrid_rerank_geo** (today) | **soft: over-fetch, fuse, unknown neutral** | **0.884** | **0.917** | 0.879 | 0 | 0 |
| qwen_hybrid_rerank_geo_strict, cached null policy | hard filter as 09-08 (replication) | 0.856 | 0.722 | 0.879 | 1 | 15 |
| qwen_hybrid_rerank_geo_strict, include_null | filter lets unlocated pages through | 0.884 | 0.917 | 0.879 | 0 | 0 |
| qwen_hybrid_geo | soft, no reranker | 0.776 | 0.875 | 0.759 | 0 | 3 |
| qwen_hybrid_agentic_geo | agent on the soft stage | **0.888** | 0.917 | 0.883 | 0 | 0 |
| qwen_hybrid_agentic_tools_geo | agent + geo tools on the soft stage | 0.882 | 0.903 | 0.879 | 0 | 1 |

The two gate checks the plan named:

| check | baseline | soft geo | strict (hard filter) |
|---|---:|---:|---:|
| the 16 scoped questions whose gold pages have no footprint | 16 / 16 | 16 / 16 | 1 / 16 |
| the near-wording questions in the sample (blizu, okolica, near, around) | 7 / 8 | 7 / 8 | 7 / 8 |

**Gate passed.** The soft method matches the text baseline question for
question on the scoped 72 and on the whole set (mrr@10 0.788 against 0.787).
The fifteen losses of the hard filter were coverage losses, not ranking
losses, and neutral-for-unknown removes all of them: the corrected strict cell,
which keeps the filter but lets unlocated pages through, recovers the same
fifteen and also lands exactly on the baseline. On this corpus the null policy
is the whole difference; the filter-versus-score choice itself is invisible
because nothing is left for either to reorder. No gain was expected on
this corpus and none appeared: with fifteen distinct coordinate pairs and every
baseline miss sitting at 0 km from its scope, there is nothing for a distance
score to reorder (plan §2a).

## Reading the cells

- **The replication.** The strict cell of the first run reproduced the 09-08
  numbers to the third decimal because the cached scopes carried
  `include_null: false` from when they were stored and the retriever read the
  policy off the scope. The policy now comes from config at retrieval time
  (`GeoScopedRetriever.include_null`), and the corrected cell was rerun
  separately; that is the fourth row. Reproducing 0.856 exactly is also a useful
  check that the run is deterministic end to end.
- **Strict with `include_null`.** With unlocated pages passing the filter,
  the strict path widened 16 of 72 scopes (the in-scope located count fell
  below five) and used a country scope for 11 questions, yet lost nothing: on
  a corpus where every located page passes every region scope, the filter
  removes no gold page once the unlocated ones are safe. Soft stays the default
  on principle rather than on this measurement: it has no fence to widen, so a
  wrong or over-narrow geocode cannot remove a located gold page either, and
  the plan's relation weights (WP3) have somewhere to plug in.
- **Selectivity gate in action.** The `Geo scope` table shows the soft method
  used level `none` for 14 of the 72 scoped questions: the 10 country scopes and
  4 of the 9 region scopes covered more than 90% of the located pages and were
  skipped. The 53 point scopes were all scored by distance.
- **Soft without a reranker** (`qwen_hybrid_geo`) is not comparable to the
  reranked baseline; its three scoped "losses" are reranker losses. Against the
  09-08 hard-filter version of the same method it gains 0.030 overall and 0.194
  on the scoped set.
- **The agents.** `qwen_hybrid_agentic_geo` is the best cell of the run at
  0.888, four questions above the baseline, all of them unscoped, so the gain is
  the agent's retry loop rather than geography. The tool agent lost one scoped
  question (the Krško data-protection page, whose gold page *is* located): the
  agent spent its budget on a place lookup and never expanded. Both agents ran
  without a single structured-output failure.
- **Speed.** The soft stage costs nothing measurable over the baseline
  (829 ms against 856 ms per query, reranking dominates); the unfiltered
  over-fetch is served by the same candidate pool the reranker already reads.

## What this settles and what it does not

Settled: on a half-located corpus geography must re-rank, never pre-filter, and
unknown footprint must be neutral. This is now the default `*_geo` shape and
the decision is recorded in [decisions.md](../../architecture/decisions.md)
("Geographic Scope").

Not settled: whether geography can *help* here. The eval set was generated
from chunks and holds almost no spatial questions (8 near-wording questions in
this sample, 37 in 3,476), and the located footprints all sit within a few
kilometres of each other. The next measurable steps are the ones the plan
ordered after WP1: widen the footprint layer (WP2, WP4: the 124 recoverable
gazetteer misses in [locate-rejections-2026-09-09.md](locate-rejections-2026-09-09.md)),
return the spatial relation from the resolver so `w` depends on near/in/about
(WP3), and write the spatial probe set that can see a lift (WP6).
