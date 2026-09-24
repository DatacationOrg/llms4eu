# Retrieval Comparison 2026-09-09

- Command: `experiments/indexing/compare_qwen_modes.py --methods qwen_hybrid_rerank,qwen_hybrid_rerank_geo_strict --variants base --span-target gold --limit 500 --log-every 100 --output docs/reports/retrieval/geo-run-2026-09-09-strict.md`
- Database: data/db/pages.db | design: **shared** | span target: **gold**
- Variants: base
- Methods (2): qwen_hybrid_rerank, qwen_hybrid_rerank_geo_strict
- Questions: 500 | Category: all | Warmup: 0
- Output: docs/reports/retrieval/geo-run-2026-09-09-strict.md | Checkpoint: docs/reports/retrieval/geo-run-2026-09-09-strict.md.checkpoint.json
- Updated: 2026-09-09T21:42:10 | Status: complete

## Progress

- base: qwen_hybrid_rerank 500/500, qwen_hybrid_rerank_geo_strict 500/500

## Variant `base`

Evaluating 500 questions

Overall
| method                        | hit@1 | hit@5 | hit@10 | recall@10 | mrr@10 |
|-------------------------------|-------|-------|--------|-----------|--------|
| qwen_hybrid_rerank            | **0.718** | **0.884** | **0.900** | **0.900** | **0.787** |
| qwen_hybrid_rerank_geo_strict | 0.716 | **0.884** | **0.900** | **0.900** | 0.786  |

Speed

| method                        | seconds | ms/query | queries/query | queries | chunk expansions |
|-------------------------------|---------|----------|---------------|---------|------------------|
| qwen_hybrid_rerank            | 719.99  | 1440.0   | 1.00          | 500     | 0                |
| qwen_hybrid_rerank_geo_strict | 784.22  | 1568.4   | 1.00          | 500     | 0                |

hit@5 by category

| method                        | crosslingual | direct_long | direct_short | vague_long | vague_short |
|-------------------------------|--------------|-------------|--------------|------------|-------------|
| qwen_hybrid_rerank            | **0.891**    | **0.940**   | **0.887**    | **0.981**  | **0.741**   |
| qwen_hybrid_rerank_geo_strict | **0.891**    | **0.940**   | **0.887**    | **0.981**  | **0.741**   |

Geo scope
| method                        | questions | scoped | scoped % | widened | levels used                                      |
|-------------------------------|-----------|--------|----------|---------|--------------------------------------------------|
| qwen_hybrid_rerank_geo_strict | 500       | 72     | 14%      | 16      | country=11, none=12, nuts2=1, nuts3=3, radius=45 |

`scoped` questions resolved to a place; `widened` needed a wider scope before enough candidates came back; `levels used` is the filter level that produced the final candidates (radius, nuts3, nuts2, country, none).

## Chunk profile

| variant | chunks | indexed chars | median tok (est.) | timed questions | anchored |
|---------|--------|---------------|-------------------|-----------------|----------|
| base    | 726    | 1,064,470     | 644               | 500             | 0        |

## Coverage

| variant | method                        | answered | scored (aligned) | seconds | failures |
|---------|-------------------------------|----------|------------------|---------|----------|
| base    | qwen_hybrid_rerank            | 500      | 500              | 720.0   | 0        |
| base    | qwen_hybrid_rerank_geo_strict | 500      | 500              | 784.2   | 0        |

## Notes

- Every variant's `Overall` table scores only the questions all of its started cells have answered (`scored (aligned)`), so a partially run cell never lowers a finished one.
- Agentic diagnostics pair each agent with its reranked baseline in the same variant; a missing pair means the baseline was not in `--methods`.
- `judge_hit@K` is absent: run with --judge-equivalence to add it.
