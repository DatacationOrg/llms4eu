# Chunk Size Sweep 2026-08-31

## Run Configuration

- Command: `experiments/indexing/compare_chunkings.py --variants base --methods qwen_hybrid_rerank,qwen_hybrid_agentic,qwen_hybrid_agentic_tools --design shared --limit 500 --agentic-diagnostics --keep-checkpoint --output docs/reports/agentic/agentic-tools-2026-08-31.md`
- Variants: base
- Methods: qwen_hybrid_rerank, qwen_hybrid_agentic, qwen_hybrid_agentic_tools
- Questions: 500 | Category: all | Warmup: 5
- Question-set design: **shared** — one question set, labels projected onto each variant by `src/eval/relabel.py`, so every column scores the same questions and one method's row is comparable straight across.
- Span target: **gold** — the base chunk the question came from, so `base` scores a trivial 1.000 and is the ruler rather than a competitor.

## Chunk Profile

| Variant | Chunks | Indexed chars | english max/over limit | nemotron max/over limit | nemotron8b max/over limit | qwen max/over limit | qwen4b max/over limit | qwen8b max/over limit | qwen_s512 max/over limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 726 | 1,064,470 | 1302 tok, 90.1% over 256 | 1167 tok, 0.0% over 32768 | 1167 tok, 0.0% over 32768 | 1253 tok, 0.0% over 32768 | 1253 tok, 0.0% over 40960 | 1253 tok, 0.0% over 32768 | 1253 tok, 69.7% over 512 |

`Indexed chars` is what `store_share@k` is a share of: every copy the store holds, so an overlapping cut shows the text it duplicates rather than hiding it. Variants over the same pages should differ here only by overlap and by the chunks each cut drops as too short.

A variant whose inputs exceed a provider's sequence limit is being truncated: its scores describe a corpus the embedder never fully read.

## Retrieval Quality (character overlap)

Measured against the answer's character span, not against whole chunks. `char_recall` is how much of the answer the top-k chunks cover, `char_precision` how much of the retrieved text is answer, `iou` the two together. `budget_recall@N` fills N characters of context in rank order and asks how much of the answer got in — the comparison that decides an indexing choice, since the generator's context window is what is scarce.

## Retrieval Cost (share of the index)

What the recall above cost. `store_share@10` is the percentage of the variant's whole index one query returns at k=10 — about 2.0% for a 1,024-token cut against 0.5% for a 256-token one, because k slots of a larger chunk are simply more text. **Lower is better here, so the bold cell in that table is the cheapest method for that variant rather than the largest number.** `recall_per_share@10` divides `char_recall@10` by it: answer coverage earned per percent of the index read.

This is the column that stops chunk size deciding the ranking on its own. Every other quality metric here can be bought with size — return four times the text and cover more of any answer without retrieving one bit better — and until this section existed nothing in the report charged a cut for the text it returned. Two cuts at equal `char_recall` are separated by what each had to read to get there. Read it beside `budget_recall@4000`, which fixes the cost instead of pricing it: agreement between the two is the strong result, and a variant that wins `char_recall@10` while losing both of these won on size.

### store_share@10

| method                    | base      |
|---------------------------|-----------|
| qwen_hybrid_rerank        | 1.417     |
| qwen_hybrid_agentic       | 1.429     |
| qwen_hybrid_agentic_tools | **1.402** |

## Retrieval Quality (whole chunks)

Every variant is scored on the same questions, with its labels projected from the shared answer anchors, so no column is measured on questions written for its own cutting and the sample size is identical across the row.

The confound that remains: **a variant with more chunks is a harder haystack**, and `hit@k` counts whole chunks, so a larger chunk is likelier to contain any given answer. Both effects survive shared labelling. The span metrics above are the ones to rank on; these are here because they are the numbers every earlier report used.

### hit@1

| method                    | base      |
|---------------------------|-----------|
| qwen_hybrid_rerank        | 0.723     |
| qwen_hybrid_agentic       | **0.725** |
| qwen_hybrid_agentic_tools | 0.719     |

### hit@5

| method                    | base      |
|---------------------------|-----------|
| qwen_hybrid_rerank        | 0.881     |
| qwen_hybrid_agentic       | **0.889** |
| qwen_hybrid_agentic_tools | 0.879     |

### hit@10

| method                    | base      |
|---------------------------|-----------|
| qwen_hybrid_rerank        | 0.907     |
| qwen_hybrid_agentic       | **0.915** |
| qwen_hybrid_agentic_tools | 0.903     |

### recall@10

| method                    | base      |
|---------------------------|-----------|
| qwen_hybrid_rerank        | 0.907     |
| qwen_hybrid_agentic       | **0.915** |
| qwen_hybrid_agentic_tools | 0.903     |

### mrr@10

| method                    | base      |
|---------------------------|-----------|
| qwen_hybrid_rerank        | 0.791     |
| qwen_hybrid_agentic       | **0.795** |
| qwen_hybrid_agentic_tools | 0.787     |

## Speed

Milliseconds per query, so the columns stay comparable when the variants hold different numbers of questions.

| method                    | base   |
|---------------------------|--------|
| qwen_hybrid_rerank        | 675.4  |
| qwen_hybrid_agentic       | 6062.1 |
| qwen_hybrid_agentic_tools | 6572.2 |

Retrieval calls per query — above 1.00 means a method issued extra searches for some queries.

| method                    | base |
|---------------------------|------|
| qwen_hybrid_rerank        | 1.00 |
| qwen_hybrid_agentic       | 1.21 |
| qwen_hybrid_agentic_tools | 1.25 |

## hit@5 by category

### crosslingual

| method                    | base  |
|---------------------------|-------|
| qwen_hybrid_rerank        | 0.857 |
| qwen_hybrid_agentic       | 0.901 |
| qwen_hybrid_agentic_tools | 0.857 |

### direct_long

| method                    | base  |
|---------------------------|-------|
| qwen_hybrid_rerank        | 0.940 |
| qwen_hybrid_agentic       | 0.952 |
| qwen_hybrid_agentic_tools | 0.940 |

### direct_short

| method                    | base  |
|---------------------------|-------|
| qwen_hybrid_rerank        | 0.905 |
| qwen_hybrid_agentic       | 0.895 |
| qwen_hybrid_agentic_tools | 0.895 |

### vague_long

| method                    | base  |
|---------------------------|-------|
| qwen_hybrid_rerank        | 0.963 |
| qwen_hybrid_agentic       | 0.953 |
| qwen_hybrid_agentic_tools | 0.963 |

### vague_short

| method                    | base  |
|---------------------------|-------|
| qwen_hybrid_rerank        | 0.752 |
| qwen_hybrid_agentic       | 0.761 |
| qwen_hybrid_agentic_tools | 0.752 |

## Agentic Diagnostics

### base | qwen_hybrid_agentic

Agentic diagnostics (paired at hit@5)

| agent | baseline | queries | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|-------|----------|---------|---------|-----------|----------|-----------------|--------------|-------------|--------|----------|-------------|
| qwen_hybrid_agentic | qwen_hybrid_rerank | 495 | 0 (0.0%) | 0 | 0 | - | 0.000 | 9 | 5 | - | - |

Retry precision is the fraction of retried queries whose first relevant rank improved. Retry recall is the fraction of baseline misses that were retried.

### base | qwen_hybrid_agentic_tools

Agentic diagnostics (paired at hit@5)

| agent | baseline | queries | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|-------|----------|---------|---------|-----------|----------|-----------------|--------------|-------------|--------|----------|-------------|
| qwen_hybrid_agentic_tools | qwen_hybrid_rerank | 495 | 0 (0.0%) | 0 | 0 | - | 0.000 | 2 | 3 | - | - |

Retry precision is the fraction of retried queries whose first relevant rank improved. Retry recall is the fraction of baseline misses that were retried.

## Coverage

| Variant | Method | Questions scored | Anchored | Seconds |
|---|---|---:|---:|---:|
| base | qwen_hybrid_rerank | 495 | 0 | 339.0 |
| base | qwen_hybrid_agentic | 495 | 0 | 3367.1 |
| base | qwen_hybrid_agentic_tools | 495 | 0 | 3613.4 |

`Anchored` is the sample the character-overlap metrics average over. Under the `gold` span target it equals the labelled sample, because every question has a base chunk. Under `anchor` it is smaller, and the same shared sample for every variant — anchors are placed once, before any chunking — so the columns stay comparable and the shortfall costs statistical power, not validity. A zero here means the span metrics never ran: the target set was empty.

## Recommendation

Best measured cell: `base` with `qwen_hybrid_agentic` at recall@10 0.915 over 495 questions. That is within noise of `base` on the same method, so the current chunking is not measurably costing recall@10. Nothing is truncated at qwen's limit. Compared variants: base. **Ranked on `recall@10` because no span metric was available**, and that metric favours large chunks by construction: anchor the answers first (`uv run python -m src.eval.anchors`) before trusting this ordering.
