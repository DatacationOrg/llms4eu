# Chunk Size Sweep 2026-09-01

## Run Configuration

- Command: `experiments/indexing/compare_chunkings.py --variants base --methods qwen_hybrid_rerank,dci --design shared --limit 500 --keep-checkpoint --output docs/reports/agentic/agentic-dci-2026-09-01.md`
- Variants: base
- Methods: qwen_hybrid_rerank, dci
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

| method             | base      |
|--------------------|-----------|
| qwen_hybrid_rerank | 1.419     |
| dci                | **1.358** |

## Retrieval Quality (whole chunks)

Every variant is scored on the same questions, with its labels projected from the shared answer anchors, so no column is measured on questions written for its own cutting and the sample size is identical across the row.

The confound that remains: **a variant with more chunks is a harder haystack**, and `hit@k` counts whole chunks, so a larger chunk is likelier to contain any given answer. Both effects survive shared labelling. The span metrics above are the ones to rank on; these are here because they are the numbers every earlier report used.

### hit@1

| method             | base      |
|--------------------|-----------|
| qwen_hybrid_rerank | **0.731** |
| dci                | 0.576     |

### hit@5

| method             | base      |
|--------------------|-----------|
| qwen_hybrid_rerank | **0.881** |
| dci                | 0.697     |

### hit@10

| method             | base      |
|--------------------|-----------|
| qwen_hybrid_rerank | **0.911** |
| dci                | 0.723     |

### recall@10

| method             | base      |
|--------------------|-----------|
| qwen_hybrid_rerank | **0.911** |
| dci                | 0.723     |

### mrr@10

| method             | base      |
|--------------------|-----------|
| qwen_hybrid_rerank | **0.796** |
| dci                | 0.632     |

## Speed

Milliseconds per query, so the columns stay comparable when the variants hold different numbers of questions.

| method             | base    |
|--------------------|---------|
| qwen_hybrid_rerank | 1423.8  |
| dci                | 22084.3 |

Retrieval calls per query — above 1.00 means a method issued extra searches for some queries.

| method             | base |
|--------------------|------|
| qwen_hybrid_rerank | 1.00 |
| dci                | 5.85 |

## hit@5 by category

### crosslingual

| method             | base  |
|--------------------|-------|
| qwen_hybrid_rerank | 0.857 |
| dci                | 0.352 |

### direct_long

| method             | base  |
|--------------------|-------|
| qwen_hybrid_rerank | 0.940 |
| dci                | 0.892 |

### direct_short

| method             | base  |
|--------------------|-------|
| qwen_hybrid_rerank | 0.895 |
| dci                | 0.800 |

### vague_long

| method             | base  |
|--------------------|-------|
| qwen_hybrid_rerank | 0.972 |
| dci                | 0.813 |

### vague_short

| method             | base  |
|--------------------|-------|
| qwen_hybrid_rerank | 0.752 |
| dci                | 0.624 |

## Coverage

| Variant | Method | Questions scored | Anchored | Seconds |
|---|---|---:|---:|---:|
| base | qwen_hybrid_rerank | 495 | 0 | 712.8 |
| base | dci | 495 | 0 | 11051.2 |

`Anchored` is the sample the character-overlap metrics average over. Under the `gold` span target it equals the labelled sample, because every question has a base chunk. Under `anchor` it is smaller, and the same shared sample for every variant — anchors are placed once, before any chunking — so the columns stay comparable and the shortfall costs statistical power, not validity. A zero here means the span metrics never ran: the target set was empty.

## Recommendation

Best measured cell: `base` with `qwen_hybrid_rerank` at recall@10 0.911 over 495 questions. That is within noise of `base` on the same method, so the current chunking is not measurably costing recall@10. Nothing is truncated at qwen's limit. Compared variants: base. **Ranked on `recall@10` because no span metric was available**, and that metric favours large chunks by construction: anchor the answers first (`uv run python -m src.eval.anchors`) before trusting this ordering.
