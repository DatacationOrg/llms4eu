# Chunk Size Sweep 2026-08-19

## Run Configuration

- Command: `experiments/indexing/compare_chunkings.py --design per-variant --variants base,tok1024 --methods qwen8b_hybrid_rerank_4b --keep-checkpoint --output docs/chunk-size-sweep-2026-08-18-rerank4b.md`
- Variants: base, tok1024
- Methods: qwen8b_hybrid_rerank_4b
- Questions: all | Category: all | Warmup: 5
- Question-set design: **per-variant** — each variant owns questions generated from its own chunks, so gold is correct by construction but the columns are differently sized samples.
- Span target: **gold** — the base chunk the question came from, so `base` scores a trivial 1.000 and is the ruler rather than a competitor.

## Chunk Profile

| Variant | Chunks | english max/over limit | nemotron max/over limit | nemotron8b max/over limit | qwen max/over limit | qwen4b max/over limit | qwen8b max/over limit | qwen_s512 max/over limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 726 | 1302 tok, 90.1% over 256 | 1167 tok, 0.0% over 32768 | 1167 tok, 0.0% over 32768 | 1253 tok, 0.0% over 32768 | 1253 tok, 0.0% over 40960 | 1253 tok, 0.0% over 32768 | 1253 tok, 69.7% over 512 |
| tok1024 | 501 | 2063 tok, 93.4% over 256 | 1409 tok, 0.0% over 32768 | 1409 tok, 0.0% over 32768 | 1573 tok, 0.0% over 32768 | 1573 tok, 0.0% over 40960 | 1573 tok, 0.0% over 32768 | 1573 tok, 90.2% over 512 |

A variant whose inputs exceed a provider's sequence limit is being truncated: its scores describe a corpus the embedder never fully read.

## Retrieval Quality (character overlap)

Measured against the answer's character span, not against whole chunks. `char_recall` is how much of the answer the top-k chunks cover, `char_precision` how much of the retrieved text is answer, `iou` the two together. `budget_recall@N` fills N characters of context in rank order and asks how much of the answer got in — the comparison that decides an indexing choice, since the generator's context window is what is scarce.

## Retrieval Quality (whole chunks)

Each variant is scored on questions generated from its own chunks, so its gold chunk is correct by construction and no variant is measured on questions written for another cutting.

The confound that replaces it: **a variant with more chunks is a harder haystack.** See the chunk counts above — finding one chunk among 2,003 is harder than among 501, independently of chunk quality, so a small-chunk variant is penalised for reasons that have nothing to do with how well it was cut. Read a large gap as real and a small one as possibly just haystack size.

### hit@1

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.845 | 0.898   |

### hit@5

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.950 | 0.986   |

### hit@10

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.961 | 0.992   |

### recall@10

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.961 | 0.992   |

### mrr@10

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.891 | 0.937   |

## Speed

Milliseconds per query, so the columns stay comparable when the variants hold different numbers of questions.

| method                  | base   | tok1024 |
|-------------------------|--------|---------|
| qwen8b_hybrid_rerank_4b | 2713.7 | 3562.9  |

Retrieval calls per query — above 1.00 means a method issued extra searches for some queries.

| method                  | base | tok1024 |
|-------------------------|------|---------|
| qwen8b_hybrid_rerank_4b | 1.00 | 1.00    |

## hit@5 by category

### crosslingual

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.959 | 0.979   |

### direct_long

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.992 | 0.993   |

### direct_short

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.949 | 0.993   |

### vague_long

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.988 | 0.990   |

### vague_short

| method                  | base  | tok1024 |
|-------------------------|-------|---------|
| qwen8b_hybrid_rerank_4b | 0.867 | 0.970   |

## Coverage

| Variant | Method | Questions scored | Anchored | Seconds |
|---|---|---:|---:|---:|
| base | qwen8b_hybrid_rerank_4b | 3471 | 0 | 9436.0 |
| tok1024 | qwen8b_hybrid_rerank_4b | 1451 | 0 | 5187.7 |

`Anchored` is the sample the character-overlap metrics average over. Under the `gold` span target it equals the labelled sample, because every question has a base chunk. Under `anchor` it is smaller, and the same shared sample for every variant — anchors are placed once, before any chunking — so the columns stay comparable and the shortfall costs statistical power, not validity. A zero here means the span metrics never ran: the target set was empty.

## Recommendation

Best measured cell: `tok1024` with `qwen8b_hybrid_rerank_4b` at recall@10 0.992 over 1451 questions. That is +0.030 recall@10 against `base` on the same method. Nothing is truncated at qwen8b's limit. Compared variants: base, tok1024. Question counts differ across variants ([1451, 3471]), which is expected when each variant owns its question set — the smaller samples are noisier, not biased. **Ranked on `recall@10` because no span metric was available**, and that metric favours large chunks by construction: anchor the answers first (`uv run python -m src.eval.anchors`) before trusting this ordering.
