# Chunk Size Sweep 2026-08-18

## Run Configuration

- Command: `experiments/indexing/compare_chunkings.py --design per-variant --variants base,tok256,tok512,tok512ov,tok1024 --methods nemotron8b,nemotron8b_hybrid_rerank --keep-checkpoint --output docs/chunk-size-sweep-2026-08-18-nemotron8b.md`
- Variants: base, tok256, tok512, tok512ov, tok1024
- Methods: nemotron8b, nemotron8b_hybrid_rerank
- Questions: all | Category: all | Warmup: 5
- Question-set design: **per-variant** — each variant owns questions generated from its own chunks, so gold is correct by construction but the columns are differently sized samples.
- Span target: **gold** — the base chunk the question came from, so `base` scores a trivial 1.000 and is the ruler rather than a competitor.

## Chunk Profile

| Variant | Chunks | english max/over limit | nemotron max/over limit | nemotron8b max/over limit | qwen max/over limit | qwen4b max/over limit | qwen8b max/over limit | qwen_s512 max/over limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 726 | 1302 tok, 90.1% over 256 | 1167 tok, 0.0% over 32768 | 1167 tok, 0.0% over 32768 | 1253 tok, 0.0% over 32768 | 1253 tok, 0.0% over 40960 | 1253 tok, 0.0% over 32768 | 1253 tok, 69.7% over 512 |
| tok256 | 2003 | 539 tok, 36.8% over 256 | 411 tok, 0.0% over 32768 | 411 tok, 0.0% over 32768 | 409 tok, 0.0% over 32768 | 409 tok, 0.0% over 40960 | 409 tok, 0.0% over 32768 | 409 tok, 0.0% over 512 |
| tok512 | 990 | 944 tok, 95.7% over 256 | 745 tok, 0.0% over 32768 | 745 tok, 0.0% over 32768 | 796 tok, 0.0% over 32768 | 796 tok, 0.0% over 40960 | 796 tok, 0.0% over 32768 | 796 tok, 16.0% over 512 |
| tok512ov | 1049 | 968 tok, 96.0% over 256 | 750 tok, 0.0% over 32768 | 750 tok, 0.0% over 32768 | 792 tok, 0.0% over 32768 | 792 tok, 0.0% over 40960 | 792 tok, 0.0% over 32768 | 792 tok, 15.3% over 512 |
| tok1024 | 501 | 2063 tok, 93.4% over 256 | 1409 tok, 0.0% over 32768 | 1409 tok, 0.0% over 32768 | 1573 tok, 0.0% over 32768 | 1573 tok, 0.0% over 40960 | 1573 tok, 0.0% over 32768 | 1573 tok, 90.2% over 512 |

A variant whose inputs exceed a provider's sequence limit is being truncated: its scores describe a corpus the embedder never fully read.

## Retrieval Quality (character overlap)

Measured against the answer's character span, not against whole chunks. `char_recall` is how much of the answer the top-k chunks cover, `char_precision` how much of the retrieved text is answer, `iou` the two together. `budget_recall@N` fills N characters of context in rank order and asks how much of the answer got in — the comparison that decides an indexing choice, since the generator's context window is what is scarce.

## Retrieval Quality (whole chunks)

Each variant is scored on questions generated from its own chunks, so its gold chunk is correct by construction and no variant is measured on questions written for another cutting.

The confound that replaces it: **a variant with more chunks is a harder haystack.** See the chunk counts above — finding one chunk among 2,003 is harder than among 501, independently of chunk quality, so a small-chunk variant is penalised for reasons that have nothing to do with how well it was cut. Read a large gap as real and a small one as possibly just haystack size.

### hit@1

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.802 | 0.826  | 0.840  | 0.821    | 0.829   |
| nemotron8b_hybrid_rerank | 0.765 | 0.754  | 0.793  | 0.765    | 0.837   |

### hit@5

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.940 | 0.943  | 0.969  | 0.960    | 0.970   |
| nemotron8b_hybrid_rerank | 0.917 | 0.909  | 0.950  | 0.939    | 0.968   |

### hit@10

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.964 | 0.961  | 0.983  | 0.977    | 0.989   |
| nemotron8b_hybrid_rerank | 0.954 | 0.942  | 0.979  | 0.969    | 0.989   |

### recall@10

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.964 | 0.961  | 0.983  | 0.977    | 0.989   |
| nemotron8b_hybrid_rerank | 0.954 | 0.942  | 0.979  | 0.969    | 0.989   |

### mrr@10

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.863 | 0.876  | 0.897  | 0.883    | 0.890   |
| nemotron8b_hybrid_rerank | 0.832 | 0.822  | 0.861  | 0.844    | 0.893   |

## Speed

Milliseconds per query, so the columns stay comparable when the variants hold different numbers of questions.

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 11.2  | 8.4    | 7.7    | 7.2      | 7.1     |
| nemotron8b_hybrid_rerank | 702.7 | 290.2  | 501.0  | 503.2    | 917.4   |

Retrieval calls per query — above 1.00 means a method issued extra searches for some queries.

| method                   | base | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|------|--------|--------|----------|---------|
| nemotron8b               | 1.00 | 1.00   | 1.00   | 1.00     | 1.00    |
| nemotron8b_hybrid_rerank | 1.00 | 1.00   | 1.00   | 1.00     | 1.00    |

## hit@5 by category

### crosslingual

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.954 | 0.909  | 0.958  | 0.931    | 0.971   |
| nemotron8b_hybrid_rerank | 0.900 | 0.825  | 0.881  | 0.846    | 0.930   |

### direct_long

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.989 | 0.983  | 0.994  | 0.994    | 0.997   |
| nemotron8b_hybrid_rerank | 0.992 | 0.974  | 0.984  | 0.988    | 0.990   |

### direct_short

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.944 | 0.971  | 0.979  | 0.973    | 0.974   |
| nemotron8b_hybrid_rerank | 0.915 | 0.939  | 0.966  | 0.969    | 0.987   |

### vague_long

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.988 | 0.937  | 0.966  | 0.965    | 0.973   |
| nemotron8b_hybrid_rerank | 0.973 | 0.895  | 0.952  | 0.936    | 0.970   |

### vague_short

| method                   | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------------|-------|--------|--------|----------|---------|
| nemotron8b               | 0.833 | 0.899  | 0.942  | 0.924    | 0.934   |
| nemotron8b_hybrid_rerank | 0.815 | 0.871  | 0.935  | 0.912    | 0.954   |

## Coverage

| Variant | Method | Questions scored | Anchored | Seconds |
|---|---|---:|---:|---:|
| base | nemotron8b | 3471 | 0 | 46.1 |
| base | nemotron8b_hybrid_rerank | 3471 | 0 | 2444.0 |
| tok256 | nemotron8b | 7801 | 0 | 68.0 |
| tok256 | nemotron8b_hybrid_rerank | 7801 | 0 | 2267.9 |
| tok512 | nemotron8b | 3605 | 0 | 28.7 |
| tok512 | nemotron8b_hybrid_rerank | 3605 | 0 | 1809.4 |
| tok512ov | nemotron8b | 3865 | 0 | 28.7 |
| tok512ov | nemotron8b_hybrid_rerank | 3865 | 0 | 1948.4 |
| tok1024 | nemotron8b | 1451 | 0 | 10.6 |
| tok1024 | nemotron8b_hybrid_rerank | 1451 | 0 | 1336.2 |

`Anchored` is the sample the character-overlap metrics average over. Under the `gold` span target it equals the labelled sample, because every question has a base chunk. Under `anchor` it is smaller, and the same shared sample for every variant — anchors are placed once, before any chunking — so the columns stay comparable and the shortfall costs statistical power, not validity. A zero here means the span metrics never ran: the target set was empty.

## Recommendation

Best measured cell: `tok1024` with `nemotron8b` at recall@10 0.989 over 1451 questions. That is +0.025 recall@10 against `base` on the same method. Nothing is truncated at nemotron8b's limit. Compared variants: base, tok1024, tok256, tok512, tok512ov. Question counts differ across variants ([1451, 3471, 3605, 3865, 7801]), which is expected when each variant owns its question set — the smaller samples are noisier, not biased. **Ranked on `recall@10` because no span metric was available**, and that metric favours large chunks by construction: anchor the answers first (`uv run python -m src.eval.anchors`) before trusting this ordering.
