# Chunk Size Sweep 2026-08-18

## Run Configuration

- Command: `experiments/indexing/compare_chunkings.py --design per-variant --variants base,tok256,tok512,tok512ov,tok1024 --methods qwen4b,qwen4b_hybrid_rerank --keep-checkpoint --output docs/chunk-size-sweep-2026-08-18-qwen4b.md`
- Variants: base, tok256, tok512, tok512ov, tok1024
- Methods: qwen4b, qwen4b_hybrid_rerank
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

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.657 | 0.695  | 0.709  | 0.693    | 0.731   |
| qwen4b_hybrid_rerank | 0.764 | 0.750  | 0.792  | 0.765    | 0.837   |

### hit@5

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.848 | 0.877  | 0.909  | 0.896    | 0.922   |
| qwen4b_hybrid_rerank | 0.907 | 0.903  | 0.946  | 0.935    | 0.966   |

### hit@10

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.899 | 0.910  | 0.947  | 0.929    | 0.957   |
| qwen4b_hybrid_rerank | 0.936 | 0.931  | 0.973  | 0.960    | 0.983   |

### recall@10

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.899 | 0.910  | 0.947  | 0.929    | 0.957   |
| qwen4b_hybrid_rerank | 0.936 | 0.931  | 0.973  | 0.960    | 0.983   |

### mrr@10

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.742 | 0.774  | 0.796  | 0.783    | 0.815   |
| qwen4b_hybrid_rerank | 0.827 | 0.815  | 0.860  | 0.840    | 0.892   |

## Speed

Milliseconds per query, so the columns stay comparable when the variants hold different numbers of questions.

| method               | base   | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|--------|--------|--------|----------|---------|
| qwen4b               | 19.0   | 8.0    | 7.5    | 6.9      | 7.0     |
| qwen4b_hybrid_rerank | 1133.6 | 287.4  | 496.0  | 498.4    | 902.2   |

Retrieval calls per query — above 1.00 means a method issued extra searches for some queries.

| method               | base | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|------|--------|--------|----------|---------|
| qwen4b               | 1.00 | 1.00   | 1.00   | 1.00     | 1.00    |
| qwen4b_hybrid_rerank | 1.00 | 1.00   | 1.00   | 1.00     | 1.00    |

## hit@5 by category

### crosslingual

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.866 | 0.840  | 0.879  | 0.859    | 0.914   |
| qwen4b_hybrid_rerank | 0.890 | 0.816  | 0.881  | 0.840    | 0.938   |

### direct_long

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.971 | 0.955  | 0.969  | 0.964    | 0.983   |
| qwen4b_hybrid_rerank | 0.989 | 0.970  | 0.983  | 0.988    | 0.987   |

### direct_short

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.869 | 0.929  | 0.934  | 0.929    | 0.957   |
| qwen4b_hybrid_rerank | 0.906 | 0.939  | 0.970  | 0.969    | 0.990   |

### vague_long

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.909 | 0.859  | 0.895  | 0.895    | 0.913   |
| qwen4b_hybrid_rerank | 0.967 | 0.888  | 0.947  | 0.936    | 0.963   |

### vague_short

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen4b               | 0.637 | 0.785  | 0.852  | 0.816    | 0.842   |
| qwen4b_hybrid_rerank | 0.795 | 0.857  | 0.919  | 0.899    | 0.947   |

## Coverage

| Variant | Method | Questions scored | Anchored | Seconds |
|---|---|---:|---:|---:|
| base | qwen4b | 3471 | 0 | 70.7 |
| base | qwen4b_hybrid_rerank | 3471 | 0 | 3942.7 |
| tok256 | qwen4b | 7801 | 0 | 65.4 |
| tok256 | qwen4b_hybrid_rerank | 7801 | 0 | 2246.0 |
| tok512 | qwen4b | 3605 | 0 | 27.8 |
| tok512 | qwen4b_hybrid_rerank | 3605 | 0 | 1791.3 |
| tok512ov | qwen4b | 3865 | 0 | 27.8 |
| tok512ov | qwen4b_hybrid_rerank | 3865 | 0 | 1929.9 |
| tok1024 | qwen4b | 1451 | 0 | 10.5 |
| tok1024 | qwen4b_hybrid_rerank | 1451 | 0 | 1313.9 |

`Anchored` is the sample the character-overlap metrics average over. Under the `gold` span target it equals the labelled sample, because every question has a base chunk. Under `anchor` it is smaller, and the same shared sample for every variant — anchors are placed once, before any chunking — so the columns stay comparable and the shortfall costs statistical power, not validity. A zero here means the span metrics never ran: the target set was empty.

## Recommendation

Best measured cell: `tok1024` with `qwen4b_hybrid_rerank` at recall@10 0.983 over 1451 questions. That is +0.047 recall@10 against `base` on the same method. Nothing is truncated at qwen4b's limit. Compared variants: base, tok1024, tok256, tok512, tok512ov. Question counts differ across variants ([1451, 3471, 3605, 3865, 7801]), which is expected when each variant owns its question set — the smaller samples are noisier, not biased. **Ranked on `recall@10` because no span metric was available**, and that metric favours large chunks by construction: anchor the answers first (`uv run python -m src.eval.anchors`) before trusting this ordering.
