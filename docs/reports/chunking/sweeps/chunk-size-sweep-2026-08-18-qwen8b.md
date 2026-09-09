# Chunk Size Sweep 2026-08-18

## Run Configuration

- Command: `experiments/indexing/compare_chunkings.py --design per-variant --variants base,tok256,tok512,tok512ov,tok1024 --methods qwen8b,qwen8b_hybrid_rerank --keep-checkpoint --output docs/chunk-size-sweep-2026-08-18-qwen8b.md`
- Variants: base, tok256, tok512, tok512ov, tok1024
- Methods: qwen8b, qwen8b_hybrid_rerank
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
| qwen8b               | 0.691 | 0.730  | 0.731  | 0.718    | 0.752   |
| qwen8b_hybrid_rerank | 0.764 | 0.751  | 0.789  | 0.766    | 0.837   |

### hit@5

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.884 | 0.898  | 0.931  | 0.916    | 0.945   |
| qwen8b_hybrid_rerank | 0.912 | 0.906  | 0.948  | 0.933    | 0.970   |

### hit@10

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.928 | 0.931  | 0.962  | 0.949    | 0.972   |
| qwen8b_hybrid_rerank | 0.945 | 0.937  | 0.975  | 0.965    | 0.988   |

### recall@10

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.928 | 0.931  | 0.962  | 0.949    | 0.972   |
| qwen8b_hybrid_rerank | 0.945 | 0.937  | 0.975  | 0.965    | 0.988   |

### mrr@10

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.775 | 0.803  | 0.817  | 0.805    | 0.833   |
| qwen8b_hybrid_rerank | 0.829 | 0.818  | 0.859  | 0.841    | 0.893   |

## Speed

Milliseconds per query, so the columns stay comparable when the variants hold different numbers of questions.

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 17.1  | 12.9   | 15.3   | 20.1     | 10.9    |
| qwen8b_hybrid_rerank | 692.1 | 463.2  | 929.9  | 676.2    | 903.8   |

Retrieval calls per query — above 1.00 means a method issued extra searches for some queries.

| method               | base | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|------|--------|--------|----------|---------|
| qwen8b               | 1.00 | 1.00   | 1.00   | 1.00     | 1.00    |
| qwen8b_hybrid_rerank | 1.00 | 1.00   | 1.00   | 1.00     | 1.00    |

## hit@5 by category

### crosslingual

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.896 | 0.874  | 0.897  | 0.885    | 0.951   |
| qwen8b_hybrid_rerank | 0.890 | 0.823  | 0.874  | 0.844    | 0.938   |

### direct_long

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.976 | 0.966  | 0.980  | 0.979    | 0.990   |
| qwen8b_hybrid_rerank | 0.986 | 0.971  | 0.984  | 0.987    | 0.987   |

### direct_short

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.902 | 0.938  | 0.950  | 0.940    | 0.970   |
| qwen8b_hybrid_rerank | 0.912 | 0.937  | 0.968  | 0.969    | 0.993   |

### vague_long

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.944 | 0.878  | 0.921  | 0.911    | 0.940   |
| qwen8b_hybrid_rerank | 0.968 | 0.895  | 0.952  | 0.936    | 0.970   |

### vague_short

| method               | base  | tok256 | tok512 | tok512ov | tok1024 |
|----------------------|-------|--------|--------|----------|---------|
| qwen8b               | 0.715 | 0.824  | 0.889  | 0.853    | 0.875   |
| qwen8b_hybrid_rerank | 0.812 | 0.861  | 0.927  | 0.889    | 0.957   |

## Coverage

| Variant | Method | Questions scored | Anchored | Seconds |
|---|---|---:|---:|---:|
| base | qwen8b | 3471 | 0 | 59.9 |
| base | qwen8b_hybrid_rerank | 3471 | 0 | 2406.6 |
| tok256 | qwen8b | 7801 | 0 | 103.1 |
| tok256 | qwen8b_hybrid_rerank | 7801 | 0 | 3617.9 |
| tok512 | qwen8b | 3605 | 0 | 56.2 |
| tok512 | qwen8b_hybrid_rerank | 3605 | 0 | 3357.8 |
| tok512ov | qwen8b | 3865 | 0 | 78.7 |
| tok512ov | qwen8b_hybrid_rerank | 3865 | 0 | 2619.8 |
| tok1024 | qwen8b | 1451 | 0 | 16.2 |
| tok1024 | qwen8b_hybrid_rerank | 1451 | 0 | 1316.4 |

`Anchored` is the sample the character-overlap metrics average over. Under the `gold` span target it equals the labelled sample, because every question has a base chunk. Under `anchor` it is smaller, and the same shared sample for every variant — anchors are placed once, before any chunking — so the columns stay comparable and the shortfall costs statistical power, not validity. A zero here means the span metrics never ran: the target set was empty.

## Recommendation

Best measured cell: `tok1024` with `qwen8b_hybrid_rerank` at recall@10 0.988 over 1451 questions. That is +0.042 recall@10 against `base` on the same method. Nothing is truncated at qwen8b's limit. Compared variants: base, tok1024, tok256, tok512, tok512ov. Question counts differ across variants ([1451, 3471, 3605, 3865, 7801]), which is expected when each variant owns its question set — the smaller samples are noisier, not biased. **Ranked on `recall@10` because no span metric was available**, and that metric favours large chunks by construction: anchor the answers first (`uv run python -m src.eval.anchors`) before trusting this ordering.
