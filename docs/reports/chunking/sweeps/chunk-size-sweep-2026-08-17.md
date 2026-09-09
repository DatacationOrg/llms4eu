# Chunk Size Sweep 2026-08-17

## Run Configuration

- Command: `experiments/indexing/compare_chunkings.py --variants base,tok256,tok512,tok512ov,tok1024 --methods sweep-completion-cheap --design per-variant --output docs/chunk-size-sweep-2026-08-17.md`
- Variants: base, tok256, tok512, tok512ov, tok1024
- Methods: nemotron_hybrid, nemotron_hybrid_rerank
- Questions: all | Category: all | Warmup: 5
- Question-set design: **per-variant** — each variant owns questions generated from its own chunks, so gold is correct by construction but the columns are differently sized samples.
- Span target: **gold** — the base chunk the question came from, so `base` scores a trivial 1.000 and is the ruler rather than a competitor.

## Chunk Profile

| Variant | Chunks | english max/over limit | nemotron max/over limit | qwen max/over limit | qwen4b max/over limit | qwen8b max/over limit | qwen_s512 max/over limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 726 | 1302 tok, 90.1% over 256 | 1167 tok, 0.0% over 32768 | 1253 tok, 0.0% over 32768 | 1253 tok, 0.0% over 40960 | 1253 tok, 0.0% over 32768 | 1253 tok, 69.7% over 512 |
| tok256 | 2003 | 539 tok, 36.8% over 256 | 411 tok, 0.0% over 32768 | 409 tok, 0.0% over 32768 | 409 tok, 0.0% over 40960 | 409 tok, 0.0% over 32768 | 409 tok, 0.0% over 512 |
| tok512 | 990 | 944 tok, 95.7% over 256 | 745 tok, 0.0% over 32768 | 796 tok, 0.0% over 32768 | 796 tok, 0.0% over 40960 | 796 tok, 0.0% over 32768 | 796 tok, 16.0% over 512 |
| tok512ov | 1049 | 968 tok, 96.0% over 256 | 750 tok, 0.0% over 32768 | 792 tok, 0.0% over 32768 | 792 tok, 0.0% over 40960 | 792 tok, 0.0% over 32768 | 792 tok, 15.3% over 512 |
| tok1024 | 501 | 2063 tok, 93.4% over 256 | 1409 tok, 0.0% over 32768 | 1573 tok, 0.0% over 32768 | 1573 tok, 0.0% over 40960 | 1573 tok, 0.0% over 32768 | 1573 tok, 90.2% over 512 |

A variant whose inputs exceed a provider's sequence limit is being truncated: its scores describe a corpus the embedder never fully read.

## Retrieval Quality (character overlap)

Measured against the answer's character span, not against whole chunks. `char_recall` is how much of the answer the top-k chunks cover, `char_precision` how much of the retrieved text is answer, `iou` the two together. `budget_recall@N` fills N characters of context in rank order and asks how much of the answer got in — the comparison that decides an indexing choice, since the generator's context window is what is scarce.

## Retrieval Quality (whole chunks)

Each variant is scored on questions generated from its own chunks, so its gold chunk is correct by construction and no variant is measured on questions written for another cutting.

The confound that replaces it: **a variant with more chunks is a harder haystack.** See the chunk counts above — finding one chunk among 2,003 is harder than among 501, independently of chunk quality, so a small-chunk variant is penalised for reasons that have nothing to do with how well it was cut. Read a large gap as real and a small one as possibly just haystack size.

### hit@1

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.546 | 0.518  | 0.573  | 0.538    | 0.604   |
| nemotron_hybrid_rerank | 0.746 | 0.720  | 0.765  | 0.743    | 0.811   |

### hit@5

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.744 | 0.713  | 0.787  | 0.774    | 0.824   |
| nemotron_hybrid_rerank | 0.875 | 0.848  | 0.902  | 0.898    | 0.925   |

### hit@10

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.824 | 0.800  | 0.865  | 0.853    | 0.888   |
| nemotron_hybrid_rerank | 0.893 | 0.868  | 0.920  | 0.916    | 0.940   |

### recall@10

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.824 | 0.800  | 0.865  | 0.853    | 0.888   |
| nemotron_hybrid_rerank | 0.893 | 0.868  | 0.920  | 0.916    | 0.940   |

### mrr@10

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.631 | 0.606  | 0.668  | 0.642    | 0.698   |
| nemotron_hybrid_rerank | 0.803 | 0.776  | 0.825  | 0.813    | 0.860   |

## Speed

Milliseconds per query, so the columns stay comparable when the variants hold different numbers of questions.

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 2.0   | 2.9    | 1.9    | 2.3      | 1.5     |
| nemotron_hybrid_rerank | 713.9 | 307.1  | 549.6  | 545.9    | 1006.2  |

Retrieval calls per query — above 1.00 means a method issued extra searches for some queries.

| method                 | base | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|------|--------|--------|----------|---------|
| nemotron_hybrid        | 1.00 | 1.00   | 1.00   | 1.00     | 1.00    |
| nemotron_hybrid_rerank | 1.00 | 1.00   | 1.00   | 1.00     | 1.00    |

## hit@5 by category

### crosslingual

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.793 | 0.696  | 0.734  | 0.749    | 0.823   |
| nemotron_hybrid_rerank | 0.858 | 0.760  | 0.816  | 0.805    | 0.885   |

### direct_long

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.899 | 0.867  | 0.907  | 0.915    | 0.924   |
| nemotron_hybrid_rerank | 0.973 | 0.952  | 0.974  | 0.978    | 0.970   |

### direct_short

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.766 | 0.786  | 0.891  | 0.849    | 0.894   |
| nemotron_hybrid_rerank | 0.893 | 0.924  | 0.964  | 0.963    | 0.987   |

### vague_long

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.825 | 0.627  | 0.711  | 0.692    | 0.749   |
| nemotron_hybrid_rerank | 0.929 | 0.789  | 0.875  | 0.861    | 0.906   |

### vague_short

| method                 | base  | tok256 | tok512 | tok512ov | tok1024 |
|------------------------|-------|--------|--------|----------|---------|
| nemotron_hybrid        | 0.453 | 0.578  | 0.663  | 0.650    | 0.730   |
| nemotron_hybrid_rerank | 0.734 | 0.771  | 0.837  | 0.839    | 0.868   |

## Coverage

| Variant | Method | Questions scored | Anchored | Seconds |
|---|---|---:|---:|---:|
| base | nemotron_hybrid | 3471 | 0 | 7.6 |
| base | nemotron_hybrid_rerank | 3471 | 0 | 2482.6 |
| tok256 | nemotron_hybrid | 7801 | 0 | 25.5 |
| tok256 | nemotron_hybrid_rerank | 7801 | 0 | 2399.4 |
| tok512 | nemotron_hybrid | 3605 | 0 | 7.9 |
| tok512 | nemotron_hybrid_rerank | 3605 | 0 | 1984.7 |
| tok512ov | nemotron_hybrid | 3865 | 0 | 9.9 |
| tok512ov | nemotron_hybrid_rerank | 3865 | 0 | 2113.2 |
| tok1024 | nemotron_hybrid | 1451 | 0 | 2.5 |
| tok1024 | nemotron_hybrid_rerank | 1451 | 0 | 1465.1 |

`Anchored` is the sample the character-overlap metrics average over. Under the `gold` span target it equals the labelled sample, because every question has a base chunk. Under `anchor` it is smaller, and the same shared sample for every variant — anchors are placed once, before any chunking — so the columns stay comparable and the shortfall costs statistical power, not validity. A zero here means the span metrics never ran: the target set was empty.

## Recommendation

Best measured cell: `tok1024` with `nemotron_hybrid_rerank` at recall@10 0.940 over 1451 questions. That is +0.047 recall@10 against `base` on the same method. Nothing is truncated at nemotron's limit. Compared variants: base, tok1024, tok256, tok512, tok512ov. Question counts differ across variants ([1451, 3471, 3605, 3865, 7801]), which is expected when each variant owns its question set — the smaller samples are noisier, not biased. **Ranked on `recall@10` because no span metric was available**, and that metric favours large chunks by construction: anchor the answers first (`uv run python -m src.eval.anchors`) before trusting this ordering.
