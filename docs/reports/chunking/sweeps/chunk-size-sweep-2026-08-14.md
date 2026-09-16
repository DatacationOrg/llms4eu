# Chunk Size Sweep 2026-08-14

## Run Configuration

- Command: `experiments/indexing/compare_chunkings.py --variants base,tok256,tok512,tok512ov,tok1024 --methods qwen,qwen_hybrid_rerank,nemotron`
- Variants: base, tok256, tok512, tok512ov, tok1024
- Methods: qwen, qwen_hybrid_rerank, nemotron
- Questions: all | Category: all | Warmup: 5
- Labels are per variant on shared question texts (`src/eval/relabel.py`), so one method's row is comparable across variants.

## Chunk Profile

| Variant | Chunks | english max/over limit | nemotron max/over limit | qwen max/over limit | qwen4b max/over limit | qwen_s512 max/over limit |
|---|---:|---:|---:|---:|---:|---:|
| base | 726 | 1302 tok, 90.1% over 256 | 1167 tok, 0.0% over 32768 | 1253 tok, 0.0% over 32768 | 1253 tok, 0.0% over 40960 | 1253 tok, 69.7% over 512 |
| tok256 | 2003 | 539 tok, 36.8% over 256 | 411 tok, 0.0% over 32768 | 409 tok, 0.0% over 32768 | 409 tok, 0.0% over 40960 | 409 tok, 0.0% over 512 |
| tok512 | 990 | 944 tok, 95.7% over 256 | 745 tok, 0.0% over 32768 | 796 tok, 0.0% over 32768 | 796 tok, 0.0% over 40960 | 796 tok, 16.0% over 512 |
| tok512ov | 1049 | 968 tok, 96.0% over 256 | 750 tok, 0.0% over 32768 | 792 tok, 0.0% over 32768 | 792 tok, 0.0% over 40960 | 792 tok, 15.3% over 512 |
| tok1024 | 501 | 2063 tok, 93.4% over 256 | 1409 tok, 0.0% over 32768 | 1573 tok, 0.0% over 32768 | 1573 tok, 0.0% over 40960 | 1573 tok, 90.2% over 512 |

A variant whose inputs exceed a provider's sequence limit is being truncated: its scores describe a corpus the embedder never fully read.

## Retrieval Quality (character overlap)

Measured against the answer's character span, not against whole chunks. `char_recall` is how much of the answer the top-k chunks cover, `char_precision` how much of the retrieved text is answer, `iou` the two together. `budget_recall@N` fills N characters of context in rank order and asks how much of the answer got in — the comparison that decides an indexing choice, since the generator's context window is what is scarce.

## Retrieval Quality (whole chunks)

Each variant is scored on questions generated from its own chunks, so its gold chunk is correct by construction and no variant is measured on questions written for another cutting.

The confound that replaces it: **a variant with more chunks is a harder haystack.** See the chunk counts above — finding one chunk among 2,003 is harder than among 501, independently of chunk quality, so a small-chunk variant is penalised for reasons that have nothing to do with how well it was cut. Read a large gap as real and a small one as possibly just haystack size.

### hit@1

| method             | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------|-------|--------|--------|----------|---------|
| qwen               | 0.518 | 0.569  | 0.581  | 0.558    | 0.612   |
| qwen_hybrid_rerank | 0.749 | 0.739  | 0.781  | 0.754    | 0.826   |
| nemotron           | 0.458 | 0.444  | 0.482  | 0.454    | 0.517   |

### hit@5

| method             | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------|-------|--------|--------|----------|---------|
| qwen               | 0.737 | 0.777  | 0.817  | 0.802    | 0.841   |
| qwen_hybrid_rerank | 0.880 | 0.881  | 0.931  | 0.914    | 0.948   |
| nemotron           | 0.664 | 0.645  | 0.715  | 0.693    | 0.733   |

### hit@10

| method             | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------|-------|--------|--------|----------|---------|
| qwen               | 0.797 | 0.828  | 0.877  | 0.855    | 0.897   |
| qwen_hybrid_rerank | 0.908 | 0.905  | 0.954  | 0.937    | 0.968   |
| nemotron           | 0.727 | 0.708  | 0.783  | 0.767    | 0.813   |

### recall@10

| method             | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------|-------|--------|--------|----------|---------|
| qwen               | 0.797 | 0.828  | 0.877  | 0.855    | 0.897   |
| qwen_hybrid_rerank | 0.908 | 0.905  | 0.954  | 0.937    | 0.968   |
| nemotron           | 0.727 | 0.708  | 0.783  | 0.767    | 0.813   |

### mrr@10

| method             | base  | tok256 | tok512 | tok512ov | tok1024 |
|--------------------|-------|--------|--------|----------|---------|
| qwen               | 0.611 | 0.658  | 0.683  | 0.665    | 0.710   |
| qwen_hybrid_rerank | 0.808 | 0.800  | 0.847  | 0.826    | 0.880   |
| nemotron           | 0.547 | 0.531  | 0.584  | 0.559    | 0.613   |

## Coverage

| Variant | Method | Questions scored | Anchored | Seconds |
|---|---|---:|---:|---:|
| base | qwen | 3471 | 0 | 11.6 |
| base | qwen_hybrid_rerank | 3471 | 0 | 2922.5 |
| base | nemotron | 3471 | 0 | 11.3 |
| tok256 | qwen | 7801 | 0 | 20.9 |
| tok256 | qwen_hybrid_rerank | 7801 | 0 | 2277.9 |
| tok256 | nemotron | 7801 | 0 | 19.4 |
| tok512 | qwen | 3605 | 0 | 8.5 |
| tok512 | qwen_hybrid_rerank | 3605 | 0 | 1803.3 |
| tok512 | nemotron | 3605 | 0 | 7.7 |
| tok512ov | qwen | 3865 | 0 | 8.5 |
| tok512ov | qwen_hybrid_rerank | 3865 | 0 | 1942.0 |
| tok512ov | nemotron | 3865 | 0 | 8.0 |
| tok1024 | qwen | 1451 | 0 | 3.1 |
| tok1024 | qwen_hybrid_rerank | 1451 | 0 | 1324.2 |
| tok1024 | nemotron | 1451 | 0 | 2.9 |

`Anchored` is the sample the character-overlap metrics average over, and it is smaller than `Questions scored`: anchoring placed 39 of 64 answers when last measured. It is the same sample for every variant — anchors are placed once, before any chunking — so the columns stay comparable and the shortfall costs statistical power, not validity.

## Recommendation

Best measured cell: `tok1024` with `qwen_hybrid_rerank` at recall@10 0.968 over 1451 questions. That is +0.061 recall@10 against `base` on the same method. Nothing is truncated at qwen's limit. Compared variants: base, tok1024, tok256, tok512, tok512ov. Question counts differ across variants ([1451, 3471, 3605, 3865, 7801]), which is expected when each variant owns its question set — the smaller samples are noisier, not biased. **Ranked on `recall@10` because no span metric was available**, and that metric favours large chunks by construction: anchor the answers first (`uv run python -m src.eval.anchors`) before trusting this ordering.
