# Retrieval Eval Results

## Dataset

Current local eval data:

- 726 page chunks
- 726 chunk summaries
- 3,476 approved questions
- 3,476 relevance labels

Chunk summaries are retrieval representations only. Summary-based methods return
the original chunk text as evidence.

## Full Local Baselines

Run:

```bash
uv run python -m src.eval.evaluate --methods bm25,qwen_chunk,qwen_summary,qwen_chunk_summary
```

| method             | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query |
|--------------------|------:|------:|-------:|-------:|---------:|
| bm25               | 0.544 | 0.668 | 0.705  | 0.596  | 3.0      |
| qwen_chunk         | 0.512 | 0.718 | 0.794  | 0.602  | 3.7      |
| qwen_summary       | 0.442 | 0.666 | 0.743  | 0.536  | 2.9      |
| qwen_chunk_summary | 0.523 | 0.731 | 0.795  | 0.611  | 2.9      |

Read: Qwen chunk+summary is the best local semantic representation overall.
BM25 still wins hit@1, but Qwen wins hit@5, hit@10, MRR, and crosslingual
retrieval.

## English MiniLM Baseline

Run:

```bash
uv run python -m src.eval.evaluate --methods bm25,english_chunk,english_summary,english_chunk_summary
```

| method                | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query |
|-----------------------|------:|------:|-------:|-------:|---------:|
| bm25                  | 0.544 | 0.668 | 0.705  | 0.596  | 2.4      |
| english_chunk         | 0.142 | 0.245 | 0.302  | 0.187  | 1.2      |
| english_summary       | 0.152 | 0.277 | 0.346  | 0.207  | 0.7      |
| english_chunk_summary | 0.161 | 0.283 | 0.351  | 0.214  | 0.6      |

Read: MiniLM is too weak for this multilingual corpus. Summaries help, but not
enough to compete with BM25 or Qwen.

## Azure Sample

Run:

```bash
uv run python -m src.eval.evaluate --limit 300 --methods bm25,english_chunk,english_summary,english_chunk_summary,azure_chunk,azure_summary,azure_chunk_summary
```

| method                | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query |
|-----------------------|------:|------:|-------:|-------:|---------:|
| bm25                  | 0.547 | 0.647 | 0.673  | 0.590  | 3.3      |
| english_chunk         | 0.110 | 0.220 | 0.267  | 0.152  | 8.2      |
| english_summary       | 0.180 | 0.287 | 0.360  | 0.228  | 1.2      |
| english_chunk_summary | 0.133 | 0.267 | 0.317  | 0.188  | 1.7      |
| azure_chunk           | 0.653 | 0.867 | 0.910  | 0.743  | 40.7     |
| azure_summary         | 0.577 | 0.773 | 0.823  | 0.662  | 37.8     |
| azure_chunk_summary   | 0.663 | 0.873 | 0.913  | 0.755  | 44.4     |

Read: Azure chunk+summary is the strongest measured method on this sample, but
it is a remote service and should be compared with cost and quota in mind.

## Composite Methods

Qwen sample:

```bash
uv run python -m src.eval.evaluate --limit 300 --methods bm25,qwen_chunk_summary,qwen_chunk_summary_bm25,qwen_chunk_summary_rerank,qwen_chunk_summary_rerank_hybrid
```

| method                           | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query |
|----------------------------------|------:|------:|-------:|-------:|---------:|
| bm25                             | 0.547 | 0.647 | 0.673  | 0.590  | 3.4      |
| qwen_chunk_summary               | 0.530 | 0.743 | 0.800  | 0.621  | 9.9      |
| qwen_chunk_summary_bm25          | 0.587 | 0.740 | 0.817  | 0.652  | 7.2      |
| qwen_chunk_summary_rerank        | 0.123 | 0.257 | 0.337  | 0.179  | 635.0    |
| qwen_chunk_summary_rerank_hybrid | 0.113 | 0.257 | 0.363  | 0.178  | 624.4    |

Azure sample:

```bash
uv run python -m src.eval.evaluate --limit 300 --methods bm25,azure_chunk_summary,azure_chunk_summary_bm25,azure_chunk_summary_rerank,azure_chunk_summary_rerank_hybrid
```

| method                            | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query |
|-----------------------------------|------:|------:|-------:|-------:|---------:|
| bm25                              | 0.547 | 0.647 | 0.673  | 0.590  | 3.5      |
| azure_chunk_summary               | 0.663 | 0.873 | 0.913  | 0.755  | 42.7     |
| azure_chunk_summary_bm25          | 0.637 | 0.753 | 0.837  | 0.693  | 47.6     |
| azure_chunk_summary_rerank        | 0.107 | 0.243 | 0.340  | 0.167  | 766.6    |
| azure_chunk_summary_rerank_hybrid | 0.107 | 0.253 | 0.360  | 0.169  | 677.4    |

Read: RRF hybrid helps Qwen hit@1/hit@10/MRR on the sample, but hurts Azure.
The current Qwen reranker is harmful on relevance metrics. Its timing should be
re-measured on an idle GPU: this run scored 15,000 query/chunk pairs for 300
questions, and another training process was using the GPU during follow-up
inspection. Do not enable it by default without diagnosing its input format,
prompt behavior, or model fit.

## Current Recommendation

Use `qwen_chunk_summary_bm25` as the best local default candidate and
`azure_chunk_summary` as the best measured remote candidate. Keep BM25 in the
report because it is strong, cheap, and explains many direct lexical wins.
