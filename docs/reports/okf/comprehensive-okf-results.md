# Comprehensive OKF and Chunk Retrieval Results

This document combines the results from [okf_benchmark.md](okf_benchmark.md) and
[retrieval-results-chunks-okf.md](../retrieval/retrieval-results-comprehensive-2026-09-07.md). Empty cells
mean that the source report did not provide the metric.

The two runs should not be compared as if they were one evaluation:

- **OKF benchmark:** 396 questions; includes `okf` and `okf_search`.
- **Chunk/agentic benchmark:** 264 aligned questions from an in-progress
  3,471-question run; chunk scoring, five warmup queries, and an optional
  evidence-equivalence judge at rank 15.
- Identically named methods are retained as separate rows because their values
  come from different runs and question subsets.

## Overall retrieval quality

| run | method | questions | hit@1 | hit@5 | hit@10 | recall@10 | mrr@10 | hit@15 | recall@15 | mrr@15 | judge_hit@15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OKF benchmark | sparse_rerank | 396 | 0.742 | 0.841 | 0.851 | 0.851 | 0.786 |  |  |  |  |
| OKF benchmark | qwen_hybrid_rerank | 396 | 0.754 | 0.873 | 0.886 | 0.886 | 0.804 |  |  |  |  |
| OKF benchmark | okf | 396 | 0.473 | 0.501 | 0.501 | 0.501 | 0.485 |  |  |  |  |
| OKF benchmark | okf_search | 396 | 0.570 | 0.605 | 0.605 | 0.605 | 0.584 |  |  |  |  |
| Chunk/agentic benchmark | sparse_rerank | 264 | 0.640 | 0.742 | 0.746 | 0.746 | 0.682 |  |  |  | 0.780 |
| Chunk/agentic benchmark | qwen4b_hybrid_rerank | 264 | 0.652 | 0.765 | 0.773 | 0.773 | 0.699 |  |  |  | 0.811 |
| Chunk/agentic benchmark | nemotron_hybrid_rerank | 264 | 0.716 | 0.848 | 0.864 | 0.864 | 0.770 |  |  |  | 0.890 |
| Chunk/agentic benchmark | azure_hybrid_rerank | 264 | 0.746 | 0.898 | 0.939 | 0.939 | 0.810 |  |  |  | 0.955 |
| Chunk/agentic benchmark | azure_hybrid_agentic | 264 | 0.746 | 0.917 | 0.932 | 0.932 | 0.814 | 0.932 | 0.932 | 0.814 | 0.955 |
| Chunk/agentic benchmark | nemotron_hybrid_agentic | 264 | 0.720 | 0.864 | 0.875 | 0.875 | 0.776 | 0.875 | 0.875 | 0.776 | 0.902 |

## Speed and retrieval effort

The `questions` column is the number of evaluated questions. The `queries`
column counts retrieval or navigation actions. The chunk benchmark had 265 timed
questions for non-agentic speed measurements and 264 aligned questions for
quality scoring.

| run | method | questions | seconds | ms/query | queries/query | queries | chunk expansions |
|---|---|---:|---:|---:|---:|---:|---:|
| OKF benchmark | sparse_rerank | 396 | 531.29 | 1345.0 | 1.00 | 396 |  |
| OKF benchmark | qwen_hybrid_rerank | 396 | 325.16 | 823.2 | 1.00 | 396 |  |
| OKF benchmark | okf | 396 | 5385.05 | 13633.0 | 4.40 | 1737 |  |
| OKF benchmark | okf_search | 396 | 4884.96 | 12367.0 | 3.72 | 1468 |  |
| Chunk/agentic benchmark | sparse_rerank | 265 | 196.55 | 744.5 | 1.00 | 265 | 0 |
| Chunk/agentic benchmark | qwen4b_hybrid_rerank | 265 | 152.30 | 576.9 | 1.00 | 265 | 0 |
| Chunk/agentic benchmark | nemotron_hybrid_rerank | 265 | 195.24 | 739.6 | 1.00 | 265 | 0 |
| Chunk/agentic benchmark | azure_hybrid_rerank | 265 | 275.68 | 1044.2 | 1.00 | 265 | 0 |
| Chunk/agentic benchmark | azure_hybrid_agentic | 264 | 2574.38 | 9751.4 | 1.31 | 347 | 2 |
| Chunk/agentic benchmark | nemotron_hybrid_agentic | 264 | 2261.21 | 8565.2 | 1.38 | 364 | 6 |

## Hit@5 by category

| run | method | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---|---:|---:|---:|---:|---:|
| OKF benchmark | sparse_rerank | 0.710 | 0.945 | 0.831 | 0.905 | 0.802 |
| OKF benchmark | qwen_hybrid_rerank | 0.899 | 0.945 | 0.892 | 0.929 | 0.721 |
| OKF benchmark | okf | 0.435 | 0.548 | 0.566 | 0.583 | 0.372 |
| OKF benchmark | okf_search | 0.594 | 0.699 | 0.687 | 0.643 | 0.419 |
| Chunk/agentic benchmark | sparse_rerank | 0.426 | 0.909 | 0.746 | 0.933 | 0.667 |
| Chunk/agentic benchmark | qwen4b_hybrid_rerank | 0.574 | 0.909 | 0.763 | 0.950 | 0.611 |
| Chunk/agentic benchmark | nemotron_hybrid_rerank | 0.830 | 0.909 | 0.847 | 0.967 | 0.685 |
| Chunk/agentic benchmark | azure_hybrid_rerank | 0.851 | 0.932 | 0.915 | 1.000 | 0.778 |
| Chunk/agentic benchmark | azure_hybrid_agentic | 0.894 | 0.932 | 0.932 | 1.000 | 0.815 |
| Chunk/agentic benchmark | nemotron_hybrid_agentic | 0.851 | 0.932 | 0.847 | 0.967 | 0.722 |

## Source metadata

| field | OKF benchmark | Chunk/agentic benchmark |
|---|---|---|
| Source report | [okf_benchmark.md](okf_benchmark.md) | [retrieval-results-chunks-okf.md](../retrieval/retrieval-results-comprehensive-2026-09-07.md) |
| Evaluation status |  | In progress |
| Evaluated/aligned questions | 396 | 264 |
| Planned questions |  | 3,471 |
| Warmup queries |  | 5 |
| Scoring unit |  | Chunk |
| Equivalence judge |  | Azure DeepSeek-V4-Pro |
| Equivalence cutoff |  | 15 |
| Last source update |  | 2026-07-22 17:52:05 |
