Methods: sparse_rerank, qwen4b_hybrid_rerank, nemotron_hybrid_rerank, embed_v4_hybrid_rerank, embed_v4_hybrid_agentic, nemotron_hybrid_agentic, okf
Scoring unit: per method (RAG: chunk; OKF: concept)
Warmup: 5 queries
Output: docs/retrieval-results-comprehensive-2026-07-27.md
Checkpoint: docs/retrieval-results-comprehensive-2026-07-27.md.checkpoint.json
Equivalence judge: azure:DeepSeek-V4-Pro
Equivalence cutoff: 15
Equivalence cache: docs/retrieval-results-comprehensive-2026-07-27.md.equivalence.json
Updated: 2026-07-27T23:46:29
Status: in-progress

Progress
- sparse_rerank: 848/3471 timed queries
- qwen4b_hybrid_rerank: 848/3471 timed queries
- nemotron_hybrid_rerank: 848/3471 timed queries
- embed_v4_hybrid_rerank: 848/3471 timed queries
- embed_v4_hybrid_agentic: 847/3471 timed queries
- nemotron_hybrid_agentic: 847/3471 timed queries
- okf: 847/3471 timed queries
- aligned_for_scoring: 847/3471

OKF coverage: 176 pages in 128 concepts
OKF failures: okf=0

Evaluating 847 questions

Overall
| method                  | hit@1     | hit@5     | hit@10    | recall@10 | mrr@10    | hit@20    | recall@20 | mrr@20    | judge_hit@15 |
|-------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|--------------|
| sparse_rerank           | 0.664     | 0.764     | 0.770     | 0.770     | 0.707     | -         | -         | -         | 0.806        |
| qwen4b_hybrid_rerank    | 0.688     | 0.784     | 0.793     | 0.793     | 0.731     | -         | -         | -         | 0.831        |
| nemotron_hybrid_rerank  | 0.737     | 0.874     | 0.889     | 0.889     | 0.796     | -         | -         | -         | 0.914        |
| embed_v4_hybrid_rerank  | 0.754     | 0.903     | 0.936     | 0.936     | 0.820     | -         | -         | -         | 0.952        |
| embed_v4_hybrid_agentic | **0.760** | **0.911** | **0.947** | **0.947** | **0.828** | **0.948** | **0.948** | **0.828** | **0.961**    |
| nemotron_hybrid_agentic | 0.743     | 0.884     | 0.903     | 0.903     | 0.805     | 0.904     | 0.904     | 0.805     | 0.928        |
| okf                     | 0.494     | 0.532     | 0.534     | 0.534     | 0.511     | -         | -         | -         | 0.543        |

Speed

| method                  | seconds  | ms/query | queries/query | queries | chunk expansions |
|-------------------------|----------|----------|---------------|---------|------------------|
| sparse_rerank           | 628.52   | 742.1    | 1.00          | 848     | 0                |
| qwen4b_hybrid_rerank    | 486.34   | 574.2    | 1.00          | 848     | 0                |
| nemotron_hybrid_rerank  | 626.43   | 739.6    | 1.00          | 848     | 0                |
| embed_v4_hybrid_rerank  | 760.46   | 897.8    | 1.00          | 848     | 0                |
| embed_v4_hybrid_agentic | 4372.26  | 5162.0   | 1.30          | 1105    | 14               |
| nemotron_hybrid_agentic | 4014.29  | 4739.4   | 1.33          | 1125    | 20               |
| okf                     | 11001.20 | 12988.4  | 3.97          | 3366    | 0                |

hit@5 by category

| method                  | crosslingual | direct_long | direct_short | vague_long | vague_short |
|-------------------------|--------------|-------------|--------------|------------|-------------|
| sparse_rerank           | 0.503        | 0.916       | 0.802        | 0.878      | 0.692       |
| qwen4b_hybrid_rerank    | 0.564        | 0.928       | 0.814        | 0.924      | 0.670       |
| nemotron_hybrid_rerank  | 0.879        | 0.946       | 0.876        | 0.948      | 0.731       |
| embed_v4_hybrid_rerank  | 0.886        | 0.964       | 0.910        | **0.983**  | 0.780       |
| embed_v4_hybrid_agentic | **0.906**    | **0.970**   | **0.915**    | **0.983**  | **0.791**   |
| nemotron_hybrid_agentic | 0.893        | 0.958       | 0.876        | 0.953      | 0.753       |
| okf                     | 0.000        | 0.000       | 0.000        | 0.000      | 0.000       |

Agentic diagnostics (paired at hit@5)

| agent | baseline | queries | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|-------|----------|---------|---------|-----------|----------|-----------------|--------------|-------------|--------|----------|-------------|
| embed_v4_hybrid_agentic | embed_v4_hybrid_rerank | 847 | 157 (18.5%) | 154 | 13 | 0.153 | 0.683 | 11 | 4 | 11171.8 | 3794.6 |
| nemotron_hybrid_agentic | nemotron_hybrid_rerank | 847 | 166 (19.6%) | 163 | 18 | 0.120 | 0.729 | 11 | 2 | 11172.0 | 3171.4 |

Retry precision is the fraction of retried queries whose first relevant rank improved. Retry recall is the fraction of baseline misses that were retried.
