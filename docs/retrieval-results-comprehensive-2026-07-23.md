Methods: sparse_rerank, qwen4b_hybrid_rerank, nemotron_hybrid_rerank, azure_hybrid_rerank, azure_hybrid_agentic, nemotron_hybrid_agentic
Scoring unit: chunk
Warmup: 5 queries
Output: docs/retrieval-results-comprehensive-2026-07-23.md
Checkpoint: docs/retrieval-results-comprehensive-2026-07-23.md.checkpoint.json
Equivalence judge: azure:DeepSeek-V4-Pro
Equivalence cutoff: 15
Equivalence cache: docs/retrieval-results-comprehensive-2026-07-23.md.equivalence.json
Updated: 2026-07-23T14:46:18
Status: in-progress

Progress
- sparse_rerank: 200/3471 timed queries
- qwen4b_hybrid_rerank: 200/3471 timed queries
- nemotron_hybrid_rerank: 200/3471 timed queries
- azure_hybrid_rerank: 200/3471 timed queries
- azure_hybrid_agentic: 199/3471 timed queries
- nemotron_hybrid_agentic: 199/3471 timed queries
- aligned_for_scoring: 199/3471

Evaluating 199 questions

Overall
| method                  | hit@1     | hit@5     | hit@10    | recall@10 | mrr@10    | judge_hit@15 |
|-------------------------|-----------|-----------|-----------|-----------|-----------|--------------|
| sparse_rerank           | 0.638     | 0.729     | 0.734     | 0.734     | 0.676     | 0.779        |
| qwen4b_hybrid_rerank    | 0.653     | 0.754     | 0.759     | 0.759     | 0.695     | 0.814        |
| nemotron_hybrid_rerank  | 0.729     | 0.849     | 0.869     | 0.869     | 0.780     | 0.899        |
| azure_hybrid_rerank     | **0.759** | 0.899     | **0.945** | **0.945** | 0.819     | **0.965**    |
| azure_hybrid_agentic    | 0.754     | **0.920** | **0.945** | **0.945** | **0.821** | **0.965**    |
| nemotron_hybrid_agentic | 0.729     | 0.864     | 0.879     | 0.879     | 0.783     | 0.905        |

Speed

| method                  | seconds | ms/query | queries/query | queries | chunk expansions |
|-------------------------|---------|----------|---------------|---------|------------------|
| sparse_rerank           | 183.52  | 922.2    | 1.01          | 200     | 0                |
| qwen4b_hybrid_rerank    | 143.51  | 721.2    | 1.01          | 200     | 0                |
| nemotron_hybrid_rerank  | 181.43  | 911.7    | 1.01          | 200     | 0                |
| azure_hybrid_rerank     | 184.29  | 926.1    | 1.01          | 200     | 0                |
| azure_hybrid_agentic    | 2279.16 | 11453.1  | 1.28          | 255     | 3                |
| nemotron_hybrid_agentic | 1884.46 | 9469.6   | 1.35          | 268     | 6                |

hit@5 by category

| method                  | crosslingual | direct_long | direct_short | vague_long | vague_short |
|-------------------------|--------------|-------------|--------------|------------|-------------|
| sparse_rerank           | 0.389        | 0.886       | 0.732        | 0.935      | 0.659       |
| qwen4b_hybrid_rerank    | 0.556        | 0.886       | 0.732        | 0.957      | 0.610       |
| nemotron_hybrid_rerank  | 0.833        | 0.886       | 0.854        | 0.978      | 0.683       |
| azure_hybrid_rerank     | 0.833        | **0.914**   | **0.902**    | **1.000**  | 0.829       |
| azure_hybrid_agentic    | **0.889**    | **0.914**   | **0.902**    | **1.000**  | **0.878**   |
| nemotron_hybrid_agentic | **0.889**    | 0.886       | 0.854        | 0.978      | 0.707       |

Agentic diagnostics (paired at hit@5)

| agent | baseline | queries | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|-------|----------|---------|---------|-----------|----------|-----------------|--------------|-------------|--------|----------|-------------|
| azure_hybrid_agentic | azure_hybrid_rerank | 199 | 34 (17.1%) | 34 | 3 | 0.206 | 0.700 | 4 | 0 | 21756.3 | 9330.0 |
| nemotron_hybrid_agentic | nemotron_hybrid_rerank | 199 | 41 (20.6%) | 41 | 6 | 0.098 | 0.733 | 4 | 1 | 19697.7 | 6815.5 |

Retry precision is the fraction of retried queries whose first relevant rank improved. Retry recall is the fraction of baseline misses that were retried.
