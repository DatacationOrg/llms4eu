Methods: sparse_rerank, qwen4b_hybrid_rerank, nemotron_hybrid_rerank, azure_hybrid_rerank, azure_hybrid_agentic, nemotron_hybrid_agentic
Scoring unit: chunk
Warmup: 5 queries
Output: docs/retrieval-results-chunks-okf.md
Checkpoint: docs/retrieval-results-chunks-okf.md.checkpoint.json
Equivalence judge: azure:DeepSeek-V4-Pro
Equivalence cutoff: 15
Equivalence cache: docs/retrieval-results-chunks-okf.md.equivalence.json
Updated: 2026-07-23T12:23:39
Status: in-progress

Progress
- sparse_rerank: 261/3471 timed queries
- qwen4b_hybrid_rerank: 261/3471 timed queries
- nemotron_hybrid_rerank: 261/3471 timed queries
- azure_hybrid_rerank: 261/3471 timed queries
- azure_hybrid_agentic: 260/3471 timed queries
- nemotron_hybrid_agentic: 260/3471 timed queries
- aligned_for_scoring: 260/3471

Evaluating 260 questions

Overall
| method                  | hit@1     | hit@5     | hit@10    | recall@10 | mrr@10    | hit@15    | recall@15 | mrr@15    | judge_hit@15 |
|-------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|--------------|
| sparse_rerank           | 0.642     | 0.746     | 0.750     | 0.750     | 0.685     | -         | -         | -         | 0.785        |
| qwen4b_hybrid_rerank    | 0.654     | 0.769     | 0.777     | 0.777     | 0.702     | -         | -         | -         | 0.815        |
| nemotron_hybrid_rerank  | 0.719     | 0.850     | 0.865     | 0.865     | 0.773     | -         | -         | -         | 0.892        |
| azure_hybrid_rerank     | **0.750** | 0.900     | **0.942** | **0.942** | 0.814     | -         | -         | -         | **0.958**    |
| azure_hybrid_agentic    | **0.750** | **0.919** | 0.935     | 0.935     | **0.817** | **0.935** | **0.935** | **0.817** | **0.958**    |
| nemotron_hybrid_agentic | 0.723     | 0.865     | 0.877     | 0.877     | 0.779     | 0.877     | 0.877     | 0.779     | 0.900        |

Speed

| method                  | seconds | ms/query | queries/query | queries | chunk expansions |
|-------------------------|---------|----------|---------------|---------|------------------|
| sparse_rerank           | 195.13  | 750.5    | 1.00          | 261     | 0                |
| qwen4b_hybrid_rerank    | 150.27  | 578.0    | 1.00          | 261     | 0                |
| nemotron_hybrid_rerank  | 192.57  | 740.7    | 1.00          | 261     | 0                |
| azure_hybrid_rerank     | 270.82  | 1041.6   | 1.00          | 261     | 0                |
| azure_hybrid_agentic    | 2519.05 | 9688.6   | 1.30          | 339     | 2                |
| nemotron_hybrid_agentic | 2226.39 | 8563.0   | 1.38          | 358     | 6                |

hit@5 by category

| method                  | crosslingual | direct_long | direct_short | vague_long | vague_short |
|-------------------------|--------------|-------------|--------------|------------|-------------|
| sparse_rerank           | 0.426        | 0.909       | 0.746        | 0.932      | 0.686       |
| qwen4b_hybrid_rerank    | 0.574        | 0.909       | 0.763        | 0.949      | 0.627       |
| nemotron_hybrid_rerank  | 0.830        | 0.909       | 0.847        | 0.966      | 0.686       |
| azure_hybrid_rerank     | 0.851        | **0.932**   | 0.915        | **1.000**  | 0.784       |
| azure_hybrid_agentic    | **0.894**    | **0.932**   | **0.932**    | **1.000**  | **0.824**   |
| nemotron_hybrid_agentic | 0.851        | **0.932**   | 0.847        | 0.966      | 0.725       |

Agentic diagnostics (paired at hit@5)

| agent | baseline | queries | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|-------|----------|---------|---------|-----------|----------|-----------------|--------------|-------------|--------|----------|-------------|
| azure_hybrid_agentic | azure_hybrid_rerank | 260 | 47 (18.1%) | 47 | 2 | 0.149 | 0.731 | 5 | 0 | - | - |
| nemotron_hybrid_agentic | nemotron_hybrid_rerank | 260 | 55 (21.2%) | 54 | 5 | 0.091 | 0.795 | 4 | 0 | - | - |

Retry precision is the fraction of retried queries whose first relevant rank improved. Retry recall is the fraction of baseline misses that were retried.
