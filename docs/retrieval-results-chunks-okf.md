Methods: sparse_rerank, qwen4b_hybrid_rerank, nemotron_hybrid_rerank, azure_hybrid_rerank, azure_hybrid_agentic, nemotron_hybrid_agentic
Scoring unit: chunk
Warmup: 5 queries
Output: docs/retrieval-results-chunks-okf.md
Checkpoint: docs/retrieval-results-chunks-okf.md.checkpoint.json
Equivalence judge: azure:DeepSeek-V4-Pro
Equivalence cutoff: 15
Equivalence cache: docs/retrieval-results-chunks-okf.md.equivalence.json
Updated: 2026-07-22T17:52:05
Status: in-progress

Progress
- sparse_rerank: 265/3471 timed queries
- qwen4b_hybrid_rerank: 265/3471 timed queries
- nemotron_hybrid_rerank: 265/3471 timed queries
- azure_hybrid_rerank: 265/3471 timed queries
- azure_hybrid_agentic: 264/3471 timed queries
- nemotron_hybrid_agentic: 264/3471 timed queries
- aligned_for_scoring: 264/3471

Evaluating 264 questions

Overall
| method                  | hit@1     | hit@5     | hit@10    | recall@10 | mrr@10    | hit@15    | recall@15 | mrr@15    | judge_hit@15 |
|-------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|--------------|
| sparse_rerank           | 0.640     | 0.742     | 0.746     | 0.746     | 0.682     | -         | -         | -         | 0.780        |
| qwen4b_hybrid_rerank    | 0.652     | 0.765     | 0.773     | 0.773     | 0.699     | -         | -         | -         | 0.811        |
| nemotron_hybrid_rerank  | 0.716     | 0.848     | 0.864     | 0.864     | 0.770     | -         | -         | -         | 0.890        |
| azure_hybrid_rerank     | **0.746** | 0.898     | **0.939** | **0.939** | 0.810     | -         | -         | -         | **0.955**    |
| azure_hybrid_agentic    | **0.746** | **0.917** | 0.932     | 0.932     | **0.814** | **0.932** | **0.932** | **0.814** | **0.955**    |
| nemotron_hybrid_agentic | 0.720     | 0.864     | 0.875     | 0.875     | 0.776     | 0.875     | 0.875     | 0.776     | 0.902        |

Speed

| method                  | seconds | ms/query | queries/query | queries | chunk expansions |
|-------------------------|---------|----------|---------------|---------|------------------|
| sparse_rerank           | 196.55  | 744.5    | 1.00          | 265     | 0                |
| qwen4b_hybrid_rerank    | 152.30  | 576.9    | 1.00          | 265     | 0                |
| nemotron_hybrid_rerank  | 195.24  | 739.6    | 1.00          | 265     | 0                |
| azure_hybrid_rerank     | 275.68  | 1044.2   | 1.00          | 265     | 0                |
| azure_hybrid_agentic    | 2574.38 | 9751.4   | 1.31          | 347     | 2                |
| nemotron_hybrid_agentic | 2261.21 | 8565.2   | 1.38          | 364     | 6                |

hit@5 by category

| method                  | crosslingual | direct_long | direct_short | vague_long | vague_short |
|-------------------------|--------------|-------------|--------------|------------|-------------|
| sparse_rerank           | 0.426        | 0.909       | 0.746        | 0.933      | 0.667       |
| qwen4b_hybrid_rerank    | 0.574        | 0.909       | 0.763        | 0.950      | 0.611       |
| nemotron_hybrid_rerank  | 0.830        | 0.909       | 0.847        | 0.967      | 0.685       |
| azure_hybrid_rerank     | 0.851        | **0.932**   | 0.915        | **1.000**  | 0.778       |
| azure_hybrid_agentic    | **0.894**    | **0.932**   | **0.932**    | **1.000**  | **0.815**   |
| nemotron_hybrid_agentic | 0.851        | **0.932**   | 0.847        | 0.967      | 0.722       |
