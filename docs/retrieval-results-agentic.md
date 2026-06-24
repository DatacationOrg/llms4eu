Methods: sparse, sparse_rerank, qwen, qwen_hybrid, qwen_rerank, qwen_hybrid_rerank, qwen4b, qwen4b_hybrid, qwen4b_rerank, qwen4b_hybrid_rerank, qwen_agentic, qwen_hybrid_agentic
Warmup: 5 queries
Output: docs/retrieval-results-agentic.md
Checkpoint: docs/retrieval-results-agentic.md.checkpoint.json
Updated: 2026-06-23T19:11:19
Status: in-progress

Progress
- sparse: 910/3471 timed queries
- sparse_rerank: 910/3471 timed queries
- qwen: 910/3471 timed queries
- qwen_hybrid: 910/3471 timed queries
- qwen_rerank: 910/3471 timed queries
- qwen_hybrid_rerank: 910/3471 timed queries
- qwen4b: 910/3471 timed queries
- qwen4b_hybrid: 910/3471 timed queries
- qwen4b_rerank: 910/3471 timed queries
- qwen4b_hybrid_rerank: 910/3471 timed queries
- qwen_agentic: 909/3471 timed queries
- qwen_hybrid_agentic: 909/3471 timed queries
- aligned_for_scoring: 909/3471

Evaluating 909 questions

Evaluating 909 questions

Overall
| method               | hit@1 | hit@5 | hit@10 | recall@10 | mrr@10 |
|----------------------|-------|-------|--------|-----------|--------|
| sparse               | 0.554 | 0.674 | 0.708  | 0.708     | 0.605  |
| sparse_rerank        | 0.664 | 0.762 | 0.768  | 0.768     | 0.707  |
| qwen                 | 0.119 | 0.189 | 0.219  | 0.219     | 0.149  |
| qwen_hybrid          | 0.155 | 0.311 | 0.546  | 0.546     | 0.235  |
| qwen_rerank          | 0.234 | 0.275 | 0.279  | 0.279     | 0.252  |
| qwen_hybrid_rerank   | 0.670 | 0.769 | 0.779  | 0.779     | 0.713  |
| qwen4b               | 0.139 | 0.231 | 0.268  | 0.268     | 0.178  |
| qwen4b_hybrid        | 0.183 | 0.372 | 0.602  | 0.602     | 0.274  |
| qwen4b_rerank        | 0.285 | 0.327 | 0.334  | 0.334     | 0.303  |
| qwen4b_hybrid_rerank | **0.690** | **0.783** | **0.793** | **0.793** | **0.731** |
| qwen_agentic         | 0.121 | 0.205 | 0.244  | 0.244     | 0.157  |
| qwen_hybrid_agentic  | 0.161 | 0.327 | 0.619  | 0.619     | 0.250  |

Speed

| method               | seconds | ms/query | queries/query | queries |
|----------------------|---------|----------|---------------|---------|
| sparse               | 1.81    | 2.0      | 1.00          | 910     |
| sparse_rerank        | 663.81  | 730.3    | 1.00          | 910     |
| qwen                 | 14.88   | 16.4     | 1.00          | 910     |
| qwen_hybrid          | 14.41   | 15.9     | 1.00          | 910     |
| qwen_rerank          | 400.87  | 441.0    | 1.00          | 910     |
| qwen_hybrid_rerank   | 490.63  | 539.7    | 1.00          | 910     |
| qwen4b               | 56.40   | 62.0     | 1.00          | 910     |
| qwen4b_hybrid        | 14.87   | 16.4     | 1.00          | 910     |
| qwen4b_rerank        | 456.07  | 501.7    | 1.00          | 910     |
| qwen4b_hybrid_rerank | 523.84  | 576.3    | 1.00          | 910     |
| qwen_agentic         | 7309.67 | 8041.4   | 2.51          | 2285    |
| qwen_hybrid_agentic  | 5177.53 | 5695.9   | 1.88          | 1707    |

hit@5 by category

| method               | crosslingual | direct_long | direct_short | vague_long | vague_short |
|----------------------|--------------|-------------|--------------|------------|-------------|
| sparse               | 0.262        | 0.885       | 0.763        | 0.834      | 0.582       |
| sparse_rerank        | 0.470        | 0.923       | 0.812        | 0.882      | **0.693**   |
| qwen                 | 0.207        | 0.268       | 0.199        | 0.171      | 0.106       |
| qwen_hybrid          | 0.232        | 0.432       | 0.333        | 0.342      | 0.212       |
| qwen_rerank          | 0.317        | 0.339       | 0.274        | 0.278      | 0.175       |
| qwen_hybrid_rerank   | 0.518        | **0.934**   | **0.839**    | 0.888      | 0.640       |
| qwen4b               | 0.256        | 0.273       | 0.210        | 0.267      | 0.153       |
| qwen4b_hybrid        | 0.311        | 0.459       | 0.387        | 0.433      | 0.265       |
| qwen4b_rerank        | 0.335        | 0.377       | 0.312        | 0.369      | 0.243       |
| qwen4b_hybrid_rerank | **0.549**    | **0.934**   | 0.823        | **0.920**  | 0.667       |
| qwen_agentic         | 0.213        | 0.273       | 0.215        | 0.187      | 0.138       |
| qwen_hybrid_agentic  | 0.244        | 0.448       | 0.339        | 0.374      | 0.222       |
