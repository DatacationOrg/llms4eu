Methods: qwen_hybrid_rerank, qwen_hybrid_agentic, qwen_hybrid_agentic_tools
Scoring unit: chunk
Warmup: 5 queries
Variant: base
Design: shared
Source checkpoints: docs/agentic-deepseek-2026-09-07.md.checkpoint.json
Rendered: 2026-09-07T15:19:39

Progress
- qwen_hybrid_rerank: 495/495 timed queries
- qwen_hybrid_agentic: 495/495 timed queries
- qwen_hybrid_agentic_tools: 495/495 timed queries

Evaluating 495 questions

Overall
| method                    | hit@1     | hit@5     | hit@10    | recall@10 | mrr@10    | hit@20    | recall@20 | mrr@20    | hit@24    | mrr@24    | recall@24 |
|---------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| qwen_hybrid_rerank        | **0.725** | 0.881     | 0.911     | 0.911     | **0.793** | -         | -         | -         | -         | -         | -         |
| qwen_hybrid_agentic       | 0.721     | **0.893** | **0.915** | **0.915** | 0.793     | **0.917** | **0.917** | **0.793** | -         | -         | -         |
| qwen_hybrid_agentic_tools | 0.719     | 0.848     | 0.899     | 0.899     | 0.778     | -         | -         | -         | **0.909** | **0.779** | **0.909** |

Speed
| method                    | seconds | ms/query | queries/query | queries |
|---------------------------|---------|----------|---------------|---------|
| qwen_hybrid_rerank        | 338.04  | 673.3    | 1.00          | 495     |
| qwen_hybrid_agentic       | 2923.25 | 5170.3   | 1.34          | 664     |
| qwen_hybrid_agentic_tools | 3355.59 | 6070.9   | 1.52          | 752     |

hit@5 by category
| method                    | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---------------------------|--------------|-------------|--------------|------------|-------------|
| qwen_hybrid_rerank        | 0.857        | **0.940**   | **0.895**    | 0.972      | 0.752       |
| qwen_hybrid_agentic       | **0.890**    | **0.940**   | 0.886        | **0.981**  | **0.780**   |
| qwen_hybrid_agentic_tools | 0.824        | 0.916       | 0.867        | 0.944      | 0.706       |

Agentic diagnostics (paired at hit@5)

| agent | baseline | queries | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|-------|----------|---------|---------|-----------|----------|-----------------|--------------|-------------|--------|----------|-------------|
| qwen_hybrid_agentic | qwen_hybrid_rerank | 495 | 101 (20.4%) | 99 | 9 | 0.139 | 0.729 | 8 | 2 | - | - |
| qwen_hybrid_agentic_tools | qwen_hybrid_rerank | 495 | 109 (22.0%) | 15 | 95 | 0.037 | 0.695 | 3 | 19 | - | - |

Retry precision is the fraction of retried queries whose first relevant rank improved. Retry recall is the fraction of baseline misses that were retried.

Page tool usage

| agent | questions | tool questions | tool calls | list_sections | search_in_page | recovered@5 | lost@5 | tool precision |
|-------|-----------|----------------|------------|---------------|----------------|-------------|--------|----------------|
| qwen_hybrid_agentic_tools | 495 | 96 (19.4%) | 259 | 2 | 257 | 2 | 19 | 0.021 |

Tool questions are questions where the agent called a page tool. Tool precision is recovered@5 over those questions; per-question action sequences and search terms are in the diagnostics JSON.

Agent versus its own baseline, across runs (hit@5)

| run      | questions | agent                     | baseline           | baseline | agent | delta  |
|----------|-----------|---------------------------|--------------------|----------|-------|--------|
| this run | 495       | qwen_hybrid_agentic       | qwen_hybrid_rerank | 0.881    | 0.893 | +1.2pp |
| this run | 495       | qwen_hybrid_agentic_tools | qwen_hybrid_rerank | 0.881    | 0.848 | -3.2pp |

Absolute scores across runs are NOT comparable: different question samples, embedders and judge models. The delta column is, because each agent is differenced against the baseline measured beside it in the same run.

Not measured in this run
- The `retried` / `rewritten` / `expanded` / `retry precision` / `retry ms` columns read 0 or `-` because the sweep harness does not persist the per-question agent action log its diagnostics are computed from. `recovered` and `lost` come from the rankings and are measured. A zero in the action-log columns is missing data, not an agent that never acted -- `queries/query` above shows it acted.
- No `judge_hit@K` column: equivalence judging was not run (`--judge-equivalence`).
