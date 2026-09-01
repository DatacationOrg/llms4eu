Methods: qwen_hybrid_rerank, qwen_hybrid_agentic, qwen_hybrid_agentic_tools, dci
Scoring unit: chunk
Warmup: 5 queries
Variant: base
Design: shared
Source checkpoints: docs/agentic-tools-2026-08-31.md.checkpoint.json, docs/agentic-dci-2026-09-01.md.checkpoint.json
Rendered: 2026-09-01T12:45:26

Progress
- qwen_hybrid_rerank: 495/495 timed queries
- qwen_hybrid_agentic: 495/495 timed queries
- qwen_hybrid_agentic_tools: 495/495 timed queries
- dci: 495/495 timed queries

Evaluating 495 questions

Overall
| method                    | hit@1     | hit@5     | hit@10    | recall@10 | mrr@10    | hit@20    | recall@20 | mrr@20    | hit@24    | mrr@24    | recall@24 |
|---------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| qwen_hybrid_rerank        | 0.723     | 0.881     | 0.907     | 0.907     | 0.791     | -         | -         | -         | -         | -         | -         |
| qwen_hybrid_agentic       | **0.725** | **0.889** | **0.915** | **0.915** | **0.795** | **0.917** | **0.917** | **0.795** | -         | -         | -         |
| qwen_hybrid_agentic_tools | 0.719     | 0.879     | 0.903     | 0.903     | 0.787     | -         | -         | -         | **0.907** | **0.787** | **0.907** |
| dci                       | 0.576     | 0.697     | 0.723     | 0.723     | 0.632     | -         | -         | -         | -         | -         | -         |

Speed
| method                    | seconds  | ms/query | queries/query | queries |
|---------------------------|----------|----------|---------------|---------|
| qwen_hybrid_rerank        | 339.02   | 675.4    | 1.00          | 495     |
| qwen_hybrid_agentic       | 3367.13  | 6062.1   | 1.21          | 597     |
| qwen_hybrid_agentic_tools | 3613.43  | 6572.2   | 1.25          | 617     |
| dci                       | 11051.16 | 22084.3  | 5.85          | 2898    |

hit@5 by category
| method                    | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---------------------------|--------------|-------------|--------------|------------|-------------|
| qwen_hybrid_rerank        | 0.857        | 0.940       | **0.905**    | **0.963**  | 0.752       |
| qwen_hybrid_agentic       | **0.901**    | **0.952**   | 0.895        | 0.953      | **0.761**   |
| qwen_hybrid_agentic_tools | 0.857        | 0.940       | 0.895        | **0.963**  | 0.752       |
| dci                       | 0.352        | 0.892       | 0.800        | 0.813      | 0.624       |

Agentic diagnostics (paired at hit@5)

| agent | baseline | queries | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|-------|----------|---------|---------|-----------|----------|-----------------|--------------|-------------|--------|----------|-------------|
| qwen_hybrid_agentic | qwen_hybrid_rerank | 495 | 0 (0.0%) | 0 | 0 | - | 0.000 | 9 | 5 | - | - |
| qwen_hybrid_agentic_tools | qwen_hybrid_rerank | 495 | 0 (0.0%) | 0 | 0 | - | 0.000 | 2 | 3 | - | - |

Retry precision is the fraction of retried queries whose first relevant rank improved. Retry recall is the fraction of baseline misses that were retried.

Agent versus its own baseline, across runs (hit@5)

| run                                        | questions | agent                     | baseline               | baseline | agent | delta  |
|--------------------------------------------|-----------|---------------------------|------------------------|----------|-------|--------|
| retrieval-results-comprehensive-2026-07-27 | 847       | embed_v4_hybrid_agentic   | embed_v4_hybrid_rerank | 0.903    | 0.911 | +0.8pp |
| retrieval-results-comprehensive-2026-07-27 | 847       | nemotron_hybrid_agentic   | nemotron_hybrid_rerank | 0.874    | 0.884 | +1.0pp |
| this run                                   | 495       | qwen_hybrid_agentic       | qwen_hybrid_rerank     | 0.881    | 0.889 | +0.8pp |
| this run                                   | 495       | qwen_hybrid_agentic_tools | qwen_hybrid_rerank     | 0.881    | 0.879 | -0.2pp |

Absolute scores across runs are NOT comparable: different question samples, embedders and judge models. The delta column is, because each agent is differenced against the baseline measured beside it in the same run.

Not measured in this run
- `base|qwen_hybrid_rerank` was measured in more than one checkpoint; the first cell seen was kept and the later one ignored.
- The `retried` / `rewritten` / `expanded` / `retry precision` / `retry ms` columns read 0 or `-` because the sweep harness does not persist the per-question agent action log its diagnostics are computed from. `recovered` and `lost` come from the rankings and are measured. A zero in the action-log columns is missing data, not an agent that never acted -- `queries/query` above shows it acted.
- No `judge_hit@K` column: equivalence judging was not run (`--judge-equivalence`).
