# Agentic retrieval by judge model: gpt-oss, gemma, DeepSeek

All cells: 495 questions (first 500 by id, 5 warm-up), `base` chunking, shared design,
`qwen` embedder + Qwen3-Reranker-0.6B first stage, hit cutoff 5 for the paired
diagnostics. The control (`qwen_hybrid_rerank`) uses no judge and reproduces at
hit@5 0.881 in every run, which is how the rows are known to be comparable.

## Headline

| method | judge | hit@1 | hit@5 | hit@10 | mrr@10 | s/query | judge calls/query |
|---|---|---|---|---|---|---|---|
| `qwen_hybrid_rerank` (control, no judge) | — | 0.723 | 0.881 | 0.907 | 0.791 | 0.7 | 0 |
| `qwen_hybrid_agentic` (retry loop) | gpt-oss:20b | 0.725 | 0.889 | 0.915 | 0.795 | 6.1 | 1.21 |
| | gemma4:31b | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ |
| | DeepSeek-V4-Pro (Azure) | 0.721 | **0.893** | 0.915 | 0.793 | 5.2 | 1.34 |
| `qwen_hybrid_agentic_tools` (page tools) | gpt-oss:20b | 0.719 | 0.879 | 0.903 | 0.787 | 6.6 | 1.25 |
| | gemma4:31b | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ |
| | DeepSeek-V4-Pro (Azure) | 0.719 | **0.849** | 0.899 | 0.778 | 6.1 | 1.52 |
| `dci` (corpus tools, grep-style) | gpt-oss:20b | 0.576 | 0.697 | 0.723 | 0.632 | 22.1 | 5.85 |
| | gemma4:31b | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ |
| | DeepSeek-V4-Pro (Azure) | skipped² | skipped² | skipped² | skipped² | — | — |

Run-to-run noise on the control is about 1 point of hit@5 (0.8808 to 0.8848 over five
reproductions), so differences under a point are not established.

## Paired diagnostics against the control (hit@5)

| method | judge | retried | rewritten | expanded | recovered | lost | retry recall |
|---|---|---|---|---|---|---|---|
| retry loop | gpt-oss | not recorded³ | not recorded³ | not recorded³ | 9 | 5 | not recorded³ |
| retry loop | DeepSeek | 101 (20%) | 99 | 9 | 8 | 2 | 73% |
| page tools | gpt-oss | not recorded³ | not recorded³ | not recorded³ | 2 | 3 | not recorded³ |
| page tools | DeepSeek | 109 (22%) | 15 | 95 | 3 | 19 | 69% |

## Page tool usage (DeepSeek only; gpt-oss not recorded³)

| tool questions | tool calls | `search_in_page` | `list_sections` | recovered on tool questions | lost on tool questions | tool precision |
|---|---|---|---|---|---|---|
| 96 of 495 (19%) | 259 | 257 | 2 | 2 | 19 | 2% |

## hit@5 by question type

| method | judge | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---|---|---|---|---|---|
| control | — | 0.857 | 0.940 | 0.895 | 0.972 | 0.752 |
| retry loop | gpt-oss | 0.901 | 0.952 | 0.895 | 0.953 | 0.761 |
| retry loop | DeepSeek | 0.890 | 0.940 | 0.886 | 0.981 | 0.780 |
| page tools | gpt-oss | 0.857 | 0.940 | 0.895 | 0.963 | 0.752 |
| page tools | DeepSeek | 0.824 | 0.916 | 0.867 | 0.944 | 0.706 |
| dci | gpt-oss | 0.352 | 0.892 | 0.800 | 0.813 | 0.624 |

## Notes

¹ **gemma4:31b: no agent cell ever completed.** Two full runs (2026-09-01, 2026-09-03/04)
each finished only the control cell (hit@5 0.883 and 0.881, so the judge was not the
issue there). The first died after 9.5 h with 136 judge failures: gemma's thinking
consumed the 1,024-token output cap on real 4k to 11k-token prompts, leaving empty
output; the retriever treated each as "insufficient" and widened the search, which
issued more failing calls. The second run, with the fixed configuration (thinking off,
function calling, 100% on 150 real prompts, zero judge failures in its first hour), was
killed twice by the server ending the user session at logout, and was then stopped on
request. Gemma is technically usable as a judge but costs 28 GB of a shared GPU and about
9 s per call against DeepSeek's 1.5 s; it is parked, not disproven.

² **DCI with DeepSeek was skipped on request** to free the machine. Its gpt-oss failure
is structural (a keyword-only agent cannot match a Slovenian answer to an English
question; crosslingual 0.857 → 0.352), so a judge swap was not expected to move it.

³ **gpt-oss retry and tool counts were never recorded.** The sweep passed the agents'
action log to the diagnostics without the rankings it is aligned by, so every earlier
report showed 0 retries and 0 tool calls for agents that made hundreds. Recovered and
lost counts come from the rankings and are valid. Fixed on 2026-09-03; the DeepSeek run
is the first with real values in these columns.

## What the numbers say

- **The retry loop is a small, consistent gain**: +0.8 (gpt-oss) and +1.2 (DeepSeek)
  points of hit@5 over the control, at 8x the latency. It is at or just above the noise
  floor. DeepSeek reaches it by rewriting the query (99 rewrites, 9 expansions); it
  retries 73% of the control's misses and recovers 8 of them while losing 2.
- **The page tools hurt, and the judge is not the excuse.** gpt-oss left the ranking
  almost untouched (−0.2 points). DeepSeek uses the tools on one question in five, 257
  of 259 calls are `search_in_page`, and on those 96 questions it recovers 2 and loses
  19: hit@5 drops 3.2 points, in every question type. Promoting the found chunks into
  the ranking displaces the correct one more often than it surfaces it. The fix, if
  any, is in the promotion rule or the search-term choice, not in the judge.
- **DCI is out of the running** on its gpt-oss numbers alone.
- **Judge choice matters little for the retrieval headline** (±1 point) and a lot for
  cost and reliability: DeepSeek needs no GPU and had zero failures in about 1,400
  calls; gpt-oss had zero failures at about 1 s per call but needs 13 GB of the shared
  GPU; gemma needs 28 GB and careful configuration.

## Sources

- gpt-oss: `docs/agentic-tools-2026-08-31.md.checkpoint.json`,
  `docs/agentic-dci-2026-09-01.md.checkpoint.json`,
  `docs/agentic-retrieval-report-2026-09-01.md`
- gemma: `docs/agentic-gemma-2026-09-01.md.checkpoint.json`,
  `docs/agentic-gemma-2026-09-03.md.checkpoint.json`, `.local/logs/gemma-suite*.log`,
  `.local/reports/schema-reliability-real-2026-09-03-*.md`
- DeepSeek: `docs/agentic-deepseek-2026-09-07.md` (rendered from its checkpoint),
  `.local/logs/deepseek-suite-2026-09-07.log`
- Narrative: `development_assets/plans/gemma-judge-rerun-handoff.md`,
  `development_assets/plans/gemma-judge-handoff-teams-2026-09-07.md`
