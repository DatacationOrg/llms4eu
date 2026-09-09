# Retrieval Results — Comprehensive Overview (2026-09-07)

One page for everything measured on the LLMs4EU retrieval stack between July and
September 2026: embedders, hybrid fusion, rerankers, chunk sizes, the Open Knowledge
Format (OKF) experiment, the agentic retrieval family, and the judge-model comparison.
Every number is copied from a committed report or checkpoint listed in the Sources
section; nothing here is re-measured.

**Corpus.** 176 Slovenian tourism / history / biography pages, 726 chunks in the `base`
cutting (1,800-character target). 3,476 approved questions in five types:
`direct_short`, `direct_long`, `vague_short`, `vague_long`, `crosslingual` (English
question, Slovenian answer). Each question has one gold chunk. `hit@k` is whether the
gold chunk is in the top k over the whole corpus.

## 0. Read this first: the runs do not share a question set

The project went through four measurement eras, each on its own slice of questions.
Rows are comparable **within** a table; across tables only directionally.

| era | when | questions scored | what it measured | source |
|---|---|---:|---|---|
| Handmade eval | July | 3,476 | first embedders, fusion, first reranker attempt | `retrieval-results-handmade.md` |
| Comprehensive / OKF | 22–27 Jul | 847 (of 3,471; in-progress run) and 396 | Azure embed-v4, nemotron, first agentic loop, OKF | `retrieval-results-comprehensive-2026-07-27.md`, `comprehensive-okf-results.md` |
| Chunk-size sweep | 14–18 Aug | 3,471 per variant (shared set, labels projected per cutting) | 12 methods × 5 chunk sizes | `chunk-size-sweep-merged-2026-08-18.md` |
| Agentic family and judges | 31 Aug – 7 Sep | 495 (first 500 by id, 5 warm-up) | retry loop, page tools, corpus agent; gpt-oss vs gemma vs DeepSeek | `agentic-retrieval-report-2026-09-01.md`, `agentic-judge-comparison-2026-09-07.md` |

The run-to-run noise floor on an identical cell is about **1 point of hit@5**
(`qwen_hybrid_rerank` on the same 495 questions: 0.8808, 0.8808, 0.8828, 0.8848,
0.8808 across five runs). Differences under a point are not established anywhere on
this page.

## 1. First-stage retrieval: embedders, fusion, rerankers

Full shared question set (3,471), `base` cutting, August sweep. `*_hybrid` fuses dense
with BM25 (70/30 weighted); `*_rerank` adds the Qwen3-Reranker-0.6B cross-encoder;
`_4b` uses the 4B reranker instead.

| method | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| `qwen` (Qwen3-Embedding-0.6B, dense only) | 0.518 | 0.737 | 0.797 | 0.611 | 3 |
| `qwen4b` (Qwen3-Embedding-4B) | 0.657 | 0.848 | 0.899 | 0.742 | 19 |
| `qwen8b` (Qwen3-Embedding-8B) | 0.691 | 0.884 | 0.928 | 0.775 | 17 |
| `nemotron` (Nemotron-3-Embed-1B) | 0.458 | 0.664 | 0.727 | 0.547 | 3 |
| `nemotron8b` | **0.802** | **0.940** | **0.964** | **0.863** | 11 |
| `nemotron_hybrid` | 0.546 | 0.744 | 0.824 | 0.631 | 2 |
| `qwen_hybrid_rerank` | 0.749 | 0.880 | 0.908 | 0.808 | 842 |
| `qwen4b_hybrid_rerank` | 0.764 | 0.907 | 0.936 | 0.827 | 1,134 |
| `qwen8b_hybrid_rerank` | 0.764 | 0.912 | 0.945 | 0.829 | 692 |
| `qwen8b_hybrid_rerank_4b` | **0.845** | **0.950** | **0.961** | **0.891** | 2,714 |
| `nemotron_hybrid_rerank` | 0.746 | 0.875 | 0.893 | 0.803 | 714 |
| `nemotron8b_hybrid_rerank` | 0.765 | 0.917 | 0.954 | 0.832 | 703 |

What this table says:

- **The reranker is the dominant factor.** With the 0.6B reranker every embedder lands
  at 0.88–0.92 hit@5 regardless of its own strength (0.66–0.94 alone). The 4B reranker
  adds another 4 points on top of the best dense model.
- **`nemotron8b` alone beats every 0.6B-reranked pipeline** at 11 ms/query against
  700+. It is the strongest single-stage retriever measured.
- **Embedder size buys crosslingual.** On the July handmade set BM25 scored 0.250 hit@5
  on crosslingual questions; Qwen3-Embedding-4B scored 0.851 and Azure embed-v4 0.871.
- **The July "reranking is harmful" finding was an input-formatting bug**, not a
  property of the reranker: a 50-question sample then scored 0.24 with rerank against
  0.68 without. After the fix the reranker became the backbone of every strong method
  above. The Qwen3 reranker needs its chat template to format query and document; the
  2026-09-07 cache restore briefly reproduced the same symptom (hit@5 0.774 vs 0.881)
  when that file was missing.

### Azure embed-v4 (July, 847 questions, since retired from the Azure deployment)

| method | hit@1 | hit@5 | hit@10 | mrr@10 | judge_hit@15 | ms/query |
|---|---:|---:|---:|---:|---:|---:|
| `sparse_rerank` | 0.664 | 0.764 | 0.770 | 0.707 | 0.806 | 742 |
| `qwen4b_hybrid_rerank` | 0.688 | 0.784 | 0.793 | 0.731 | 0.831 | 574 |
| `nemotron_hybrid_rerank` | 0.737 | 0.874 | 0.889 | 0.796 | 0.914 | 740 |
| `embed_v4_hybrid_rerank` | 0.754 | 0.903 | 0.936 | 0.820 | 0.952 | 898 |
| `embed_v4_hybrid_agentic` | **0.760** | **0.911** | **0.947** | **0.828** | **0.961** | 5,162 |
| `nemotron_hybrid_agentic` | 0.743 | 0.884 | 0.903 | 0.805 | 0.928 | 4,739 |

`judge_hit@15` credits a retrieved set that a DeepSeek equivalence judge rules can
replace the gold chunk; it runs 4–5 points above strict hit@10 for every method. The
`qwen4b` row is low here because this run predates the sequence-length fix (see §2).

## 2. Chunk size

Five cuttings of the same pages, one shared question set with labels projected onto
each cutting (a question counts only where its gold text survives whole). hit@5:

| method | base (726 chunks) | tok256 (2,003) | tok512 (990) | tok512ov (1,049) | tok1024 (501) |
|---|---:|---:|---:|---:|---:|
| `qwen` | 0.737 | 0.777 | 0.817 | 0.802 | 0.841 |
| `qwen4b` | 0.848 | 0.877 | 0.909 | 0.896 | 0.922 |
| `qwen8b` | 0.884 | 0.898 | 0.931 | 0.916 | 0.945 |
| `nemotron` | 0.664 | 0.645 | 0.715 | 0.693 | 0.733 |
| `nemotron8b` | 0.940 | 0.943 | 0.969 | 0.960 | 0.970 |
| `qwen_hybrid_rerank` | 0.880 | 0.881 | 0.931 | 0.914 | 0.948 |
| `qwen4b_hybrid_rerank` | 0.907 | 0.903 | 0.946 | 0.935 | 0.966 |
| `qwen8b_hybrid_rerank` | 0.912 | 0.906 | 0.948 | 0.933 | 0.970 |
| `qwen8b_hybrid_rerank_4b` | 0.950 | – | – | – | **0.986** |
| `nemotron_hybrid_rerank` | 0.875 | 0.848 | 0.902 | 0.898 | 0.925 |
| `nemotron8b_hybrid_rerank` | 0.917 | 0.909 | 0.950 | 0.939 | 0.968 |

- **Bigger chunks score higher for every method**, but the question count shrinks with
  them (3,471 → 1,451 at tok1024) because fewer questions have a gold span that a large
  chunk contains whole, so the tok1024 column is easier as well as larger. The
  character-overlap metrics in the source report are the fairer ruler; on those the
  ordering holds but the gaps narrow.
- **tok256 is never better than base** despite being the finest cutting.
- **The token audit (11 Aug) explained the early qwen numbers:** 70% of base chunks
  exceed the 512-token limit the 0.6B/4B embedders had been configured with, so the
  tail of every long chunk was silently dropped at index time. The sweep reads each
  model's full context, which is why `qwen4b` moved from 0.78 (July) to 0.85–0.91 here.

## 3. Open Knowledge Format (OKF)

OKF replaces chunk retrieval with concept-level navigation (176 pages in 128 concepts).
396 questions, July:

| method | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query | steps/question |
|---|---:|---:|---:|---:|---:|---:|
| `sparse_rerank` | 0.742 | 0.841 | 0.851 | 0.786 | 1,345 | 1.0 |
| `qwen_hybrid_rerank` | 0.754 | **0.873** | **0.886** | **0.804** | 823 | 1.0 |
| `okf` | 0.473 | 0.501 | 0.501 | 0.485 | 13,633 | 4.4 |
| `okf_search` | 0.570 | 0.605 | 0.605 | 0.584 | 12,367 | 3.7 |

OKF scored 27–37 points below chunk retrieval at 15× the latency, and 0.000 on every
question type in the 847-question run where it was scored per concept. It was not
pursued further; the design record is in `architecture-decisions.md`.

## 4. The agentic family

Three generations of "let an LLM steer retrieval", all sitting on top of
`qwen_hybrid_rerank`, all scored on the same 495 questions:

| | Reranker only | Retry loop (Gen 1) | Page tools (Gen 2) | Corpus agent / DCI (Gen 3) |
|---|---|---|---|---|
| what the judge can do | nothing | say *sufficient*, or `reformulate` the query, or `expand` the candidate list | Gen 1 plus `list_sections` and `search_in_page` inside a retrieved page | no vector index at all: `search` (regex over the corpus), `toc`, `read`, then `answer` with chunk ids |
| judge steps / question | 0 | 1.2–1.3 | 1.3–1.5 | 5.9 |

### 4a. Results by judge model

| method | judge | hit@1 | hit@5 | hit@10 | mrr@10 | s/query |
|---|---|---:|---:|---:|---:|---:|
| Reranker only (control) | — | 0.723 | 0.881 | 0.907 | 0.791 | 0.7 |
| Retry loop | gpt-oss:20b (local) | 0.725 | 0.889 | 0.915 | 0.795 | 6.1 |
| | gemma4:31b (local) | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ |
| | DeepSeek-V4-Pro (Azure) | 0.721 | **0.893** | 0.915 | 0.793 | 5.2 |
| Page tools | gpt-oss:20b | 0.719 | 0.879 | 0.903 | 0.787 | 6.6 |
| | gemma4:31b | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ |
| | DeepSeek-V4-Pro | 0.719 | **0.849** | 0.899 | 0.778 | 6.1 |
| Corpus agent (DCI) | gpt-oss:20b | 0.576 | 0.697 | 0.723 | 0.632 | 22.1 |
| | gemma4:31b | NA¹ | NA¹ | NA¹ | NA¹ | NA¹ |
| | DeepSeek-V4-Pro | skipped² | skipped² | skipped² | skipped² | – |

### 4b. What the agents actually did (paired against the control at hit@5)

| method | judge | retried | rewritten | expanded | recovered | lost | retry recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Retry loop | gpt-oss | n/r³ | n/r³ | n/r³ | 9 | 5 | n/r³ |
| Retry loop | DeepSeek | 101 (20%) | 99 | 9 | 8 | 2 | 73% |
| Page tools | gpt-oss | n/r³ | n/r³ | n/r³ | 2 | 3 | n/r³ |
| Page tools | DeepSeek | 109 (22%) | 15 | 95 | 3 | 19 | 69% |

Page-tool usage under DeepSeek: 96 of 495 questions (19%) triggered a tool, 259 calls,
257 of them `search_in_page` and 2 `list_sections`; on those 96 questions the agent
recovered 2 hits and lost 19 (tool precision 2%).

The July retry loop, with DeepSeek judging over a different first stage and 847
questions, showed the same shape: 18–20% retried, retry recall 0.68–0.73, 11 recovered
against 2–4 lost, +0.8 to +1.0 points of hit@5. The behaviour is stable across two
embedders, two question sets and two months.

### 4c. hit@5 by question type (495 questions)

| method | judge | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---|---:|---:|---:|---:|---:|
| Reranker only | — | 0.857 | 0.940 | 0.895 | 0.972 | 0.752 |
| Retry loop | gpt-oss | **0.901** | 0.952 | 0.895 | 0.953 | 0.761 |
| Retry loop | DeepSeek | 0.890 | 0.940 | 0.886 | **0.981** | **0.780** |
| Page tools | gpt-oss | 0.857 | 0.940 | 0.895 | 0.963 | 0.752 |
| Page tools | DeepSeek | 0.824 | 0.916 | 0.867 | 0.944 | 0.706 |
| DCI | gpt-oss | **0.352** | 0.892 | 0.800 | 0.813 | 0.624 |

### Notes to §4

¹ **gemma4:31b never completed an agent cell.** Two full runs each finished only the
control (hit@5 0.883 and 0.881). The first died after 9.5 h with 136 judge failures:
gemma's thinking consumed the 1,024-token output cap on real prompts of 4k–11k tokens
and returned empty output, which the agent treated as "insufficient" and answered with
another failing call. With the corrected configuration (thinking off, function-calling
output, 100% first-attempt success on 150 real prompts, zero failures in its first hour)
the second run was killed twice by the server ending the user session at logout and was
then stopped on request. Gemma is usable but needs 28 GB of a shared GPU at ~9 s/call
against DeepSeek's 1.5 s and no GPU. Parked, not disproven.

² **DCI with DeepSeek was skipped on request.** Its failure is structural: a grep-only
agent cannot match a Slovenian answer to an English question (crosslingual 0.857 →
0.352), and a judge swap does not change that.

³ **n/r = never recorded.** Until 2026-09-03 the sweep passed the agents' action log to
the diagnostics without the rankings it is aligned by, so every earlier report showed 0
retries and 0 tool calls for agents that made hundreds. Recovered/lost come from the
rankings and are valid. The DeepSeek run is the first with real values.

## 5. Reliability of local judges

Structured-output success is a property of the (model, schema, output method,
thinking) combination, and a synthetic test prompt does not predict real prompts.
First-attempt success on **50 real benchmark prompts** (2026-09-03):

| model | config | ChunkSufficiency | ToolAction | CorpusAction | s/call |
|---|---|---:|---:|---:|---:|
| gemma4:31b | thinking on, json_schema, cap 1024 (as first run) | 76% | – | – | 36 |
| gemma4:31b | thinking on, json_schema, cap 4096 | 98% | – | – | 61 |
| gemma4:31b | **thinking off, function_calling** | **100%** | **100%** | **100%** | 9–14 |
| gpt-oss:20b | thinking low, function_calling / json_schema (per schema); full 495-question runs | 0 failures in 597 real calls | 44 failures in 617 (7.1%) | 35 failures in 2,898 after retries (7.1% of questions) | ~1 |
| DeepSeek-V4-Pro | json_object response format | 0 failures in ~1,400 real calls | | | ~1.5 |

The same gemma configuration scored 10/10 on the synthetic probe prompt (253 tokens)
before failing a quarter of real ones (median 6,000 tokens). Every failure stopped on
the output cap with empty content. Gemma's `think` is boolean: `low` and `high` produce
byte-identical output, so reasoning-effort experiments are gpt-oss-only.

## 6. Conclusions

1. **Hybrid + cross-encoder rerank is the production baseline.** On base chunks
   `qwen8b_hybrid_rerank_4b` reaches 0.950 hit@5 (0.986 at tok1024); the 0.6B reranker
   gives 0.88–0.92 for a third of the latency. `nemotron8b` alone is a strong
   single-stage alternative at 0.940 and 11 ms.
2. **Larger chunks help, up to the point where questions stop fitting.** tok512 is a
   safe improvement over base for every method; tok1024 scores highest but on a smaller,
   easier question set.
3. **The retry loop is worth about a point** with either judge, consistently, at 8×
   the latency. It is the only agentic idea that has never scored below the control.
4. **Page tools do not help, and the judge is not the excuse.** A judge that uses them
   (DeepSeek, one question in five) loses 3.2 points; a judge that barely uses them
   (gpt-oss) is flat. The fault is in promoting found chunks into the ranking, not in the
   model.
5. **DCI and OKF are out.** Both replace the index with navigation and both lose 18–37
   points, DCI catastrophically on crosslingual.
6. **Judge choice barely moves quality (±1 point) but decides cost.** DeepSeek on Azure:
   no GPU, no failures, 1.5 s/call. gpt-oss: no failures on the retry loop but 7% on the
   tool and corpus schemas, 1 s/call, 13 GB of the shared GPU. gemma: 28 GB, 9 s/call, and only in one exact configuration. The judge is now
   `agentic_judge_provider: azure` in `src/retrieval/config.yaml`.
7. **Operational lessons that cost days:** verify judges on real prompts, not synthetic
   ones; check the judge-failure count before quoting any agent number; launch
   multi-hour runs through cron with the venv interpreter (`launch_cron.sh`), because
   detached processes and the `uv` snap both die when the login session ends on this
   host; and restore models to the cache as whole snapshots.

## Sources

| document | content |
|---|---|
| `docs/retrieval-results-handmade.md` | July: first embedders, fusion, first reranker attempt (3,476 q) |
| `docs/retrieval-results-comprehensive-2026-07-27.md` | July: Azure embed-v4, nemotron, first agentic loop, OKF (847 q) |
| `docs/comprehensive-okf-results.md`, `docs/okf_benchmark.md` | OKF vs chunk retrieval (396 q) |
| `docs/chunk-token-audit-2026-08-11.md` | 70% of base chunks over the 512-token embedder limit |
| `docs/chunk-size-sweep-merged-2026-08-18.md` (+ `docs/chunk-size-sweeps/`) | 12 methods × 5 cuttings, full question set |
| `docs/agentic-retrieval-report-2026-09-01.md` | gpt-oss agentic family, noise floor, DCI analysis |
| `docs/agentic-tools-2026-08-31.md`, `docs/agentic-dci-2026-09-01.md` | the sweep reports behind it |
| `docs/agentic-deepseek-2026-09-07.md` | DeepSeek-judged run (rendered from checkpoint) |
| `docs/agentic-judge-comparison-2026-09-07.md` | three-judge comparison with footnotes |
| `docs/agentic-findings.md` | July agentic experiments: fusion, multi-query, answerability gating |
| `docs/architecture-decisions.md` | design record: retrieval names, hybrid and rerank history, OKF, DCI, local inference |
| `development_assets/plans/gemma-judge-rerun-handoff.md` | day-by-day log of the gemma track and the infrastructure fixes |
| `.local/reports/schema-reliability-real-2026-09-03-*.md` | judge reliability on real prompts |
