# LLMs4EU Retrieval Experiments — Every Run, One Document (2026-09-14)

Every result the project has reported or generated with more than 150 samples,
from the first BM25 baseline in May 2026 to the geo runs and corpus expansion of
September 2026, copied into one place with the tables unified wherever the runs
share a metric. Nothing here is re-measured. Sources are the committed reports
under `docs/reports/`, the documents that were deleted or rewritten in git
history (recovered by commit hash, Appendix B), the local checkpoints under
`.local/`, and the numbers quoted only in `docs/architecture/`. Results with
150 or fewer samples are listed, not tabulated, in Appendix A so nothing is lost
silently.

Method names are the ones each run printed. Two renamings matter: the May–June
`bm25` / `*_chunk` / `*_chunk_summary_bm25` names map to today's `sparse` /
`<provider>` / `<provider>_hybrid`, and the July `azure_*` embedding methods were
renamed `embed_v4_*` in the 2026-07-27 run (same Azure `embed-v-4-0` deployment,
since retired). `*_hybrid` fuses dense with BM25 (weighted 70/30 from June on;
RRF in the May 300-question samples), `*_rerank` adds the Qwen3-Reranker-0.6B
cross-encoder, `_4b` uses the 4B reranker, `*_agentic` adds the LLM sufficiency
judge with `reformulate` / `expand`, `*_agentic_tools` adds `list_sections` /
`search_in_page`, `dci` is the grep-driven corpus agent, `*_geo` adds the
geographic stage.

## 0. Corpus, metrics, and why the tables cannot be read across eras

**Corpus.** 176 Slovenian tourism / history / biography pages from four sources,
726 `base` chunks (1,800-character target, 1,064,470 indexed characters),
3,476 approved questions (3,471 usable in most runs) in five types:
`direct_short`, `direct_long`, `vague_short`, `vague_long`, `crosslingual`
(English question, Slovenian answer). One gold chunk per question. Since
2026-09-13 the corpus is 1,338 pages / 11,261 base chunks across ten localities
(§7.4); no retrieval run has been scored on the expanded corpus yet.

**Metrics.** `hit@k`: gold chunk in the top k over the whole corpus. `recall@k`
equals `hit@k` because there is one gold chunk; it is kept where the source
printed it. `mrr@k`: reciprocal rank of the gold chunk, cut at k.
`judge_hit@15`: a DeepSeek equivalence judge rules the retrieved set can replace
the gold chunk. `ms/query`: wall-clock per question including the first stage,
reranker and any judge calls. `queries/query`: retrieval or navigation calls per
question (1.00 = no agent action).

**Eras.** Seven measurement eras, each on its own slice of questions. Rows are
comparable within a table; across eras only in direction.

| era | when | samples scored | design | what it measured | § |
|---|---|---:|---|---|---|
| A | May–Jun 2026 | 3,476 full set; 300-question samples; 711 crosslingual | full set / `--limit 300` | first embedders, summaries, MiniLM, RRF fusion, first reranker attempt | 1 |
| B | 2026-06-23 | 909 of 3,471 (in-progress run) | prefix by id | qwen / qwen4b ladders, first agentic loop | 2 |
| C | 22–27 Jul 2026 | 199, 260, 264, 847 (snapshots of one configuration); 396 (OKF-eligible) | prefix by id | Azure embed-v4, Nemotron, DeepSeek retry loop, equivalence judge, OKF | 3 |
| D | 14–19 Aug 2026 | 3,471 / 7,801 / 3,605 / 3,865 / 1,451 per cutting | per-variant | 12 methods × 5 chunk sizes; token audit; span and cost metrics | 4 |
| E | 31 Aug – 7 Sep 2026 | 495 (first 500 by id, 5 warm-up) | shared | retry loop, page tools, DCI; gpt-oss vs gemma vs DeepSeek judges | 5 |
| F | 8–9 Sep 2026 | 500 | shared | hard geo filter vs soft geo; agents on the geo stage | 6 |
| G | 8–13 Sep 2026 | 176 pages, 375 mentioned rows, 151 names, 1,338 pages | — | enrichment coverage, gazetteer misses, corpus expansion, seed quality | 7 |

**Three comparability traps.**

- Everything before 2026-08-11 embedded `base` chunks with the Qwen embedders
  capped at 512 tokens, and 69.7% of base chunks exceed that (§4.1), so the
  tail of most chunks was never embedded. The August sweep and everything after
  read each model's full context; `qwen4b` moved from 0.78 (July, 847 q) to
  0.907 (August, 3,471 q) hit@5 for that reason alone.
- The Azure `embed-v-4-0` embedding deployment is retired; `azure_*` /
  `embed_v4_*` rows are historical.
- The May reranker rows (§1.3) and the "reranking is harmful" reading were an
  input-formatting bug (the Qwen3 reranker needs its chat template); the same
  symptom reappeared briefly on 2026-09-07 when the template file was missing
  from the restored cache (hit@5 0.774 against 0.881).

**Noise floor.** `qwen_hybrid_rerank` has no LLM and should be deterministic; it
is not (Chroma's approximate nearest-neighbour search, inferred not
instrumented). On the same 495 questions it scored hit@5 0.8808, 0.8808, 0.8828,
0.8848 and 0.8808 across five runs (including the two gemma runs that finished
only this control cell: 0.883 and 0.881), with hit@1 moving 0.7232 → 0.7313 and
`direct_short` moving 1 point between two runs. On the same 500 geo questions it
scored 0.884 in all three runs. **Differences under about 1 point of hit@5 are
not established anywhere in this document.**

## 1. Era A — handmade evaluation (May–June 2026)

Old 512-token cap. `*_summary` methods index an LLM chunk summary instead of, or
next to, the chunk text and return the chunk as evidence. Sources: the deleted
`docs/retrieval-results.md` at commits `933d944` (2026-05-29) and `8acdc9e`
(2026-06-02), now `docs/reports/retrieval/retrieval-results-handmade.md`.

### 1.1 Full question set, 3,476 questions

Where both renderings exist the June figures are used; the May rendering of
`qwen_chunk` read 0.718 / 0.794 for hit@5 / hit@10.

| method (as run) | current name | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query |
|---|---|---:|---:|---:|---:|---:|
| bm25 | sparse | 0.544 | 0.668 | 0.705 | 0.596 | 3.0 |
| qwen_chunk | qwen | 0.512 | 0.719 | 0.795 | 0.602 | 3.7 |
| qwen_summary | — | 0.442 | 0.666 | 0.743 | 0.536 | 2.9 |
| qwen_chunk_summary | — | 0.523 | 0.731 | 0.795 | 0.611 | 2.9 |
| qwen_chunk_summary_bm25 (weighted 70/30) | qwen_hybrid | 0.603 | 0.794 | 0.851 | 0.685 | — |
| qwen4b_chunk_summary | qwen4b | 0.641 | 0.837 | 0.891 | 0.726 | — |
| english_chunk (MiniLM) | english | 0.142 | 0.245 | 0.302 | 0.187 | 1.2 |
| english_summary | — | 0.152 | 0.277 | 0.346 | 0.207 | 0.7 |
| english_chunk_summary | — | 0.161 | 0.283 | 0.351 | 0.214 | 0.6 |
| azure_chunk | azure (embed-v-4-0) | 0.669 | 0.861 | 0.905 | 0.752 | — |
| azure_chunk_summary_bm25 (weighted 70/30) | azure_hybrid | **0.726** | **0.893** | **0.930** | **0.797** | — |

### 1.2 Crosslingual subset, 711 questions

| method (as run) | current name | hit@1 | hit@5 | hit@10 | mrr@10 |
|---|---|---:|---:|---:|---:|
| bm25 | sparse | 0.145 | 0.250 | 0.297 | 0.190 |
| qwen4b_chunk_summary | qwen4b | 0.608 | 0.851 | 0.911 | 0.710 |
| azure_chunk_summary | azure | 0.599 | **0.871** | **0.928** | **0.715** |

### 1.3 300-question sample (`--limit 300`, May 2026)

One sample: `bm25` scored identically in all four sample runs. `_bm25` here is
RRF fusion, not the weighted fusion of §1.1. The `_rerank` rows are the
formatting bug, kept because they were reported; the reranker scored 15,000
query/chunk pairs on a GPU shared with a training job, so its latency is also
unreliable.

| method (as run) | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.547 | 0.647 | 0.673 | 0.590 | 3.3–3.5 |
| english_chunk | 0.110 | 0.220 | 0.267 | 0.152 | 8.2 |
| english_summary | 0.180 | 0.287 | 0.360 | 0.228 | 1.2 |
| english_chunk_summary | 0.133 | 0.267 | 0.317 | 0.188 | 1.7 |
| qwen_chunk_summary | 0.530 | 0.743 | 0.800 | 0.621 | 9.9 |
| qwen_chunk_summary_bm25 (RRF) | 0.587 | 0.740 | 0.817 | 0.652 | 7.2 |
| qwen_chunk_summary_rerank | 0.123 | 0.257 | 0.337 | 0.179 | 635.0 |
| qwen_chunk_summary_rerank_hybrid | 0.113 | 0.257 | 0.363 | 0.178 | 624.4 |
| azure_chunk | 0.653 | 0.867 | 0.910 | 0.743 | 40.7 |
| azure_summary | 0.577 | 0.773 | 0.823 | 0.662 | 37.8 |
| azure_chunk_summary | **0.663** | **0.873** | **0.913** | **0.755** | 42.7–44.4 |
| azure_chunk_summary_bm25 (RRF) | 0.637 | 0.753 | 0.837 | 0.693 | 47.6 |
| azure_chunk_summary_rerank | 0.107 | 0.243 | 0.340 | 0.167 | 766.6 |
| azure_chunk_summary_rerank_hybrid | 0.107 | 0.253 | 0.360 | 0.169 | 677.4 |

What era A established: weighted fusion beat RRF (RRF hurt Azure); summary-only
was always weaker than chunk-only; MiniLM cannot serve a Slovenian corpus;
embedder size buys crosslingual (BM25 0.250 → qwen4b 0.851 → Azure 0.871).

## 2. Era B — first agentic sweep, 909 questions (2026-06-23)

Old 512-token cap. In-progress run over 3,471 questions, 909 aligned when it
stopped. Source: deleted `docs/retrieval-results-agentic.md` (commit `e47194b`).
The dense-only cells here (`qwen` 0.189, `qwen4b` 0.231 hit@5) are far below the
same methods in every other era (0.719 / 0.837 in §1.1); the bug list of that
period (`agentic-findings.md`: un-cached Chroma client leaking connections until
readiness checks failed mid-run, agentic result truncation) dates from these
runs. Read the `*_rerank` rows as valid and the dense and agentic rows as a
record of the broken state that led to the fixes.

### 2.1 Overall

| method | hit@1 | hit@5 | hit@10 | mrr@10 |
|---|---:|---:|---:|---:|
| sparse | 0.554 | 0.674 | 0.708 | 0.605 |
| sparse_rerank | 0.664 | 0.762 | 0.768 | 0.707 |
| qwen | 0.119 | 0.189 | 0.219 | 0.149 |
| qwen_hybrid | 0.155 | 0.311 | 0.546 | 0.235 |
| qwen_rerank | 0.234 | 0.275 | 0.279 | 0.252 |
| qwen_hybrid_rerank | 0.670 | 0.769 | 0.779 | 0.713 |
| qwen4b | 0.139 | 0.231 | 0.268 | 0.178 |
| qwen4b_hybrid | 0.183 | 0.372 | 0.602 | 0.274 |
| qwen4b_rerank | 0.285 | 0.327 | 0.334 | 0.303 |
| qwen4b_hybrid_rerank | **0.690** | **0.783** | **0.793** | **0.731** |
| qwen_agentic | 0.121 | 0.205 | 0.244 | 0.157 |
| qwen_hybrid_agentic | 0.161 | 0.327 | 0.619 | 0.250 |

### 2.2 Speed and hit@5 by question type

| method | ms/query | queries/query | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse | 2.0 | 1.00 | 0.262 | 0.885 | 0.763 | 0.834 | 0.582 |
| sparse_rerank | 730.3 | 1.00 | 0.470 | 0.923 | 0.812 | 0.882 | **0.693** |
| qwen | 16.4 | 1.00 | 0.207 | 0.268 | 0.199 | 0.171 | 0.106 |
| qwen_hybrid | 15.9 | 1.00 | 0.232 | 0.432 | 0.333 | 0.342 | 0.212 |
| qwen_rerank | 441.0 | 1.00 | 0.317 | 0.339 | 0.274 | 0.278 | 0.175 |
| qwen_hybrid_rerank | 539.7 | 1.00 | 0.518 | **0.934** | **0.839** | 0.888 | 0.640 |
| qwen4b | 62.0 | 1.00 | 0.256 | 0.273 | 0.210 | 0.267 | 0.153 |
| qwen4b_hybrid | 16.4 | 1.00 | 0.311 | 0.459 | 0.387 | 0.433 | 0.265 |
| qwen4b_rerank | 501.7 | 1.00 | 0.335 | 0.377 | 0.312 | 0.369 | 0.243 |
| qwen4b_hybrid_rerank | 576.3 | 1.00 | **0.549** | **0.934** | 0.823 | **0.920** | 0.667 |
| qwen_agentic | 8,041.4 | 2.51 | 0.213 | 0.273 | 0.215 | 0.187 | 0.138 |
| qwen_hybrid_agentic | 5,695.9 | 1.88 | 0.244 | 0.448 | 0.339 | 0.374 | 0.222 |

## 3. Era C — comprehensive runs with Azure embed-v4 and the OKF benchmark (July 2026)

Old 512-token cap for the Qwen embedders (which is why `qwen4b_hybrid_rerank`
sits at 0.78 here and 0.91 in §4). Chunk scoring, 5 warm-up queries, DeepSeek-V4-Pro
(Azure) as retry-loop judge and as equivalence judge at cutoff 15. Four
snapshots of one configuration exist because the run was restarted; they are
prefixes by question id of different lengths, not independent samples. The OKF
benchmark is a separate 396-question run over the questions whose gold page the
OKF bundle covers. Sources: `retrieval-results-comprehensive-2026-07-27.md` (847),
`comprehensive-okf-results.md` (264 / 265 timed, last updated 2026-07-22),
deleted `docs/retrieval-results-chunks-okf.md` (260, 2026-07-23 12:23) and
`docs/retrieval-results-comprehensive-2026-07-23.md` (199, 2026-07-23 14:46),
`okf_benchmark.md` (396).

### 3.1 Overall, all snapshots

| run | n | method | hit@1 | hit@5 | hit@10 | mrr@10 | hit@20 | mrr@20 | judge_hit@15 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 07-27 | 847 | sparse_rerank | 0.664 | 0.764 | 0.770 | 0.707 | — | — | 0.806 |
| 07-27 | 847 | qwen4b_hybrid_rerank | 0.688 | 0.784 | 0.793 | 0.731 | — | — | 0.831 |
| 07-27 | 847 | nemotron_hybrid_rerank | 0.737 | 0.874 | 0.889 | 0.796 | — | — | 0.914 |
| 07-27 | 847 | embed_v4_hybrid_rerank | 0.754 | 0.903 | 0.936 | 0.820 | — | — | 0.952 |
| 07-27 | 847 | embed_v4_hybrid_agentic | **0.760** | **0.911** | **0.947** | **0.828** | 0.948 | 0.828 | **0.961** |
| 07-27 | 847 | nemotron_hybrid_agentic | 0.743 | 0.884 | 0.903 | 0.805 | 0.904 | 0.805 | 0.928 |
| 07-27 | 847 | okf (concept-scored) | 0.494 | 0.532 | 0.534 | 0.511 | — | — | 0.543 |
| 07-22 | 264 | sparse_rerank | 0.640 | 0.742 | 0.746 | 0.682 | — | — | 0.780 |
| 07-22 | 264 | qwen4b_hybrid_rerank | 0.652 | 0.765 | 0.773 | 0.699 | — | — | 0.811 |
| 07-22 | 264 | nemotron_hybrid_rerank | 0.716 | 0.848 | 0.864 | 0.770 | — | — | 0.890 |
| 07-22 | 264 | azure_hybrid_rerank | 0.746 | 0.898 | 0.939 | 0.810 | — | — | 0.955 |
| 07-22 | 264 | azure_hybrid_agentic | 0.746 | 0.917 | 0.932 | 0.814 | 0.932 (@15) | 0.814 (@15) | 0.955 |
| 07-22 | 264 | nemotron_hybrid_agentic | 0.720 | 0.864 | 0.875 | 0.776 | 0.875 (@15) | 0.776 (@15) | 0.902 |
| 07-23a | 260 | sparse_rerank | 0.642 | 0.746 | 0.750 | 0.685 | — | — | 0.785 |
| 07-23a | 260 | qwen4b_hybrid_rerank | 0.654 | 0.769 | 0.777 | 0.702 | — | — | 0.815 |
| 07-23a | 260 | nemotron_hybrid_rerank | 0.719 | 0.850 | 0.865 | 0.773 | — | — | 0.892 |
| 07-23a | 260 | azure_hybrid_rerank | 0.750 | 0.900 | 0.942 | 0.814 | — | — | 0.958 |
| 07-23a | 260 | azure_hybrid_agentic | 0.750 | 0.919 | 0.935 | 0.817 | 0.935 (@15) | 0.817 (@15) | 0.958 |
| 07-23a | 260 | nemotron_hybrid_agentic | 0.723 | 0.865 | 0.877 | 0.779 | 0.877 (@15) | 0.779 (@15) | 0.900 |
| 07-23b | 199 | sparse_rerank | 0.638 | 0.729 | 0.734 | 0.676 | — | — | 0.779 |
| 07-23b | 199 | qwen4b_hybrid_rerank | 0.653 | 0.754 | 0.759 | 0.695 | — | — | 0.814 |
| 07-23b | 199 | nemotron_hybrid_rerank | 0.729 | 0.849 | 0.869 | 0.780 | — | — | 0.899 |
| 07-23b | 199 | azure_hybrid_rerank | 0.759 | 0.899 | 0.945 | 0.819 | — | — | 0.965 |
| 07-23b | 199 | azure_hybrid_agentic | 0.754 | 0.920 | 0.945 | 0.821 | — | — | 0.965 |
| 07-23b | 199 | nemotron_hybrid_agentic | 0.729 | 0.864 | 0.879 | 0.783 | — | — | 0.905 |
| OKF | 396 | sparse_rerank | 0.742 | 0.841 | 0.851 | 0.786 | — | — | — |
| OKF | 396 | qwen_hybrid_rerank | 0.754 | **0.873** | **0.886** | **0.804** | — | — | — |
| OKF | 396 | okf | 0.473 | 0.501 | 0.501 | 0.485 | — | — | — |
| OKF | 396 | okf_search | 0.570 | 0.605 | 0.605 | 0.584 | — | — | — |

Notes. `judge_hit@15` runs 4–5 points above strict hit@10 for every chunk method.
The 264-snapshot's agentic rows were rendered at depth 15 rather than 20. OKF
navigated 176 pages in 128 concepts in the 07-27 run (170–171 covered pages and
2,312–2,377 eligible questions in the local checkpoints); the surviving OKF
checkpoint (`.local/retrieval-results-okf-v2.checkpoint.json`) holds 390 timed
questions with one failure each for `okf` and `okf_search`, while the report
prints 396 queries; the report's figures are kept here.

### 3.2 Speed, all snapshots

`n timed` is one more than `n scored` where the last question had not aligned.

| run | n timed | method | ms/query | queries/query | queries | chunk expansions |
|---|---:|---|---:|---:|---:|---:|
| 07-27 | 848 | sparse_rerank | 742.1 | 1.00 | 848 | 0 |
| 07-27 | 848 | qwen4b_hybrid_rerank | 574.2 | 1.00 | 848 | 0 |
| 07-27 | 848 | nemotron_hybrid_rerank | 739.6 | 1.00 | 848 | 0 |
| 07-27 | 848 | embed_v4_hybrid_rerank | 897.8 | 1.00 | 848 | 0 |
| 07-27 | 847 | embed_v4_hybrid_agentic | 5,162.0 | 1.30 | 1,105 | 14 |
| 07-27 | 847 | nemotron_hybrid_agentic | 4,739.4 | 1.33 | 1,125 | 20 |
| 07-27 | 847 | okf | 12,988.4 | 3.97 | 3,366 | 0 |
| 07-22 | 265 | sparse_rerank | 744.5 | 1.00 | 265 | 0 |
| 07-22 | 265 | qwen4b_hybrid_rerank | 576.9 | 1.00 | 265 | 0 |
| 07-22 | 265 | nemotron_hybrid_rerank | 739.6 | 1.00 | 265 | 0 |
| 07-22 | 265 | azure_hybrid_rerank | 1,044.2 | 1.00 | 265 | 0 |
| 07-22 | 264 | azure_hybrid_agentic | 9,751.4 | 1.31 | 347 | 2 |
| 07-22 | 264 | nemotron_hybrid_agentic | 8,565.2 | 1.38 | 364 | 6 |
| 07-23a | 261 | sparse_rerank | 750.5 | 1.00 | 261 | 0 |
| 07-23a | 261 | qwen4b_hybrid_rerank | 578.0 | 1.00 | 261 | 0 |
| 07-23a | 261 | nemotron_hybrid_rerank | 740.7 | 1.00 | 261 | 0 |
| 07-23a | 261 | azure_hybrid_rerank | 1,041.6 | 1.00 | 261 | 0 |
| 07-23a | 260 | azure_hybrid_agentic | 9,688.6 | 1.30 | 339 | 2 |
| 07-23a | 260 | nemotron_hybrid_agentic | 8,563.0 | 1.38 | 358 | 6 |
| 07-23b | 200 | sparse_rerank | 922.2 | 1.01 | 200 | 0 |
| 07-23b | 200 | qwen4b_hybrid_rerank | 721.2 | 1.01 | 200 | 0 |
| 07-23b | 200 | nemotron_hybrid_rerank | 911.7 | 1.01 | 200 | 0 |
| 07-23b | 200 | azure_hybrid_rerank | 926.1 | 1.01 | 200 | 0 |
| 07-23b | 199 | azure_hybrid_agentic | 11,453.1 | 1.28 | 255 | 3 |
| 07-23b | 199 | nemotron_hybrid_agentic | 9,469.6 | 1.35 | 268 | 6 |
| OKF | 396 | sparse_rerank | 1,345.0 | 1.00 | 396 | — |
| OKF | 396 | qwen_hybrid_rerank | 823.2 | 1.00 | 396 | — |
| OKF | 396 | okf | 13,633.0 | 4.40 | 1,737 | — |
| OKF | 396 | okf_search | 12,367.0 | 3.72 | 1,468 | — |

### 3.3 hit@5 by question type, all snapshots

| run | n | method | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---:|---|---:|---:|---:|---:|---:|
| 07-27 | 847 | sparse_rerank | 0.503 | 0.916 | 0.802 | 0.878 | 0.692 |
| 07-27 | 847 | qwen4b_hybrid_rerank | 0.564 | 0.928 | 0.814 | 0.924 | 0.670 |
| 07-27 | 847 | nemotron_hybrid_rerank | 0.879 | 0.946 | 0.876 | 0.948 | 0.731 |
| 07-27 | 847 | embed_v4_hybrid_rerank | 0.886 | 0.964 | 0.910 | **0.983** | 0.780 |
| 07-27 | 847 | embed_v4_hybrid_agentic | **0.906** | **0.970** | **0.915** | **0.983** | **0.791** |
| 07-27 | 847 | nemotron_hybrid_agentic | 0.893 | 0.958 | 0.876 | 0.953 | 0.753 |
| 07-27 | 847 | okf | 0.000¹ | 0.000¹ | 0.000¹ | 0.000¹ | 0.000¹ |
| 07-22 | 264 | sparse_rerank | 0.426 | 0.909 | 0.746 | 0.933 | 0.667 |
| 07-22 | 264 | qwen4b_hybrid_rerank | 0.574 | 0.909 | 0.763 | 0.950 | 0.611 |
| 07-22 | 264 | nemotron_hybrid_rerank | 0.830 | 0.909 | 0.847 | 0.967 | 0.685 |
| 07-22 | 264 | azure_hybrid_rerank | 0.851 | 0.932 | 0.915 | 1.000 | 0.778 |
| 07-22 | 264 | azure_hybrid_agentic | 0.894 | 0.932 | 0.932 | 1.000 | 0.815 |
| 07-22 | 264 | nemotron_hybrid_agentic | 0.851 | 0.932 | 0.847 | 0.967 | 0.722 |
| 07-23a | 260 | sparse_rerank | 0.426 | 0.909 | 0.746 | 0.932 | 0.686 |
| 07-23a | 260 | qwen4b_hybrid_rerank | 0.574 | 0.909 | 0.763 | 0.949 | 0.627 |
| 07-23a | 260 | nemotron_hybrid_rerank | 0.830 | 0.909 | 0.847 | 0.966 | 0.686 |
| 07-23a | 260 | azure_hybrid_rerank | 0.851 | 0.932 | 0.915 | 1.000 | 0.784 |
| 07-23a | 260 | azure_hybrid_agentic | 0.894 | 0.932 | 0.932 | 1.000 | 0.824 |
| 07-23a | 260 | nemotron_hybrid_agentic | 0.851 | 0.932 | 0.847 | 0.966 | 0.725 |
| 07-23b | 199 | sparse_rerank | 0.389 | 0.886 | 0.732 | 0.935 | 0.659 |
| 07-23b | 199 | qwen4b_hybrid_rerank | 0.556 | 0.886 | 0.732 | 0.957 | 0.610 |
| 07-23b | 199 | nemotron_hybrid_rerank | 0.833 | 0.886 | 0.854 | 0.978 | 0.683 |
| 07-23b | 199 | azure_hybrid_rerank | 0.833 | 0.914 | 0.902 | 1.000 | 0.829 |
| 07-23b | 199 | azure_hybrid_agentic | 0.889 | 0.914 | 0.902 | 1.000 | 0.878 |
| 07-23b | 199 | nemotron_hybrid_agentic | 0.889 | 0.886 | 0.854 | 0.978 | 0.707 |
| OKF | 396 | sparse_rerank | 0.710 | 0.945 | 0.831 | 0.905 | **0.802** |
| OKF | 396 | qwen_hybrid_rerank | **0.899** | 0.945 | **0.892** | **0.929** | 0.721 |
| OKF | 396 | okf | 0.435 | 0.548 | 0.566 | 0.583 | 0.372 |
| OKF | 396 | okf_search | 0.594 | 0.699 | 0.687 | 0.643 | 0.419 |

¹ Rendering artefact: `okf` was scored per concept in this run and its overall
hit@5 is 0.532, so the per-type zeros are not a measurement.

### 3.4 Retry-loop diagnostics, paired against each agent's own reranked baseline at hit@5

Retry precision: fraction of retried queries whose first relevant rank improved.
Retry recall: fraction of baseline misses that were retried. The 264 snapshot
published no diagnostics.

| run | n | agent | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 07-27 | 847 | embed_v4_hybrid_agentic | 157 (18.5%) | 154 | 13 | 0.153 | 0.683 | 11 | 4 | 11,171.8 | 3,794.6 |
| 07-27 | 847 | nemotron_hybrid_agentic | 166 (19.6%) | 163 | 18 | 0.120 | 0.729 | 11 | 2 | 11,172.0 | 3,171.4 |
| 07-23a | 260 | azure_hybrid_agentic | 47 (18.1%) | 47 | 2 | 0.149 | 0.731 | 5 | 0 | — | — |
| 07-23a | 260 | nemotron_hybrid_agentic | 55 (21.2%) | 54 | 5 | 0.091 | 0.795 | 4 | 0 | — | — |
| 07-23b | 199 | azure_hybrid_agentic | 34 (17.1%) | 34 | 3 | 0.206 | 0.700 | 4 | 0 | 21,756.3 | 9,330.0 |
| 07-23b | 199 | nemotron_hybrid_agentic | 41 (20.6%) | 41 | 6 | 0.098 | 0.733 | 4 | 1 | 19,697.7 | 6,815.5 |

The shape is stable at every sample size: the judge retries 17–21% of questions,
catches 68–80% of the baseline's misses (retry recall) and repairs 9–21% of the
ones it retries (retry precision). It is a good miss detector and a poor repair
mechanism.

## 4. Era D — chunk size (14–19 August 2026)

Five cuttings of the same 176 pages in `.local/db/pages-variants.db`, **per-variant**
question design: each cutting owns questions generated from its own chunks, so
gold is correct by construction but the columns are differently sized samples,
and a cutting with more chunks is a harder haystack while a larger chunk is
likelier to contain any answer. Sequence limits raised after the token audit, so
nothing is truncated. Three runs joined (08-14: `qwen`, `qwen_hybrid_rerank`,
`nemotron`; 08-17: `nemotron_hybrid`, `nemotron_hybrid_rerank`; 08-18/19:
`qwen4b`, `qwen8b`, `nemotron8b`, their hybrid-rerank rungs, and
`qwen8b_hybrid_rerank_4b`), with identical question sets per cutting verified
across runs. Source: `chunk-size-sweep-merged-2026-08-18.md` and `sweeps/`.
Bold is the best cell of the metric.

### 4.1 The cuttings and the token audit

| variant | chunks | indexed chars | questions scored | max tokens (qwen) | over 512 (qwen at old cap) | over 256 (MiniLM) |
|---|---:|---:|---:|---:|---:|---:|
| base | 726 | 1,064,470 | 3,471 | 1,253 | 69.7% | 90.1% |
| tok256 | 2,003 | 1,077,031 | 7,801 | 409 | 0.0% | 36.8% |
| tok512 | 990 | 1,078,330 | 3,605 | 796 | 16.0% | 95.7% |
| tok512ov | 1,049 | 1,151,174 | 3,865 | 792 | 15.3% | 96.0% |
| tok1024 | 501 | 1,079,332 | 1,451 | 1,573 | 90.2% | 93.4% |

Token audit of 2026-08-11 over the 726 `base` chunks (Qwen3-Embedding-0.6B and
4B, both configured at 512 tokens; v1 = plain chunk text, v2 = chunk plus
representation metadata):

| representation | tokens p50 / p90 / p99 / max | chunks over 512 | tokens lost total / max per chunk | overhead mean / max |
|---|---|---:|---:|---:|
| v1 | 616.5 / 921.5 / 1,179 / 1,253 | 506 (69.70%) | 103,114 / 741 | 24.3 / 56 |
| v2 | 635 / 941.5 / 1,200.5 / 1,275 | 524 (72.18%) | 113,250 / 763 | 44.4 / 81 |

Chunk characters: p50 1,572.5, p90 2,324, max 2,599 (target 1,800, max 2,600).
Across both providers and representations 2,060 of 2,904 inputs (70.94%) exceeded
the limit. This is the explanation for every July `qwen` / `qwen4b` figure.

### 4.2 hit@1

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| qwen | 08-14 | 0.518 | 0.569 | 0.581 | 0.558 | 0.612 |
| qwen4b | 08-18 | 0.657 | 0.695 | 0.709 | 0.693 | 0.731 |
| qwen8b | 08-18 | 0.691 | 0.730 | 0.731 | 0.718 | 0.752 |
| nemotron | 08-14 | 0.458 | 0.444 | 0.482 | 0.454 | 0.517 |
| nemotron8b | 08-18 | 0.802 | 0.826 | 0.840 | 0.821 | 0.829 |
| nemotron_hybrid | 08-17 | 0.546 | 0.518 | 0.573 | 0.538 | 0.604 |
| qwen_hybrid_rerank | 08-14 | 0.749 | 0.739 | 0.781 | 0.754 | 0.826 |
| qwen4b_hybrid_rerank | 08-18 | 0.764 | 0.750 | 0.792 | 0.765 | 0.837 |
| qwen8b_hybrid_rerank | 08-18 | 0.764 | 0.751 | 0.789 | 0.766 | 0.837 |
| qwen8b_hybrid_rerank_4b | 08-19 | 0.845 | — | — | — | **0.898** |
| nemotron_hybrid_rerank | 08-17 | 0.746 | 0.720 | 0.765 | 0.743 | 0.811 |
| nemotron8b_hybrid_rerank | 08-18 | 0.765 | 0.754 | 0.793 | 0.765 | 0.837 |

### 4.3 hit@5

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| qwen | 08-14 | 0.737 | 0.777 | 0.817 | 0.802 | 0.841 |
| qwen4b | 08-18 | 0.848 | 0.877 | 0.909 | 0.896 | 0.922 |
| qwen8b | 08-18 | 0.884 | 0.898 | 0.931 | 0.916 | 0.945 |
| nemotron | 08-14 | 0.664 | 0.645 | 0.715 | 0.693 | 0.733 |
| nemotron8b | 08-18 | 0.940 | 0.943 | 0.969 | 0.960 | 0.970 |
| nemotron_hybrid | 08-17 | 0.744 | 0.713 | 0.787 | 0.774 | 0.824 |
| qwen_hybrid_rerank | 08-14 | 0.880 | 0.881 | 0.931 | 0.914 | 0.948 |
| qwen4b_hybrid_rerank | 08-18 | 0.907 | 0.903 | 0.946 | 0.935 | 0.966 |
| qwen8b_hybrid_rerank | 08-18 | 0.912 | 0.906 | 0.948 | 0.933 | 0.970 |
| qwen8b_hybrid_rerank_4b | 08-19 | 0.950 | — | — | — | **0.986** |
| nemotron_hybrid_rerank | 08-17 | 0.875 | 0.848 | 0.902 | 0.898 | 0.925 |
| nemotron8b_hybrid_rerank | 08-18 | 0.917 | 0.909 | 0.950 | 0.939 | 0.968 |

### 4.4 hit@10 (equal to recall@10 in every cell)

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| qwen | 08-14 | 0.797 | 0.828 | 0.877 | 0.855 | 0.897 |
| qwen4b | 08-18 | 0.899 | 0.910 | 0.947 | 0.929 | 0.957 |
| qwen8b | 08-18 | 0.928 | 0.931 | 0.962 | 0.949 | 0.972 |
| nemotron | 08-14 | 0.727 | 0.708 | 0.783 | 0.767 | 0.813 |
| nemotron8b | 08-18 | 0.964 | 0.961 | 0.983 | 0.977 | 0.989 |
| nemotron_hybrid | 08-17 | 0.824 | 0.800 | 0.865 | 0.853 | 0.888 |
| qwen_hybrid_rerank | 08-14 | 0.908 | 0.905 | 0.954 | 0.937 | 0.968 |
| qwen4b_hybrid_rerank | 08-18 | 0.936 | 0.931 | 0.973 | 0.960 | 0.983 |
| qwen8b_hybrid_rerank | 08-18 | 0.945 | 0.937 | 0.975 | 0.965 | 0.988 |
| qwen8b_hybrid_rerank_4b | 08-19 | 0.961 | — | — | — | **0.992** |
| nemotron_hybrid_rerank | 08-17 | 0.893 | 0.868 | 0.920 | 0.916 | 0.940 |
| nemotron8b_hybrid_rerank | 08-18 | 0.954 | 0.942 | 0.979 | 0.969 | 0.989 |

### 4.5 mrr@10

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| qwen | 08-14 | 0.611 | 0.658 | 0.683 | 0.665 | 0.710 |
| qwen4b | 08-18 | 0.742 | 0.774 | 0.796 | 0.783 | 0.815 |
| qwen8b | 08-18 | 0.775 | 0.803 | 0.817 | 0.805 | 0.833 |
| nemotron | 08-14 | 0.547 | 0.531 | 0.584 | 0.559 | 0.613 |
| nemotron8b | 08-18 | 0.863 | 0.876 | 0.897 | 0.883 | 0.890 |
| nemotron_hybrid | 08-17 | 0.631 | 0.606 | 0.668 | 0.642 | 0.698 |
| qwen_hybrid_rerank | 08-14 | 0.808 | 0.800 | 0.847 | 0.826 | 0.880 |
| qwen4b_hybrid_rerank | 08-18 | 0.827 | 0.815 | 0.860 | 0.840 | 0.892 |
| qwen8b_hybrid_rerank | 08-18 | 0.829 | 0.818 | 0.859 | 0.841 | 0.893 |
| qwen8b_hybrid_rerank_4b | 08-19 | 0.891 | — | — | — | **0.937** |
| nemotron_hybrid_rerank | 08-17 | 0.803 | 0.776 | 0.825 | 0.813 | 0.860 |
| nemotron8b_hybrid_rerank | 08-18 | 0.832 | 0.822 | 0.861 | 0.844 | 0.893 |

### 4.6 Speed, ms/query

08-14 figures are total seconds / questions scored; later runs measured per
query. Shared GPU, so indicative only.

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| qwen | 08-14 | 3.3 | 2.7 | 2.4 | 2.2 | 2.1 |
| qwen4b | 08-18 | 19.0 | 8.0 | 7.5 | 6.9 | 7.0 |
| qwen8b | 08-18 | 17.1 | 12.9 | 15.3 | 20.1 | 10.9 |
| nemotron | 08-14 | 3.3 | 2.5 | 2.1 | 2.1 | 2.0 |
| nemotron8b | 08-18 | 11.2 | 8.4 | 7.7 | 7.2 | 7.1 |
| nemotron_hybrid | 08-17 | 2.0 | 2.9 | 1.9 | 2.3 | 1.5 |
| qwen_hybrid_rerank | 08-14 | 842.0 | 292.0 | 500.2 | 502.5 | 912.6 |
| qwen4b_hybrid_rerank | 08-18 | 1,133.6 | 287.4 | 496.0 | 498.4 | 902.2 |
| qwen8b_hybrid_rerank | 08-18 | 692.1 | 463.2 | 929.9 | 676.2 | 903.8 |
| qwen8b_hybrid_rerank_4b | 08-19 | 2,713.7 | — | — | — | 3,562.9 |
| nemotron_hybrid_rerank | 08-17 | 713.9 | 307.1 | 549.6 | 545.9 | 1,006.2 |
| nemotron8b_hybrid_rerank | 08-18 | 702.7 | 290.2 | 501.0 | 503.2 | 917.4 |

### 4.7 hit@5 by question type (runs that published a split)

| method | variant | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---|---:|---:|---:|---:|---:|
| qwen4b | base | 0.866 | 0.971 | 0.869 | 0.909 | 0.637 |
| qwen4b | tok256 | 0.840 | 0.955 | 0.929 | 0.859 | 0.785 |
| qwen4b | tok512 | 0.879 | 0.969 | 0.934 | 0.895 | 0.852 |
| qwen4b | tok512ov | 0.859 | 0.964 | 0.929 | 0.895 | 0.816 |
| qwen4b | tok1024 | 0.914 | 0.983 | 0.957 | 0.913 | 0.842 |
| qwen8b | base | 0.896 | 0.976 | 0.902 | 0.944 | 0.715 |
| qwen8b | tok256 | 0.874 | 0.966 | 0.938 | 0.878 | 0.824 |
| qwen8b | tok512 | 0.897 | 0.980 | 0.950 | 0.921 | 0.889 |
| qwen8b | tok512ov | 0.885 | 0.979 | 0.940 | 0.911 | 0.853 |
| qwen8b | tok1024 | 0.951 | 0.990 | 0.970 | 0.940 | 0.875 |
| nemotron8b | base | 0.954 | 0.989 | 0.944 | 0.988 | 0.833 |
| nemotron8b | tok256 | 0.909 | 0.983 | 0.971 | 0.937 | 0.899 |
| nemotron8b | tok512 | 0.958 | 0.994 | 0.979 | 0.966 | 0.942 |
| nemotron8b | tok512ov | 0.931 | 0.994 | 0.973 | 0.965 | 0.924 |
| nemotron8b | tok1024 | 0.971 | 0.997 | 0.974 | 0.973 | 0.934 |
| nemotron_hybrid | base | 0.793 | 0.899 | 0.766 | 0.825 | 0.453 |
| nemotron_hybrid | tok256 | 0.696 | 0.867 | 0.786 | 0.627 | 0.578 |
| nemotron_hybrid | tok512 | 0.734 | 0.907 | 0.891 | 0.711 | 0.663 |
| nemotron_hybrid | tok512ov | 0.749 | 0.915 | 0.849 | 0.692 | 0.650 |
| nemotron_hybrid | tok1024 | 0.823 | 0.924 | 0.894 | 0.749 | 0.730 |
| qwen4b_hybrid_rerank | base | 0.890 | 0.989 | 0.906 | 0.967 | 0.795 |
| qwen4b_hybrid_rerank | tok256 | 0.816 | 0.970 | 0.939 | 0.888 | 0.857 |
| qwen4b_hybrid_rerank | tok512 | 0.881 | 0.983 | 0.970 | 0.947 | 0.919 |
| qwen4b_hybrid_rerank | tok512ov | 0.840 | 0.988 | 0.969 | 0.936 | 0.899 |
| qwen4b_hybrid_rerank | tok1024 | 0.938 | 0.987 | 0.990 | 0.963 | 0.947 |
| qwen8b_hybrid_rerank | base | 0.890 | 0.986 | 0.912 | 0.968 | 0.812 |
| qwen8b_hybrid_rerank | tok256 | 0.823 | 0.971 | 0.937 | 0.895 | 0.861 |
| qwen8b_hybrid_rerank | tok512 | 0.874 | 0.984 | 0.968 | 0.952 | 0.927 |
| qwen8b_hybrid_rerank | tok512ov | 0.844 | 0.987 | 0.969 | 0.936 | 0.889 |
| qwen8b_hybrid_rerank | tok1024 | 0.938 | 0.987 | 0.993 | 0.970 | 0.957 |
| qwen8b_hybrid_rerank_4b | base | 0.959 | 0.992 | 0.949 | 0.988 | 0.867 |
| qwen8b_hybrid_rerank_4b | tok1024 | **0.979** | 0.993 | 0.993 | **0.990** | **0.970** |
| nemotron_hybrid_rerank | base | 0.858 | 0.973 | 0.893 | 0.929 | 0.734 |
| nemotron_hybrid_rerank | tok256 | 0.760 | 0.952 | 0.924 | 0.789 | 0.771 |
| nemotron_hybrid_rerank | tok512 | 0.816 | 0.974 | 0.964 | 0.875 | 0.837 |
| nemotron_hybrid_rerank | tok512ov | 0.805 | 0.978 | 0.963 | 0.861 | 0.839 |
| nemotron_hybrid_rerank | tok1024 | 0.885 | 0.970 | 0.987 | 0.906 | 0.868 |
| nemotron8b_hybrid_rerank | base | 0.900 | 0.992 | 0.915 | 0.973 | 0.815 |
| nemotron8b_hybrid_rerank | tok256 | 0.825 | 0.974 | 0.939 | 0.895 | 0.871 |
| nemotron8b_hybrid_rerank | tok512 | 0.881 | 0.984 | 0.966 | 0.952 | 0.935 |
| nemotron8b_hybrid_rerank | tok512ov | 0.846 | 0.988 | 0.969 | 0.936 | 0.912 |
| nemotron8b_hybrid_rerank | tok1024 | 0.930 | 0.990 | 0.987 | 0.970 | 0.954 |

### 4.8 What chunk size costs and buys, retrieval held constant

Two measurements from `experiments/indexing/README.md` that separate chunk size
from retrieval quality. First, span coverage with retrieval held perfect (every
variant handed the chunk that best covers the answer region of each of the 3,476
`base`-anchored questions; every row scores hit@1 = 1.000, so chunk metrics
cannot tell them apart). The chunk counts are from an earlier variants database
than §4.1.

| variant | chunks | char_recall@1 | char_precision@1 | budget_recall@4000 |
|---|---:|---:|---:|---:|
| base (the ruler) | 726 | 1.000 | 1.000 | 1.000 |
| tok1024 | 618 | 0.857 | 0.637 | 0.865 |
| tok512ov | 1,271 | 0.694 | 0.907 | 0.935 |
| tok512 | 1,220 | 0.694 | 0.913 | 0.958 |
| tok256 | 2,609 | 0.424 | 0.981 | 0.972 |

Second, the share of the whole index one query returns, with `char_recall@10`
held at 0.9 for every row (measured on `.local/db/pages-variants.db`; lower is
better):

| variant | chunks | indexed chars | store_share@1 | store_share@10 | recall_per_share@10 |
|---|---:|---:|---:|---:|---:|
| base | 726 | 1,064,470 | 0.235% | 1.167% | 0.771 |
| tok256 | 2,003 | 1,077,031 | 0.046% | 0.471% | **1.909** |
| tok512 | 990 | 1,078,330 | 0.099% | 1.066% | 0.844 |
| tok512ov | 1,049 | 1,151,174 | 0.093% | 0.951% | 0.947 |
| tok1024 | 501 | 1,079,332 | 0.280% | 2.045% | 0.440 |

`tok1024` reads 4.3× more of its index than `tok256` for the same answer
coverage, and overlap costs `tok512ov` 6.8% more embedded text. On the
2026-08-31 and 09-01 agentic runs `store_share@10` on `base` was 1.402–1.429 for
the reranked methods and 1.358 for `dci`.

Density-mode question generation (one question per 256 tokens of chunk, so every
cutting is probed at the same density), measured over 163 real chunks:

| variant | requested | delivered | per chunk |
|---|---:|---:|---:|
| tok256 | 49 | 49 | 1.00 |
| tok512 | 114 | 112 | 1.87 |
| tok1024 | 215 | 172 | 4.06 |

The per-variant design probed `tok256` at 17.6 questions per 10,000 indexed
characters against 4.2 for `tok1024` (7,806 against 1,456 questions on the sweep
database), which is why `tok1024` winning every row above cannot be separated
from its smaller, more salient question set.

What era D established: the reranker is the dominant factor (every embedder
lands at 0.88–0.92 hit@5 on `base` with the 0.6B reranker, whatever it scores
alone); `nemotron8b` alone (0.940, 11 ms) beats every 0.6B-reranked pipeline;
the 4B reranker adds 4 points on top of the best dense model; bigger chunks score
higher for every method but on fewer, easier questions; `tok256` is never better
than `base`; Nemotron's model card lists 42 languages without Slovenian, so the
nemotron-to-qwen gap is not evidence about scale.

## 5. Era E — the agentic family by judge model, 495 questions (31 Aug – 7 Sep 2026)

Shared design, `base` cutting, `qwen` (0.6B) embedder + Qwen3-Reranker-0.6B
first stage, first 500 questions by id with 5 warm-up, store `.local/chroma-sweep`.
Three generations of "let an LLM steer retrieval": the retry loop
(`qwen_hybrid_agentic`: judge says *sufficient*, `reformulate`, or `expand`, up
to 3 attempts), page tools (`qwen_hybrid_agentic_tools`: plus `list_sections`
and `search_in_page`, found chunks inserted after the ranked chunk of the same
page), and the corpus agent (`dci`: no vector index, BM25 shortlist written to
disk, `search` / `read` / `toc` then `answer` with chunk ids, 8-step ceiling).
Judges: local `gpt-oss:20b` (2026-08-31 and 09-01 runs), local `gemma4:31b` (two
runs, no agent cell completed), DeepSeek-V4-Pro on Azure (2026-09-07). Sources:
`agentic-tools-2026-08-31.md`, `agentic-dci-2026-09-01.md`,
`agentic-retrieval-report-2026-09-01.md`, `agentic-deepseek-2026-09-07.md`,
`agentic-judge-comparison-2026-09-07.md`, the deleted `*-qwen-format.md`
renderings (commit `998dd60`).

### 5.1 Overall

`depth` is the deepest rank an expanding agent returned; `hit@depth` is its
score there (not comparable across methods).

| method | judge | run | hit@1 | hit@5 | hit@10 | mrr@10 | depth | hit@depth |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen_hybrid_rerank (control) | — | 08-31 | 0.723 | 0.881 | 0.907 | 0.791 | 10 | — |
| qwen_hybrid_rerank (control) | — | 09-01 (DCI run) | 0.731 | 0.881 | 0.911 | 0.796 | 10 | — |
| qwen_hybrid_rerank (control) | — | 09-07 | 0.725 | 0.881 | 0.911 | 0.793 | 10 | — |
| qwen_hybrid_agentic | gpt-oss:20b | 08-31 | 0.725 | 0.889 | 0.915 | 0.795 | 20 | 0.917 |
| qwen_hybrid_agentic | gemma4:31b | 09-01, 09-03 | NA¹ | NA¹ | NA¹ | NA¹ | — | — |
| qwen_hybrid_agentic | DeepSeek-V4-Pro | 09-07 | 0.721 | **0.893** | **0.915** | 0.793 | 20 | 0.917 |
| qwen_hybrid_agentic_tools | gpt-oss:20b | 08-31 | 0.719 | 0.879 | 0.903 | 0.787 | 24 | 0.907 |
| qwen_hybrid_agentic_tools | gemma4:31b | 09-01, 09-03 | NA¹ | NA¹ | NA¹ | NA¹ | — | — |
| qwen_hybrid_agentic_tools | DeepSeek-V4-Pro | 09-07 | 0.719 | 0.848 | 0.899 | 0.778 | 24 | 0.909 |
| dci | gpt-oss:20b | 09-01 | 0.576 | 0.697 | 0.723 | 0.632 | 10 | — |
| dci | gemma4:31b | 09-01, 09-03 | NA¹ | NA¹ | NA¹ | NA¹ | — | — |
| dci | DeepSeek-V4-Pro | — | skipped² | skipped² | skipped² | skipped² | — | — |

Deltas against the control measured in the same run, hit@5: retry loop +0.8
(gpt-oss), +1.2 (DeepSeek); page tools −0.2 (gpt-oss), −3.2 (DeepSeek); DCI
−18.4 (gpt-oss). Only the DCI and DeepSeek page-tool deltas exceed the noise
floor.

### 5.2 Speed

The control took 339 s in the 08-31 run and 713 s in the 09-01 run on the same
questions because three other users were on the GPU; DCI's 22,084 ms/query is
15.5× its own run's control and 32.7× the uncontended control.

| method | judge | run | seconds | ms/query | queries/query | queries | judge/schema failures |
|---|---|---|---:|---:|---:|---:|---|
| qwen_hybrid_rerank | — | 08-31 | 339.0 | 675.4 | 1.00 | 495 | — |
| qwen_hybrid_rerank | — | 09-01 | 712.8 | 1,423.8 | 1.00 | 495 | — |
| qwen_hybrid_rerank | — | 09-07 | 338.0 | 673.3 | 1.00 | 495 | — |
| qwen_hybrid_agentic | gpt-oss:20b | 08-31 | 3,367.1 | 6,062.1 | 1.21 | 597 | 0 of 597 calls |
| qwen_hybrid_agentic | DeepSeek | 09-07 | 2,923.3 | 5,170.3 | 1.34 | 664 | 0 |
| qwen_hybrid_agentic_tools | gpt-oss:20b | 08-31 | 3,613.4 | 6,572.2 | 1.25 | 617 | 44 of 617 calls (7.1%), silent fallback to `expand` |
| qwen_hybrid_agentic_tools | DeepSeek | 09-07 | 3,355.6 | 6,070.9 | 1.52 | 752 | 0 |
| dci | gpt-oss:20b | 09-01 | 11,051.2 | 22,084.3 | 5.85 | 2,898 | 35 of 495 questions abandoned (7.1%) |

DeepSeek made about 1,400 judge calls across its run with zero failures at about
1.5 s per call and no GPU; gpt-oss ran at about 1 s per call on 13 GB of the shared
GPU; gemma needs 28 GB at about 9 s per call.

### 5.3 hit@5 by question type

| method | judge | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---|---:|---:|---:|---:|---:|
| qwen_hybrid_rerank (08-31) | — | 0.857 | 0.940 | 0.905 | 0.963 | 0.752 |
| qwen_hybrid_rerank (09-01) | — | 0.857 | 0.940 | 0.895 | 0.972 | 0.752 |
| qwen_hybrid_rerank (09-07) | — | 0.857 | 0.940 | 0.895 | 0.972 | 0.752 |
| qwen_hybrid_agentic | gpt-oss | **0.901** | **0.952** | 0.895 | 0.953 | 0.761 |
| qwen_hybrid_agentic | DeepSeek | 0.890 | 0.940 | 0.886 | **0.981** | **0.780** |
| qwen_hybrid_agentic_tools | gpt-oss | 0.857 | 0.940 | 0.895 | 0.963 | 0.752 |
| qwen_hybrid_agentic_tools | DeepSeek | 0.824 | 0.916 | 0.867 | 0.944 | 0.706 |
| dci | gpt-oss | 0.352 | 0.892 | 0.800 | 0.813 | 0.624 |

### 5.4 What the agents did, paired against the control at hit@5

Until 2026-09-03 the sweep passed the action log to the diagnostics without the
rankings it is aligned by, so every gpt-oss report printed 0 retries and 0 tool
calls for agents that made hundreds (`n/r`); recovered and lost come from the
rankings and are valid in every run.

| method | judge | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | net | rankings changed | improved / worsened (any rank) | sign test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen_hybrid_agentic | gpt-oss | n/r | n/r | n/r | n/r | n/r | 9 | 5 | +4 | 34 / 495 (6.9%) | 18 / 16 | p = 0.42 |
| qwen_hybrid_agentic | DeepSeek | 101 (20.4%) | 99 | 9 | 0.139 | 0.729 | 8 | 2 | +6 | — | — | — |
| qwen_hybrid_agentic_tools | gpt-oss | n/r | n/r | n/r | n/r | n/r | 2 | 3 | −1 | 24 / 495 (4.8%) | 10 / 14 | p = 1.00 |
| qwen_hybrid_agentic_tools | DeepSeek | 109 (22.0%) | 15 | 95 | 0.037 | 0.695 | 3 | 19 | −16 | — | — | — |

Page-tool usage under DeepSeek: 96 of 495 questions (19.4%) triggered a tool,
259 calls, 257 `search_in_page` and 2 `list_sections`; on those 96 questions the
agent recovered 2 and lost 19 (tool precision 0.021). DCI: mean 5.85 of 8 steps
per question; the 35 abandoned questions split into 16 "no structured output",
7 `chunk_ids` returned as null, 5 unknown tool `answer`, 5 unknown tool `search`,
2 unknown tool `toc` / `read` (the model tried to call the four actions as four
tools rather than one `CorpusAction`).

### 5.5 The retry loop's delta against its own baseline, every run that has one

Absolute scores are not comparable across rows (different samples, embedders,
judges); the delta is, because each agent is differenced against the baseline
measured beside it. Deltas in points of hit@5.

| run | n | judge | first stage | agent | baseline | agent | delta |
|---|---:|---|---|---|---:|---:|---:|
| 2026-06-23 (§2, broken dense stage) | 909 | DeepSeek | qwen_hybrid | qwen_hybrid_agentic | 0.311 | 0.327 | +1.6 |
| 2026-06-23 (§2, broken dense stage) | 909 | DeepSeek | qwen | qwen_agentic | 0.189 | 0.205 | +1.6 |
| 2026-07-23b | 199 | DeepSeek | azure_hybrid_rerank | azure_hybrid_agentic | 0.899 | 0.920 | +2.1 |
| 2026-07-23b | 199 | DeepSeek | nemotron_hybrid_rerank | nemotron_hybrid_agentic | 0.849 | 0.864 | +1.5 |
| 2026-07-23a | 260 | DeepSeek | azure_hybrid_rerank | azure_hybrid_agentic | 0.900 | 0.919 | +1.9 |
| 2026-07-23a | 260 | DeepSeek | nemotron_hybrid_rerank | nemotron_hybrid_agentic | 0.850 | 0.865 | +1.5 |
| 2026-07-22 | 264 | DeepSeek | azure_hybrid_rerank | azure_hybrid_agentic | 0.898 | 0.917 | +1.9 |
| 2026-07-22 | 264 | DeepSeek | nemotron_hybrid_rerank | nemotron_hybrid_agentic | 0.848 | 0.864 | +1.6 |
| 2026-07-27 | 847 | DeepSeek | embed_v4_hybrid_rerank | embed_v4_hybrid_agentic | 0.903 | 0.911 | +0.8 |
| 2026-07-27 | 847 | DeepSeek | nemotron_hybrid_rerank | nemotron_hybrid_agentic | 0.874 | 0.884 | +1.0 |
| 2026-08-31 | 495 | gpt-oss | qwen_hybrid_rerank | qwen_hybrid_agentic | 0.881 | 0.889 | +0.8 |
| 2026-08-31 | 495 | gpt-oss | qwen_hybrid_rerank | qwen_hybrid_agentic_tools | 0.881 | 0.879 | −0.2 |
| 2026-09-07 | 495 | DeepSeek | qwen_hybrid_rerank | qwen_hybrid_agentic | 0.881 | 0.893 | +1.2 |
| 2026-09-07 | 495 | DeepSeek | qwen_hybrid_rerank | qwen_hybrid_agentic_tools | 0.881 | 0.848 | −3.2 |
| 2026-09-08 (§6, hard geo stage) | 500 | DeepSeek | qwen_hybrid_rerank_geo | qwen_hybrid_agentic_geo | 0.856 | 0.866 | +1.0 |
| 2026-09-08 (§6, hard geo stage) | 500 | DeepSeek | qwen_hybrid_rerank_geo | qwen_hybrid_agentic_tools_geo | 0.856 | 0.856 | 0.0 |
| 2026-09-09 (§6, soft geo stage) | 500 | DeepSeek | qwen_hybrid_rerank_geo | qwen_hybrid_agentic_geo | 0.884 | 0.888 | +0.4 |
| 2026-09-09 (§6, soft geo stage) | 500 | DeepSeek | qwen_hybrid_rerank_geo | qwen_hybrid_agentic_tools_geo | 0.884 | 0.882 | −0.2 |

The retry loop has never scored below its baseline: +0.4 to +2.1 points across
five embedders, two judges, six question samples and three months, with the gain
concentrated on `crosslingual` (+4.4 under gpt-oss). The page tools have never
scored above theirs.

### Notes to §5

¹ gemma4:31b never completed an agent cell. Two runs each finished only the
control (hit@5 0.883 and 0.881). The first died after 9.5 h with 136 judge
failures: gemma's thinking consumed the 1,024-token output cap on real prompts
of 4k–11k tokens and returned empty output, which the retriever treated as
"insufficient" and widened, issuing more failing calls. With the corrected
configuration (thinking off, function-calling output) the second run was killed
twice by the server ending the user session at logout and was then stopped on
request. Parked, not disproven. Gemma's `think` is boolean: `low` and `high`
produce byte-identical output.

² DCI with DeepSeek was skipped on request to free the machine. Its failure is
structural: a grep-only agent cannot match a Slovenian answer to an English
question (crosslingual 0.857 → 0.352), and a judge swap does not change that.

Earlier agentic experiments without a recorded sample size (July,
`agentic-findings.md`): RRF-merging distinct rerankers lifted recall@10 from
about 0.79 to 0.83 at a small hit@1 cost; LLM multi-query expansion did not help
and cost 3–5×; an LLM answerability gate asked back on about 17% of questions
(42% of `vague_short`) and lifted `vague_short` 0.76 → 0.83; a BERT-NER gate's
denial precision capped at about 24%.

## 6. Era F — geography, 500 questions (8–9 September 2026)

Shared design, `base`, `qwen` embedder + 0.6B reranker, DeepSeek judge for the
agents, Azure resolver for the question's place scope, `data/db/pages.db`. 89 of
176 pages carry a primary location. The resolver scoped the same 72 questions in
every run (53 radius, 10 country, 6 NUTS-3, 3 NUTS-2; warm cache). Three runs:
2026-09-08 (`*_geo` = hard filter to the scope, widen when too few candidates,
boost; `_v3` a variant of the same), 2026-09-09 (`*_geo` = soft: over-fetch,
fuse the distance score, unknown footprint neutral; `_geo_strict` = the hard
filter, which reproduced 09-08 because cached scopes carried
`include_null: false`), and 2026-09-09-strict (the hard filter with unlocated
pages allowed through). Zero judge failures in any run. Sources:
`geo-run-2026-09-08.md`, `geo-run-2026-09-09.md`, `geo-run-2026-09-09-strict.md`,
`geo-soft-vs-strict-2026-09-09.md`.

### 6.1 Overall

| run | method | geo shape | hit@1 | hit@5 | hit@10 | mrr@10 | hit@20 |
|---|---|---|---:|---:|---:|---:|---:|
| 09-08 | qwen_hybrid_rerank | none (text baseline) | 0.718 | 0.884 | 0.900 | 0.787 | — |
| 09-08 | qwen_hybrid_geo | hard filter, no reranker | 0.552 | 0.746 | 0.802 | 0.637 | — |
| 09-08 | qwen_hybrid_rerank_geo | hard filter | 0.696 | 0.856 | 0.872 | 0.763 | — |
| 09-08 | qwen_hybrid_rerank_geo_v3 | hard filter, v3 | 0.692 | 0.852 | 0.868 | 0.759 | — |
| 09-08 | qwen_hybrid_agentic_geo | agent on hard filter | 0.700 | 0.866 | 0.884 | 0.769 | 0.884 |
| 09-08 | qwen_hybrid_agentic_tools_geo | agent + tools on hard filter | 0.700 | 0.856 | 0.878 | 0.767 | 0.878 |
| 09-09 | qwen_hybrid_rerank | none | 0.718 | 0.884 | 0.900 | 0.787 | — |
| 09-09 | qwen_hybrid_rerank_geo | **soft** | 0.718 | 0.884 | 0.900 | 0.788 | — |
| 09-09 | qwen_hybrid_rerank_geo_strict | hard filter, cached `include_null: false` (replication of 09-08) | 0.696 | 0.856 | 0.872 | 0.763 | — |
| 09-09 | qwen_hybrid_geo | soft, no reranker | 0.576 | 0.776 | 0.832 | 0.664 | — |
| 09-09 | qwen_hybrid_agentic_geo | agent on soft | **0.736** | **0.888** | **0.906** | **0.800** | 0.906 |
| 09-09 | qwen_hybrid_agentic_tools_geo | agent + tools on soft | 0.720 | 0.882 | 0.902 | 0.789 | 0.902 |
| 09-09-strict | qwen_hybrid_rerank | none | 0.718 | 0.884 | 0.900 | 0.787 | — |
| 09-09-strict | qwen_hybrid_rerank_geo_strict | hard filter, `include_null: true` | 0.716 | 0.884 | 0.900 | 0.786 | — |

### 6.2 Speed

| run | method | seconds | ms/query | queries/query | queries | chunk expansions |
|---|---|---:|---:|---:|---:|---:|
| 09-08 | qwen_hybrid_rerank | 539.4 | 1,078.8 | 1.00 | 500 | 0 |
| 09-08 | qwen_hybrid_geo | 997.2 | 1,994.4 | 1.00 | 500 | 0 |
| 09-08 | qwen_hybrid_rerank_geo | 540.1 | 1,080.2 | 1.00 | 500 | 0 |
| 09-08 | qwen_hybrid_rerank_geo_v3 | 476.4 | 952.8 | 1.00 | 500 | 0 |
| 09-08 | qwen_hybrid_agentic_geo | 2,892.1 | 5,784.3 | 1.39 | 695 | 21 |
| 09-08 | qwen_hybrid_agentic_tools_geo | 2,531.0 | 5,062.1 | 1.35 | 673 | 32 |
| 09-09 | qwen_hybrid_rerank | 427.8 | 855.5 | 1.00 | 500 | 0 |
| 09-09 | qwen_hybrid_rerank_geo | 414.6 | 829.1 | 1.00 | 500 | 0 |
| 09-09 | qwen_hybrid_rerank_geo_strict | 407.4 | 814.7 | 1.00 | 500 | 0 |
| 09-09 | qwen_hybrid_geo | 4.0 | 8.1 | 1.00 | 500 | 0 |
| 09-09 | qwen_hybrid_agentic_geo | 3,310.6 | 6,621.3 | 1.34 | 669 | 13 |
| 09-09 | qwen_hybrid_agentic_tools_geo | 2,659.2 | 5,318.4 | 1.30 | 648 | 34 |
| 09-09-strict | qwen_hybrid_rerank | 720.0 | 1,440.0 | 1.00 | 500 | 0 |
| 09-09-strict | qwen_hybrid_rerank_geo_strict | 784.2 | 1,568.4 | 1.00 | 500 | 0 |

### 6.3 hit@5 by question type

| run | method | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---|---:|---:|---:|---:|---:|
| all three | qwen_hybrid_rerank | 0.891 | 0.940 | 0.887 | 0.981 | 0.741 |
| 09-08 | qwen_hybrid_geo | 0.728 | 0.880 | 0.821 | 0.841 | 0.500 |
| 09-08 | qwen_hybrid_rerank_geo (hard) | 0.837 | 0.892 | 0.877 | 0.953 | 0.732 |
| 09-08 | qwen_hybrid_rerank_geo_v3 | 0.815 | 0.916 | 0.887 | 0.953 | 0.705 |
| 09-08 | qwen_hybrid_agentic_geo | 0.848 | 0.892 | 0.887 | 0.963 | 0.750 |
| 09-08 | qwen_hybrid_agentic_tools_geo | 0.859 | 0.880 | 0.877 | 0.944 | 0.732 |
| 09-09 | qwen_hybrid_rerank_geo (soft) | 0.891 | 0.940 | 0.887 | 0.981 | 0.741 |
| 09-09 | qwen_hybrid_rerank_geo_strict (cached null) | 0.837 | 0.892 | 0.877 | 0.953 | 0.732 |
| 09-09 | qwen_hybrid_geo (soft) | 0.772 | 0.928 | 0.830 | 0.869 | 0.527 |
| 09-09 | qwen_hybrid_agentic_geo | **0.902** | 0.940 | 0.887 | 0.981 | **0.750** |
| 09-09 | qwen_hybrid_agentic_tools_geo | 0.891 | 0.928 | 0.887 | 0.981 | 0.741 |
| 09-09-strict | qwen_hybrid_rerank_geo_strict (include_null) | 0.891 | 0.940 | 0.887 | 0.981 | 0.741 |

### 6.4 Agents on the geo stage, paired against `qwen_hybrid_rerank_geo` of the same run at hit@5

| run | agent | retried | rewritten | expanded | retry precision | retry recall | recovered@5 | lost@5 | retry ms | no-retry ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 09-08 | qwen_hybrid_agentic_geo | 115 (23.0%) | 109 | 15 | 0.096 | 0.778 | 6 | 1 | 13,325.6 | 3,531.7 |
| 09-08 | qwen_hybrid_agentic_tools_geo | 79 (15.8%) | 19 | 25 | 0.038 | 0.528 | 3 | 3 | 12,536.9 | 3,659.4 |
| 09-09 | qwen_hybrid_agentic_geo | 102 (20.4%) | 101 | 12 | 0.127 | 0.707 | 3 | 1 | 13,940.5 | 4,745.5 |
| 09-09 | qwen_hybrid_agentic_tools_geo | 71 (14.2%) | 11 | 25 | 0.028 | 0.483 | 2 | 3 | 14,216.5 | 3,845.8 |

Page-tool usage:

| run | tool questions | tool calls | list_sections | search_in_page | recovered@5 | lost@5 | tool precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| 09-08 | 68 (13.6%) | 157 | 0 | 157 | 1 | 3 | 0.015 |
| 09-09 | 65 (13.0%) | 149 | 0 | 149 | 2 | 3 | 0.031 |

### 6.5 Geo scope resolution

`scoped` questions resolved to a place; `widened` needed a wider scope before
enough candidates came back; `levels used` is the level that produced the final
candidates. Agents count every retrieval call, so their question totals exceed 500.

| run | method | calls | scoped | scoped % | widened | levels used |
|---|---|---:|---:|---:|---:|---|
| 09-08 | qwen_hybrid_geo, _rerank_geo, _rerank_geo_v3 | 500 | 72 | 14% | 5 | country 14, none 1, nuts2 1, nuts3 4, radius 52 |
| 09-08 | qwen_hybrid_agentic_geo | 695 | 142 | 20% | 11 | country 36, none 1, nuts2 1, nuts3 4, radius 100 |
| 09-08 | qwen_hybrid_agentic_tools_geo | 544 | 90 | 17% | 5 | country 21, none 1, nuts2 1, nuts3 4, radius 63 |
| 09-09 | qwen_hybrid_rerank_geo (soft), qwen_hybrid_geo | 500 | 72 | 14% | 14 | none 14, nuts2 3, nuts3 2, radius 53 |
| 09-09 | qwen_hybrid_rerank_geo_strict (cached null) | 500 | 72 | 14% | 5 | country 14, none 1, nuts2 1, nuts3 4, radius 52 |
| 09-09 | qwen_hybrid_agentic_geo | 669 | 109 | 16% | 24 | none 24, nuts2 3, nuts3 2, radius 80 |
| 09-09 | qwen_hybrid_agentic_tools_geo | 521 | 76 | 15% | 16 | none 16, nuts2 3, nuts3 2, radius 55 |
| 09-09-strict | qwen_hybrid_rerank_geo_strict (include_null) | 500 | 72 | 14% | 16 | country 11, none 12, nuts2 1, nuts3 3, radius 45 |

Under the soft method "none" for 14 of 72 means the selectivity gate skipped the
10 country scopes and 4 of the 9 region scopes because they covered more than 90%
of located pages; all 53 point scopes were scored by distance.

### 6.6 Scoped against unscoped (the WP1 gate), hit@5

Wins and losses are per question against `qwen_hybrid_rerank` on the 72 scoped
questions.

| method | shape | all (500) | scoped (72) | unscoped (428) | wins | losses |
|---|---|---:|---:|---:|---:|---:|
| qwen_hybrid_rerank | text baseline | 0.884 | 0.917 | 0.879 | | |
| qwen_hybrid_rerank_geo (09-08) | hard filter, widen on count, boost | 0.856 | 0.722 | 0.879 | 1 | 15 |
| qwen_hybrid_rerank_geo (09-09) | **soft: over-fetch, fuse, unknown neutral** | **0.884** | **0.917** | 0.879 | 0 | 0 |
| qwen_hybrid_rerank_geo_strict, cached null | hard filter (replication) | 0.856 | 0.722 | 0.879 | 1 | 15 |
| qwen_hybrid_rerank_geo_strict, include_null | hard filter, unlocated pages pass | 0.884 | 0.917 | 0.879 | 0 | 0 |
| qwen_hybrid_geo (09-09) | soft, no reranker | 0.776 | 0.875 | 0.759 | 0 | 3 |
| qwen_hybrid_agentic_geo (09-09) | agent on soft | **0.888** | 0.917 | **0.883** | 0 | 0 |
| qwen_hybrid_agentic_tools_geo (09-09) | agent + tools on soft | 0.882 | 0.903 | 0.879 | 0 | 1 |

| gate check | baseline | soft geo | hard filter |
|---|---:|---:|---:|
| the 16 scoped questions whose gold pages have no footprint | 16 / 16 | 16 / 16 | 1 / 16 |
| near-wording questions in the sample (blizu, okolica, near, around) | 7 / 8 | 7 / 8 | 7 / 8 |

What era F established: the 15 losses of the hard filter were coverage losses
(gold page unlocated and filtered out), not ranking losses; neutral-for-unknown
removes all of them whether geography filters or scores; the soft stage costs
nothing measurable over the baseline (829 against 856 ms); geography cannot yet
*help* on this corpus (15 distinct coordinate pairs, 85 of 89 located pages in one
NUTS-3 region, 37 near-wording questions in 3,476), which is what motivated §7.4.

## 7. Era G — enrichment, gazetteer misses, corpus expansion, seed quality

### 7.1 Enrichment coverage of the 176-page corpus (2026-09-08 apply run)

| measure | value |
|---|---:|
| pages with a primary location | 89 of 176 (74 per-source default, 13 Wikidata, 2 LLM + Nominatim) |
| pages unlocated by design (biographies, concepts, registers, clubs) | 87 |
| mentioned-place rows attached | 375 |
| located pages within 2 km / 25 km of Rajhenburg Castle | 80 / 86 of 89 |
| located pages in one NUTS-3 region (Posavska) | 85 of 89 |
| distinct coordinate pairs among primaries | 15 |
| mentioned rows more than 25 km away | 204 of 375 |
| mentioned rows geocoded inside Slovenia | 374 of 375 |

The last row is wrong data: with `geo_country_hint: si` Nominatim returns the
Slovenian hamlet that carries a Slovenian exonym, and the name check passes
because the token is identical (Gradec → 45.465, 13.904 instead of Graz; Dunaj →
46.319, 14.046 instead of Vienna; Rim → 45.540, 15.293 instead of Rome; Trst →
45.610, 13.916 instead of Trieste; Gorica → 46.298, 15.254). These rows reach no
filter yet (only primaries reach chunk metadata) but must be re-resolved before
WP2 makes them scoreable.

### 7.2 Gazetteer misses classified (2026-09-09, dry run: 88 primaries, 366 mentioned rows, 147 "no hit" + 4 "name mismatch")

151 distinct rejected names over 160 page mentions, classified by the cheapest
fix, four classifier passes (categories exclusive, cheapest first):

| category | pass 1 | pass 2 | pass 3 | **pass 4 (final)** | mentions | meaning |
|---|---:|---:|---:|---:|---:|---|
| corpus | — | 31 | 17 | **14** | 16 | head of the name is a primary location or NUTS region the corpus already knows |
| qualifier | 42 | 12 | 16 | **17** | 17 | a type word or trailing qualifier hid the place |
| inflected | 3 | 0 | 0 | **0** | 0 | Slovenian oblique form; none survived tightening |
| outside_hint | 43 | 39 | 40 | **42** | 44 | resolves once `geo_country_hint: si` is lifted |
| wikidata | 7 | 42 | 51 | **51** | 56 | Nominatim has nothing; Slovenian-language Wikidata search has coordinates (exonyms, historical names) |
| unresolved | 56 | 27 | 27 | **27** | 27 | hamlets, Roman-era sites, geological features |
| total | 151 | 151 | 151 | **151** | 160 | |

124 of 151 names (82%) resolve with no new model and no new gazetteer; 86 of
those 124 are places abroad. The country hint, not Slovenian morphology, is the
cause of the misses.

### 7.3 Where `qwen_hybrid_rerank` misses, 300 questions (motivation for the page tools)

| outcome | share |
|---|---:|
| gold chunk in top 10 | 75.3% |
| miss, but the gold chunk's page is in the ranking (recoverable by page tools) | 13.3% |
| miss, page absent (out of reach) | 11.3% |

### 7.4 Corpus expansion (2026-09-13)

Nine localities of the same shape (attraction site, town site, national
biographical lexicon, national-language Wikipedia) added around Brestanica, which
stays untouched as the gold standard.

| measure | before | after pruning |
|---|---:|---:|
| pages | 176 | 1,338 |
| characters | 1.08 M | 14.5 M |
| base chunks | 726 | 11,261 |
| sources | 4 | 38 |
| languages | 1 (nominal) | 8 |
| pages with a primary location | 89 (SI only) | 690 in 8 countries (602 of 1,159 new pages via Wikidata or source default; LLM + Nominatim tier not yet run) |
| NUTS-3 regions with ≥ 58 located pages | 1 | 8 (SI036, SI032, HR064, AT224, HU222, CZ064, SK022, DE214) |
| share of the store a `tok1024` query reads at k=10 | 3.8% | about 0.3% |

Pruning removed, from new sources only, 151 pages with under 300 characters of
prose (mostly Czech lexicon index records) and 91 pages whose content repeated
within the source (cookie notices and navigation blocks). Miramare's 35 pages
have coordinates but no NUTS code (the point sits on the coast outside every
polygon). Twelve `castle_rajhenburg` pages are English although every source is
stamped `sl`.

### 7.5 Seed quality against the Brestanica reference (2026-09-13)

Prose is the text left after stripping links, images, markup and mostly
non-letter lines; a stub has under 300 prose characters. Verdicts: `thin` when
median prose is under 50% of the reference of the same kind, `stubs` when the stub
share is over 2× the reference and over 10%, `duplicates` likewise for repeated
content hashes, `language` when under 60% of pages detect as the declared
language.

Reference (gold standard, unchanged by pruning except Wikipedia 49 → 51 pages):

| source | kind | pages | median chars | median prose | prose share | stubs | non-prose | duplicates | language match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| castle_rajhenburg | attraction | 43 | 1,196 | 757 | 87% | 33% | 21% | 16% | 64% |
| brestanica_webpage | town | 31 | 701 | 590 | 86% | 32% | 3% | 0% | 100% |
| svn_biography | biography | 51 | 2,512 | 2,502 | 98% | 0% | 0% | 0% | 100% |
| wikipedia | wikipedia | 51 | 5,390 | 2,908 | 57% | 2% | 0% | 2% | 100% |

New sources after pruning (1 of 34 flagged):

| source | kind | pages | median chars | median prose | prose share | stubs | non-prose | duplicates | language match | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bojnice_castle | attraction | 29 | 1,314 | 1,180 | 95% | 0% | 0% | 0% | 100% | ok |
| burghausen_castle | attraction | 42 | 1,380 | 1,234 | 93% | 0% | 0% | 0% | 100% | ok |
| duino_castle | attraction | 2 | 969 | 931 | 96% | 0% | 0% | 0% | 100% | ok |
| jurisics_castle | attraction | 22 | 1,052 | 780 | 83% | 0% | 0% | 0% | 100% | ok |
| lednice_castle | attraction | 37 | 1,763 | 1,297 | 94% | 0% | 8% | 0% | 100% | ok |
| ptuj_museum | attraction | 30 | 1,238 | 1,072 | 82% | 0% | 0% | 0% | 100% | ok |
| riegersburg_castle | attraction | 43 | 704 | 700 | 95% | 0% | 0% | 0% | 100% | ok |
| valtice_castle | attraction | 36 | 1,567 | 1,248 | 89% | 0% | 8% | 0% | 100% | ok |
| veliki_tabor_castle | attraction | 25 | 1,744 | 1,312 | 92% | 0% | 0% | 0% | 100% | ok |
| at_riegersburg_biography | biography | 34 | 1,546 | 1,539 | 99% | 0% | 0% | 0% | 100% | ok |
| cz_lednice_biography | biography | 6 | 4,396 | 4,079 | 90% | 0% | 0% | 0% | 100% | ok |
| de_burghausen_biography | biography | 45 | 1,013 | 753 | 88% | 0% | 0% | 0% | 100% | thin: median prose 753 vs 2,502 |
| hr_zagorje_biography | biography | 42 | 6,334 | 6,198 | 98% | 0% | 0% | 0% | 100% | ok |
| hu_koszeg_biography | biography | 50 | 1,363 | 1,363 | 100% | 0% | 0% | 0% | 100% | ok |
| it_trieste_biography | biography | 50 | 13,526 | 13,382 | 99% | 0% | 0% | 0% | 100% | ok |
| si_ptuj_biography | biography | 16 | 1,919 | 1,911 | 99% | 0% | 0% | 0% | 100% | ok |
| bojnice_town | town | 28 | 5,013 | 4,634 | 90% | 0% | 0% | 0% | 100% | ok |
| burghausen_town | town | 32 | 1,363 | 1,170 | 66% | 0% | 3% | 0% | 100% | ok |
| desinic_municipality | town | 17 | 4,518 | 4,444 | 94% | 0% | 6% | 0% | 100% | ok |
| koszeg_town | town | 34 | 1,952 | 1,518 | 86% | 0% | 3% | 0% | 100% | ok |
| lednice_municipality | town | 18 | 694 | 508 | 92% | 0% | 0% | 0% | 100% | ok |
| miramare_castle | town | 25 | 3,792 | 3,531 | 96% | 0% | 4% | 0% | 100% | ok |
| ptuj_tourism | town | 22 | 552 | 449 | 88% | 0% | 0% | 0% | 100% | ok |
| riegersburg_municipality | town | 24 | 1,271 | 1,088 | 88% | 0% | 0% | 0% | 100% | ok |
| valtice_town | town | 26 | 1,509 | 1,138 | 85% | 0% | 0% | 0% | 100% | ok |
| at_riegersburg_wikipedia | wikipedia | 45 | 10,675 | 10,282 | 96% | 0% | 0% | 0% | 100% | ok |
| cz_lednice_wikipedia | wikipedia | 45 | 13,078 | 11,290 | 94% | 0% | 0% | 0% | 100% | ok |
| cz_valtice_wikipedia | wikipedia | 26 | 11,120 | 9,781 | 95% | 0% | 0% | 0% | 100% | ok |
| de_burghausen_wikipedia | wikipedia | 52 | 13,099 | 12,736 | 97% | 0% | 0% | 0% | 100% | ok |
| hr_zagorje_wikipedia | wikipedia | 52 | 2,710 | 2,399 | 92% | 0% | 0% | 0% | 100% | ok |
| hu_koszeg_wikipedia | wikipedia | 52 | 13,272 | 12,584 | 92% | 0% | 0% | 0% | 100% | ok |
| it_trieste_wikipedia | wikipedia | 40 | 13,690 | 12,386 | 93% | 0% | 0% | 0% | 100% | ok |
| si_ptuj_wikipedia | wikipedia | 45 | 5,582 | 5,159 | 94% | 0% | 0% | 0% | 100% | ok |
| sk_bojnice_wikipedia | wikipedia | 65 | 3,760 | 3,282 | 89% | 0% | 0% | 0% | 100% | ok |

Before pruning 8 of 34 sources were flagged; the rows that changed:

| source | pages before → after | median prose before → after | stubs before | duplicates before | verdict before |
|---|---|---|---:|---:|---|
| bojnice_castle | 45 → 29 | 1,014 → 1,180 | 0% | 33% | duplicates |
| duino_castle | 4 → 2 | 344 → 931 | 50% | 25% | thin |
| cz_lednice_biography | 50 → 6 | 143 → 4,079 | 88% | 0% | thin, stubs |
| de_burghausen_biography | 50 → 45 | 516 → 753 | 10% | 0% | thin (still flagged) |
| desinic_municipality | 34 → 17 | 3,088 → 4,444 | 0% | 47% | duplicates |
| lednice_municipality | 35 → 18 | 301 → 508 | 49% | 31% | duplicates |
| miramare_castle | 35 → 25 | 2,365 → 3,531 | 0% | 26% | duplicates |
| ptuj_tourism | 35 → 22 | 488 → 449 | 3% | 31% | duplicates |

Unflagged sources also lost stub pages (jurisics_castle 45 → 22 pages with 33%
stubs before; veliki_tabor_castle 45 → 25 with 31%; ptuj_museum 43 → 30 with 26%;
riegersburg_municipality 35 → 24 with 31%; valtice_town 35 → 26 with 26%;
bojnice_town 35 → 28 with 20%).

## 8. What the whole body of evidence says

1. **Hybrid retrieval plus a cross-encoder reranker is the production baseline.**
   With the 0.6B reranker every embedder lands at 0.88–0.92 hit@5 on `base`;
   `qwen8b_hybrid_rerank_4b` reaches 0.950 (0.986 on `tok1024`). `nemotron8b`
   alone is the strongest single stage at 0.940 and 11 ms, with the caveat that
   its model card does not list Slovenian.
2. **Chunk size buys hit@k for every method, and the per-variant design flatters
   large chunks.** `tok512` is a safe improvement over `base`; `tok1024` scores
   highest on a smaller and more salient question set while reading 4.3× more of
   the store per query than `tok256`. The expanded corpus (§7.4) is what will
   settle it.
3. **The retry loop is worth about a point** in every one of its eighteen paired
   measurements, at 8–9× the latency, with its real gain on `crosslingual`. Its
   judge catches about 70% of misses and repairs about 13% of the ones it
   retries.
4. **Page tools have never helped**: flat under a judge that barely uses them,
   −3.2 points under one that does (recovered 2, lost 19 on the 96 tool
   questions). The fault is in promoting found chunks into the ranking.
5. **Navigation instead of an index loses 18–37 points.** OKF 0.501–0.605
   against 0.873 at 15× the latency; DCI 0.697 against 0.881 and 0.352 on
   `crosslingual`, a structural failure of grep against a vocabulary gap.
6. **Geography must re-rank, never pre-filter, and unknown footprint must be
   neutral.** The hard filter lost 15 of 72 scoped questions and won one; soft
   geography matches the text baseline question for question; nothing on the
   176-page corpus gives a distance score anything to reorder.
7. **Judge choice moves quality by about a point and decides cost and
   reliability.** DeepSeek on Azure: no GPU, no failures in about 1,400 calls,
   1.5 s per call. gpt-oss: 1 s per call, 13 GB, no failures on the sufficiency
   schema but 7.1% on the tool and corpus schemas. gemma: 28 GB, 9 s per call,
   usable in one exact configuration, two runs lost to a shared-host logout.
8. **The measurement harness matters as much as the methods.** The 512-token cap
   hid 70% of chunk text for three months; the reranker template bug produced
   the "reranking is harmful" finding; the diagnostics printed zero actions for
   agents that made hundreds until 2026-09-03; the noise floor is about 1 point;
   and the `data/db/pages.db` gold labels are positional, so re-chunking the
   durable store deletes them.

## Appendix A — results with 150 or fewer samples (listed, not tabulated)

| result | n | where |
|---|---:|---|
| Handmade reranker sample: best non-rerank hit@5 0.680–0.880 against best rerank 0.220–0.260 per family | 50 questions | `retrieval-results-handmade.md` |
| OKF vs RAG evidence-acquisition pilot (2026-07-15): page hit `okf` 1.000, `qwen_hybrid` 0.579, `qwen_hybrid_agentic` 0.526 | 19 questions | deleted `docs/archived_okf-rag-results-2026-07-15.md` (commit `2a2a8dc`) |
| First full-method OKF checkpoint (13 methods, June 2026) | 20 questions | `.local/retrieval-results-okf-full.checkpoint.json` |
| DCI before / after citation fix: bare page ids 13 → 0, hit@10 0.520 → 0.480; 14 of 25 questions emitted zero citations at the 8-step ceiling | 25 questions | `agentic-retrieval-report-2026-09-01.md` §7 |
| Page-tool prompt experiment on recoverable misses: generic prompt 2/40 recovered (1 search_in_page, 37 reformulate); literal-answer gate 5/40 (42 search_in_page, 0 reformulate) | 40 questions | `docs/architecture/decisions.md` |
| Judge structured-output reliability on real prompts (2026-09-03): gemma thinking on, cap 1,024: 76%; cap 4,096: 98%; thinking off + function calling: 100% on ChunkSufficiency, ToolAction and CorpusAction at 9–14 s per call | 50 prompts per cell | `.local/reports/schema-reliability-real-2026-09-03-*.md` |
| gemma corrected configuration first-attempt success | 150 prompts (not over 150) | `agentic-judge-comparison-2026-09-07.md` note 1 |
| gemma synthetic probe (253-token prompt) 10/10 before failing a quarter of real prompts | 10 prompts | `retrieval-results-comprehensive-2026-09-07.md` §5 |
| Answer anchoring coverage when last measured: 39 of 64 answers placed | 64 answers | `chunk-size-sweep-2026-08-14.md` coverage note |
| Country-hint mis-geocoding examples (Gradec, Dunaj, Rim, Trst, Gorica) | 5 names, drawn from the 375-row finding in §7.1 | `locate-rejections-2026-09-09.md` |

## Appendix B — sources

| era | document | what was taken | status |
|---|---|---|---|
| A | `docs/retrieval-results.md` at commit `933d944` (2026-05-29) | §1.1 summary and MiniLM rows, §1.3 300-question samples | deleted; recovered from git |
| A | `docs/reports/retrieval/retrieval-results-handmade.md` (= `docs/retrieval-results.md` at `8acdc9e`) | §1.1, §1.2 | current |
| B | `docs/retrieval-results-agentic.md` at commit `e47194b` (2026-06-23) | §2 | deleted; recovered from git (the root-level `retrieval-results-agentic.md` is an empty 0/3471 snapshot of the same run) |
| C | `docs/reports/retrieval/retrieval-results-comprehensive-2026-07-27.md` | §3, 847 rows | current |
| C | `docs/reports/okf/comprehensive-okf-results.md` | §3, 264 rows and 396 rows | current |
| C | `docs/retrieval-results-chunks-okf.md` at commit `2c0a427` | §3, 260 rows | deleted; recovered from git |
| C | `docs/retrieval-results-comprehensive-2026-07-23.md` at commit `2c0a427` | §3, 199 rows | deleted; recovered from git |
| C | `docs/reports/okf/okf_benchmark.md`; `.local/retrieval-results-okf-v2.checkpoint.json` | §3, OKF 396 rows; checkpoint counts | current / local |
| D | `docs/reports/chunking/chunk-size-sweep-merged-2026-08-18.md` and `sweeps/*.md` | §4.2–4.7 | current |
| D | `docs/reports/chunking/chunk-token-audit-2026-08-11.md` | §4.1 audit | current |
| D | `experiments/indexing/README.md` | §4.8 span, store-share and density tables | current |
| E | `docs/reports/agentic/agentic-tools-2026-08-31.md`, `agentic-dci-2026-09-01.md`, `agentic-retrieval-report-2026-09-01.md`, `agentic-deepseek-2026-09-07.md`, `agentic-judge-comparison-2026-09-07.md`, `agentic-findings.md` | §5 | current |
| E | `docs/agentic-retrieval-qwen-format.md`, `docs/agentic-tools-2026-08-31-qwen-format.md` at commit `998dd60` | §5.1 hit@depth, §5.2 seconds | deleted; content also in the reports above |
| E | `docs/reports/retrieval/retrieval-results-comprehensive-2026-09-07.md` | §0 noise floor, §5.2 reliability counts | current (previous overview; superseded by this document for coverage, not for its prose) |
| F | `docs/reports/retrieval/geo-run-2026-09-08.md`, `geo-run-2026-09-09.md`, `geo-run-2026-09-09-strict.md`, `docs/reports/geo/geo-soft-vs-strict-2026-09-09.md` | §6 | current |
| G | `docs/reports/geo/locate-rejections-2026-09-09.md`; `.local/reports/locate-rejections-2026-09-09-pass{1,2,3}.md` | §7.2 | current / local |
| G | `docs/architecture/geo-rationale.md`, `geo-retrieval.md`, `geo-improvement-plan.md` | §7.1 coverage counts | current |
| G | `docs/architecture/decisions.md` | §7.3 miss taxonomy, §7.4 corpus expansion | current (uncommitted edits on `feature/geo-location`) |
| G | `docs/reports/scraping/seed-quality-2026-09-13.md`, `seed-quality-2026-09-13-before-prune.md` | §7.5 | untracked |
