# Agentic Retrieval — Evaluation Report

**Date:** 2026-09-01
**Corpus:** 726 `base` chunks over 176 Slovenian-language tourism / history / biography pages
**Questions:** 495 of 3,476 approved, shared design — one question set, labels projected per variant
**Store:** `.local/chroma-sweep` · **Database:** `.local/db/pages-shared.db` · **Judge:** local `gpt-oss:20b`
**Checkpoints:** `docs/agentic-tools-2026-08-31.md.checkpoint.json`, `docs/agentic-dci-2026-09-01.md.checkpoint.json`
**Compute:** 1h56m + 3h04m on one RTX A6000, shared with other users

---

## 1. Overview — every approach, one table

495 identical questions, `base` variant, `qwen` embedder, local `gpt-oss:20b` judge.

|                          | **Reranker only** | **Retry loop (Gen 1)**  | **Page tools (Gen 2)**              | **Corpus agent (Gen 3)** |
|--------------------------|-------------------|-------------------------|-------------------------------------|--------------------------|
| Tools the agent gets     | none              | `reformulate`, `expand` | + `list_sections`, `search_in_page` | `search`, `read`, `toc`  |
|                          |                   |                         |                                     |                          |
| hit@1                    | 0.723             | 0.725                   | 0.719                               | 0.576                    |
| **hit@5**                | 0.881             | 0.889                   | 0.879                               | 0.697                    |
| hit@10                   | 0.907             | 0.915                   | 0.903                               | 0.723                    |
| mrr@10                   | 0.791             | 0.795                   | 0.787                               | 0.632                    |
| **vs baseline (hit@5)**  | —                 | +0.8pp                  | -0.2pp                              | -18.4pp                  |
|                          |                   |                         |                                     |                          |
| **crosslingual**         | 0.857             | 0.901                   | 0.857                               | 0.352                    |
| direct_long              | 0.940             | 0.952                   | 0.940                               | 0.892                    |
| direct_short             | 0.905             | 0.895                   | 0.895                               | 0.800                    |
| vague_long               | 0.963             | 0.953                   | 0.963                               | 0.813                    |
| vague_short              | 0.752             | 0.761                   | 0.752                               | 0.624                    |
| **crosslingual vs base** | —                 | +4.4pp                  | +0.0pp                              | -50.5pp                  |
|                          |                   |                         |                                     |                          |
| ms / query               | 675               | 6,062                   | 6,572                               | 22,084                   |
| Slowdown (same run)      | 1.0×              | 9.0×                    | 9.7×                                | 15.5×                    |
| Total wall clock         | 6 min             | 56 min                  | 60 min                              | 184 min                  |
| LLM steps / question     | 1.00              | 1.21                    | 1.25                                | 5.85                     |
|                          |                   |                         |                                     |                          |
| Rankings it changed      | —                 | 6.9% (34/495)           | 4.8% (24/495)                       | n/a (not paired)         |
| Judge/schema failures    | —                 | 0                       | 44 (7.1%)                           | 35 (7.1%)                |
| **Verdict**              | **Baseline**      | Inside noise floor      | **No benefit**                      | **Fails badly**          |

**Notes on this table.** `Slowdown (same run)` compares each method to the reranker cell measured
beside it, which is the fair comparison; DCI's own run had a reranker baseline 2.1× slower than the
first run's because three other users were on the GPU, so DCI's 22,084 ms/query is **32.7×** the
uncontended 675 ms rather than the 15.5× shown. `Rankings it changed` is unavailable for DCI
because the paired diagnostics only pair a method with a `_rerank` baseline it can name, and `dci`
has no such name. Judge failures mean different things per column: for the page tools the agent
silently degrades to `expand`; for DCI it abandons the question and returns the BM25 shortlist.

**In one paragraph:** the cross-encoder reranker does essentially all the useful work. An LLM retry
loop on top moves the score by less than the harness's own run-to-run variation (§2). Adding
page-navigation tools to that loop makes it very slightly worse and leaves 95% of rankings
untouched. Replacing the vector index with a grep-driven corpus agent costs 18 points of hit@5 and
**halves cross-lingual retrieval** — the number that matters most for a multilingual European
deployment. The only place any agent clearly beats the reranker is the retry loop on
`crosslingual`, at +4.4pp.

## 2. Read this first: the measurement noise floor

`qwen_hybrid_rerank` was run **twice** on the **same 495 questions** — once as a method, once as a
control in the DCI run. It contains no LLM and no agency. It should be deterministic. It was not:

| Metric | Run 1 | Run 2 | Difference |
|---|---|---|---|
| hit@1 | 0.7232 | 0.7313 | **+0.8pp (4 questions)** |
| hit@5 | 0.8808 | 0.8808 | +0.0pp |
| hit@10 | 0.9071 | 0.9111 | +0.4pp (2 questions) |
| mrr@10 | 0.7908 | 0.7959 | +0.5pp |
| `direct_short` | 0.9048 | 0.8952 | **−1.0pp** |
| `vague_long` | 0.9626 | 0.9720 | **+0.9pp** |

Almost certainly Chroma's HNSW index: approximate nearest-neighbour search is not guaranteed
run-to-run identical, so the candidate set shifts slightly, the reranker sees different input, and
near-ties flip. *(Inferred from the pattern, not instrumented.)*

**Consequence: any difference below roughly ±1pp on this harness is not a result.** That directly
governs how §4 should be read — the retry loop's +0.8pp is the same size as the noise on a method
that cannot vary by design. It does not invalidate the retry-loop finding, but it does mean the
finding rests on *direction replicating across runs*, not on any single measurement.

It does not affect DCI at all: −18.4pp is 20× the noise floor.

---

## 3. What had ever been measured before this week

| Approach | Benchmarked before? | First real measurement |
|---|---|---|
| Reranker only | Yes, many runs since June | — |
| Retry loop (Gen 1) | Yes — 2026-07-27, 847 questions, 2 providers | — |
| Page tools (Gen 2) | **Never** | 2026-08-31 |
| Corpus agent (Gen 3) | **Never** | 2026-09-01 |

Two of the four approaches were fully implemented, configured, unit-tested and unmeasured. The 14
GPU-hours of chunk sweeps from 2026-08-14 to 08-19 contained no agentic arm at all, even though
`compare_chunkings.py` ships an `"agentic"` preset that includes `nemotron_hybrid_agentic_tools`.
It had never been passed to `--methods`.

---

## 4. The four approaches

**Reranker only** — `qwen_hybrid_rerank`. Dense vectors + BM25 fused 70/30, then a Qwen3
cross-encoder rescores the top 30 and returns 10. No LLM, no agency.

**Retry loop (Gen 1)** — `qwen_hybrid_agentic`. An LLM reads the retrieved chunks and answers *is
this enough to answer the question?* If not it rewrites the query or asks for more chunks, up to 3
attempts. Both tools act on the query; neither looks inside a document.

**Page tools (Gen 2)** — `qwen_hybrid_agentic_tools`. The same loop plus `list_sections(page_id)`
and `search_in_page(page_id, term)`. Found chunks are inserted right after the ranked chunk from
the same page, so the agent's choices are scored. **Both tools need a `page_id` already in the
ranking** — the agent can drill down but not sideways; the only escape is `reformulate`, which
discards everything learned.

**Corpus agent (Gen 3, DCI)** — `dci`. No vector index. BM25 shortlists pages, they are written to
disk as line-numbered Markdown, and the agent explores with `search` (real ripgrep), `read` (line
ranges) and `toc`, then names chunk ids. After RISE / *Beyond Semantic Similarity* (arXiv
2605.05242).

---

## 5. Results

### Ranking quality — 495 identical questions

| Method | hit@1 | hit@5 | hit@10 | recall@10 | mrr@10 |
|---|---|---|---|---|---|
| `qwen_hybrid_rerank` | 0.723 | **0.881** | 0.907 | 0.907 | 0.791 |
| `qwen_hybrid_agentic` | **0.725** | **0.889** | **0.915** | **0.915** | **0.795** |
| `qwen_hybrid_agentic_tools` | 0.719 | 0.879 | 0.903 | 0.903 | 0.787 |
| `dci` | 0.576 | 0.697 | 0.723 | 0.723 | 0.632 |

### Deltas against the baseline measured in the same run

| Method | hit@1 | hit@5 | hit@10 | mrr@10 | Beyond noise floor? |
|---|---|---|---|---|---|
| `qwen_hybrid_agentic` | +0.2pp | +0.8pp | +0.8pp | +0.4pp | **No** |
| `qwen_hybrid_agentic_tools` | −0.4pp | −0.2pp | −0.4pp | −0.4pp | **No** |
| `dci` | **−15.6pp** | **−18.4pp** | **−18.8pp** | **−16.4pp** | **Yes, by 20×** |

### Beyond k=10

An agent that expands returns more than 10 chunks, so each reaches its own depth. Not comparable
between methods, but it shows what the extra chunks bought.

| Method | Depth reached | hit@depth | Gain over own hit@10 |
|---|---|---|---|
| `qwen_hybrid_rerank` | 10 | 0.907 | — |
| `qwen_hybrid_agentic` | 20 | 0.917 | +0.2pp |
| `qwen_hybrid_agentic_tools` | 24 | 0.907 | +0.4pp |

Both agents expanded well past the benchmark cutoff for almost nothing. The page tools needed **24
chunks to reach the score the reranker reaches in 10**.

### Cost

| Method | Total | ms/query | vs baseline | Steps or retrievals per question |
|---|---|---|---|---|
| `qwen_hybrid_rerank` | 339 s | 675 | 1.0× | 1.00 |
| `qwen_hybrid_agentic` | 3,367 s | 6,062 | 9.0× | 1.21 |
| `qwen_hybrid_agentic_tools` | 3,613 s | 6,572 | 9.7× | 1.25 |
| `dci` | **11,051 s (3h04m)** | **22,084** | **32.7×** | **5.85** |

The retry loop's net gain was +4 questions of 495 for 3,028 extra seconds — about **13 minutes of
GPU per additional question answered**, and that gain is inside the noise floor. DCI spent 3 hours
to finish 91 questions *behind* the baseline.

The sweep harness prices every agentic cell at a flat 5.1 s/query. Measured: 6.1, 6.6, and **22.1**.
Any cost projection including DCI at the flat rate was low by 4.3×.

---

## 6. Where the differences live

### hit@5 by question category

| Method | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---|---|---|---|---|
| `qwen_hybrid_rerank` | 0.857 | 0.940 | 0.905 | 0.963 | 0.752 |
| `qwen_hybrid_agentic` | **0.901** | **0.952** | 0.895 | 0.953 | **0.761** |
| `qwen_hybrid_agentic_tools` | 0.857 | 0.940 | 0.895 | 0.963 | 0.752 |
| `dci` | **0.352** | 0.892 | 0.800 | 0.813 | 0.624 |

Three things to take from this table.

**The retry loop's one genuine strength is `crosslingual`: +4.4pp**, five times its overall gain and
its largest category movement. This is the category that matters most for a multilingual European
product, and it is the only place any agent clearly beats the reranker. If one thread of the agentic
work survives, it is this one.

**The page tools are identical to the baseline on four of five categories** — to three decimals.
They differ only on `direct_short`, where they are worse. That is the signature of an agent
returning the first-stage ranking untouched.

**DCI collapses on `crosslingual`: 0.857 → 0.352, −50.5pp.** This is not a tuning problem, it is
the architecture. DCI finds evidence by matching text with ripgrep, and a cross-lingual question
cannot be matched literally against an answer in another language. Its per-category ranking is
exactly what a lexical-only method predicts: best on `direct_long` (−4.8pp), where questions carry
distinctive literal terms, and catastrophic wherever meaning has to bridge a vocabulary gap.

### Did the agent change anything?

From per-question `baseline_rank` vs `agent_rank`:

| Method | Ranking changed | Improved | Worsened | Untouched |
|---|---|---|---|---|
| `qwen_hybrid_agentic` | 34 / 495 (6.9%) | 18 | 16 | 461 |
| `qwen_hybrid_agentic_tools` | **24 / 495 (4.8%)** | 10 | **14** | 471 |

At the hit@5 cutoff:

| Method | Recovered@5 | Lost@5 | Net | Sign test |
|---|---|---|---|---|
| `qwen_hybrid_agentic` | 9 | 5 | +4 | p = 0.42 |
| `qwen_hybrid_agentic_tools` | 2 | 3 | −1 | p = 1.00 |

Five discordant pairs out of 495 for the page tools. Their effect is indistinguishable from zero in
both directions.

---

## 7. The corpus agent (DCI) in detail

DCI is the most interesting failure here, because two separate bugs had to be fixed before it could
even be measured, and the measurement then showed the approach itself is wrong for this corpus.

### Two bugs found and fixed before benchmarking

**Cited page ids were silently binned.** The workspace names each file after its page
(`03293d43-….md`); a chunk id is that string plus `:index`. The agent routinely returned the file
name — the document, not the paragraph. The hallucination guard in `_ranking` correctly rejects
unknown ids but cannot tell a fabrication from a truncation, so it discarded them **with no log
line**, and the ranking fell back to BM25. DCI scored BM25 under its own name. Over 25 questions:
13 of 27 citations were page ids and 15 of 25 questions resolved to nothing.

**The repeated-search guard never fired.** `_signature` compared `path` even for `search`, which
ignores it, so a stray path made an identical ripgrep look like a new step. One question spent 3 of
8 steps searching `"Slovenia"`.

Fixed via `_resolve_citations`, `_answer_rejection`, `_record_resolution` and a corrected
`_signature`, with 9 new tests; 312 pass.

### The fixes did not help, and the logging showed why

| | Before | After |
|---|---|---|
| Bare page ids (n=25) | 13 | **0** |
| hit@10 (n=25) | 0.520 | 0.480 |

The prompt clarification eliminated page-id citations — and the agent replaced them with *silence*
rather than correctness. Over the full 495-question run, the page-id resolver **fired zero times**
(`dci expanded: 0`), so that fix is dead code in practice. Only 6 genuine fabrications were
rejected across 495 questions.

The diagnostic that explains the score:

```
questions emitting ZERO citations: 14 of 25   (mean steps: 8.0 — the ceiling, exactly)
questions emitting ≥1 citation:   11 of 25   (mean steps: 4.4, hit@10 0.818)
```

**The agent frequently never commits to an answer.** It searches and reads until `dci_max_steps: 8`
is exhausted, the loop exits with no citations, and the shortlist is returned. Nothing in the prompt
tells it how many steps remain. Full-run mean steps: **5.85 of 8**.

### Schema failures

35 of 495 questions (**7.1%**) ended in `corpus agent failed` — the judge could not produce a valid
`CorpusAction` in 3 attempts, so the agent abandoned the question entirely:

| Failure | Count |
|---|---|
| `model returned no structured output` | 16 |
| `chunk_ids: Input should be a valid list` (got `None`) | 7 |
| `Unknown tool type: 'answer'` | 5 |
| `Unknown tool type: 'search'` | 5 |
| `Unknown tool type: 'toc'` / `'read'` | 2 |

The `Unknown tool type` cluster (12 cases) is diagnostic: **the model tries to call the four actions
as four separate tools** rather than emitting one `CorpusAction` with an `action` field. It is
fighting the single-schema design. Anthropic's tool-design guidance and the model's own behaviour
both point the same way — four namespaced single-purpose tools.

### Is DCI salvageable?

The three fixable problems — the step ceiling, the schema shape, the 7% abandonment — would plausibly
recover a good part of the 18-point gap. On the questions where the agent commits, it scores 0.818
hit@10 in 4.4 steps.

**But the `crosslingual` collapse is not fixable by tuning.** A grep-based agent cannot match a
Slovenian answer to an English question. Given that cross-lingual retrieval is the point of this
project, DCI is the wrong architecture here regardless of how well the remaining bugs are fixed. It
would need the semantic search tool the RISE paper gives its agent and this implementation
deliberately withholds.

---

## 8. Replication across runs

Absolute scores across runs are **not** comparable — different samples, embedders, judge models.
Each agent's delta against the baseline measured beside it in the same run is.

| Run | Questions | Agent | Baseline | Agent | Delta |
|---|---|---|---|---|---|
| 2026-07-27 | 847 | `embed_v4_hybrid_agentic` | 0.903 | 0.911 | +0.8pp |
| 2026-07-27 | 847 | `nemotron_hybrid_agentic` | 0.874 | 0.884 | +1.0pp |
| 2026-09-01 | 495 | `qwen_hybrid_agentic` | 0.881 | 0.889 | +0.8pp |
| 2026-09-01 | 495 | `qwen_hybrid_agentic_tools` | 0.881 | 0.879 | −0.2pp |

The retry loop lands positive three times — +0.8, +1.0, +0.8pp — across three embedders (Azure
`embed-v-4-0`, Nemotron, Qwen), two judge models (Azure DeepSeek-V4, local `gpt-oss:20b`) and two
question samples. Three independent same-sign results would happen by chance 1 time in 4, so this is
suggestive rather than conclusive, and §2 shows each individual measurement is noise-limited. The
honest statement: **the retry loop probably helps slightly; it has not been established that it
does.**

The page tools are the only row with the wrong sign, and the only one with no prior measurement.

### The July diagnostics, which this run could not reproduce

| Agent | Retried | Retry precision | Retry recall | Recovered | Lost |
|---|---|---|---|---|---|
| `embed_v4_hybrid_agentic` | 157 (18.5%) | 0.153 | 0.683 | 11 | 4 |
| `nemotron_hybrid_agentic` | 166 (19.6%) | 0.120 | 0.729 | 11 | 2 |

This remains the sharpest finding in the whole body of work:

> **The sufficiency judge is a good miss detector and a poor repair mechanism.**
> Retry recall 0.68–0.73 — it catches ~70% of the baseline's failures.
> Retry precision 0.12–0.15 — its chosen fix works ~13% of the time.

It also predicted this run's result: giving a poor repairer *better repair tools* (Gen 2) does not
help when the problem is choosing the repair.

---

## 9. Measurement limitations

| Limitation | Consequence |
|---|---|
| **Noise floor ≈ ±1pp** (§2) | The Gen 1 and Gen 2 results are both inside it. DCI's is not. |
| **Action log not persisted.** The sweep harness stores aggregates only. | `retried`, `rewritten`, `expanded`, `tool_calls`, `tools_used` are empty for all 495 questions. A `0` is missing data, not inaction — `queries/query` 1.25 proves it acted. **We still cannot say how often `search_in_page` was called.** |
| **44 judge failures (7.1%) in the Gen 2 cell** → silent fallback to `expand`. | On ≥8.9% of questions the tool agent was not acting as a tool agent. The plain sufficiency judge failed 0 times; `ToolAction` was validated at 9/10 on ten samples and fails ~1 in 14 at scale. |
| **35 agent failures (7.1%) in the DCI cell** → question abandoned. | DCI's score is depressed by roughly this much on top of the step-ceiling problem. |
| **n = 495**, ±2pp on hit@5. | A ~1pp effect cannot be resolved even before the noise floor. |
| **No `judge_hit@K`.** | Retrieved sets answering via different evidence score as misses. |
| **One provider, one variant** (`qwen`, `base`). | Provider- and chunking-independence not established. |
| **Shared GPU.** The control cell took 339 s in run 1 and 713 s in run 2. | Latency figures are indicative; quality figures are unaffected. |

---

## 10. Recommendations

**1. Do not pursue DCI for this project.** −18.4pp overall and −50.5pp cross-lingual. The bugs are
fixable; the architecture's blindness to cross-language meaning is not, and that is precisely what
LLMs4EU needs. If anyone revisits it, the first change is giving the agent a semantic-search tool
alongside grep, which is what the source paper actually does.

**2. Do not ship the page tools.** −0.2pp, 9.7× latency, inert on 95% of questions, dependent on a
schema that fails 1 call in 14. Close them out or delete them.

**3. Keep the retry loop only where it earns its cost — cross-lingual queries.** +4.4pp on
`crosslingual` against +0.8pp overall. Route to it on language mismatch rather than running it on
every query at 9× cost. This is the one positive, reproducible-in-direction signal in the whole
exercise.

**4. Move the sufficiency judge to generation time.** Its measured strength is detecting
insufficiency (recall 0.70), not repairing it (precision 0.13). Value lands in selective generation
— abstain or ask back — not retry (*Sufficient Context*, ICLR 2025). Your own answerability gating
moved `vague_short` 0.76 → 0.83, more than the entire retry loop delivered.

**5. Fix the harness before running more agentic experiments.** Three things, in order:
persist the action log (the July run has better diagnostics than this one — that is a regression);
make retrieval reproducible or measure the noise floor explicitly, because right now a 1pp result is
unreadable; parallelise `retrieve_batch`, which is a sequential dict comprehension over an HTTP
judge and is why these two runs took 5 hours.

**6. If the schema work continues, split `CorpusAction` and `ToolAction` into separate
single-purpose tools.** Both failure clusters — `Unknown tool type: 'answer'` and `Invalid json
output` — say the model wants namespaced tools, not one schema with an action discriminator.

---

## Appendix — reproducing this

```bash
# Gen 1 + Gen 2 + baseline (1h56m measured)
CHROMA_PATH=.local/chroma-sweep PAGES_DB_PATH=.local/db/pages-shared.db \
uv run python experiments/indexing/compare_chunkings.py \
  --variants base --design shared --limit 500 --agentic-diagnostics --keep-checkpoint \
  --methods qwen_hybrid_rerank,qwen_hybrid_agentic,qwen_hybrid_agentic_tools \
  --output docs/agentic-tools-2026-08-31.md

# DCI + baseline control (3h04m measured). Separate checkpoint because adding a
# method to a finished run changes the signature and discards every completed cell.
CHROMA_PATH=.local/chroma-sweep PAGES_DB_PATH=.local/db/pages-shared.db \
uv run python experiments/indexing/compare_chunkings.py \
  --variants base --design shared --limit 500 --keep-checkpoint \
  --methods qwen_hybrid_rerank,dci \
  --output docs/agentic-dci-2026-09-01.md

# Both, merged, in the compare_qwen report layout
uv run python experiments/indexing/render_qwen_report.py \
  docs/agentic-tools-2026-08-31.md.checkpoint.json \
  docs/agentic-dci-2026-09-01.md.checkpoint.json \
  --compare docs/retrieval-results-comprehensive-2026-07-27.md \
  --output docs/agentic-retrieval-qwen-format.md
```

Two environment traps that cost real time:

- `.env` sets `CHROMA_PATH=.local/chroma`, which has **no `qwen8b` collection** and a dead
  `english` collection at 0 rows; every dense method reports a missing index against it.
  `.local/chroma-sweep` is the store with all five providers.
- `data/db/pages.db` has **no `variant` column**, so the uncommitted variant threading makes `dci`
  raise `no such column: variant` against the durable store. DCI runs only against the sweep and
  shared copies.
