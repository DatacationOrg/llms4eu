# Indexing Experiments

Small runnable comparisons for page-chunk retrieval. These scripts use the
approved eval questions in `data/db/pages.db`, reusable retrievers from
`src/retrieval`, and metrics from `src/eval`.

```bash
uv run python experiments/indexing/evaluate_embedding_models.py
uv run python experiments/indexing/evaluate_embedding_models.py --category crosslingual
uv run python experiments/indexing/evaluate_embedding_models.py --providers qwen --limit 20

uv run python experiments/indexing/compare_qwen_modes.py
uv run python experiments/indexing/compare_qwen_modes.py --category crosslingual
uv run python experiments/indexing/compare_qwen_modes.py --limit 100
```

When a selected agentic method and its matching reranked baseline are both in
the run, the report adds paired agentic diagnostics at the configured category
hit cutoff. For example, `nemotron_hybrid_agentic` is paired with
`nemotron_hybrid_rerank`. The summary reports retry/rewrite/expansion counts,
recovered and lost hits, retry precision, retry recall, and mean retried versus
non-retried latency. Here retry precision is the fraction of retries that
improve the first relevant rank; retry recall is the fraction of baseline misses
that the agent retries.

A sibling `*-agentic-diagnostics.json` retains per-question category,
baseline/agent hit and rank, rank delta, reranker top score, top-1/top-2 margin,
score spread, judge verdict/reason, query count, total latency, and retrieval,
judge, and top-up stage latency. Existing checkpoints without observations can
still derive quality/action metrics from their action logs, but stage and
per-query latency fields are available only for newly measured questions.

`evaluate_embedding_models.py` compares the embedding providers listed in
`src/indexing/config.yaml`. Missing vector collections are built through the
normal indexing path. Every provider is local.

`config.yaml` sets the default warmup count. Warmup queries are excluded from
timing.

## The comprehensive comparison (`compare_qwen_modes.py`)

`compare_qwen_modes.py` is the full-comparison driver (2026-09-08 rewrite; OKF
is gone, it is measured out). A cell is one (chunk variant, method) pair and
keeps per-question rankings, action logs and observations in the checkpoint, so
the report carries the elaborate diagnostics of the earlier comprehensive runs
(paired agentic diagnostics, `judge_hit@K`, per-question sidecars) *and* the
chunk sweep's span and cost metrics (`budget_recall@4000`, `char_recall@10`,
`store_share@10`, `recall_per_share@10`), per variant and across variants.

Axes, all in one resumable run:

| axis | how | group |
|---|---|---|
| embedder | `qwen`, `qwen4b`, `qwen8b`, `nemotron`, `nemotron8b` | `embedders`, `pipelines` |
| pipeline rung | bare, `_hybrid`, `_hybrid_rerank`, `_hybrid_rerank_4b` | `pipelines`, `reranker-ladder` |
| geo scope | `_hybrid_geo`, `_hybrid_rerank_geo` (soft: over-fetch and re-score), `_hybrid_rerank_geo_strict` (filter, widen) | `geo` |
| agent / no agent, tools | `_hybrid_agentic`, `_hybrid_agentic_tools`, `dci`, `dci_k50`, beside `_hybrid_rerank` | `agents` |
| agents on geo | `_hybrid_agentic_geo`, `_hybrid_agentic_tools_geo` (with `find_pages_near` / `pages_in_region`) | `agents-geo` |
| judge LLM | unsuffixed = azure DeepSeek; `_gptoss`, `_gemma` from `agentic_judges` in `src/retrieval/config.yaml` | `judges` |
| reasoning effort | `_high` (gpt-oss only; gemma's `think` is boolean) | `reasoning` |
| chunk size | `--variants base,tok256,tok512,tok512ov,tok1024` | |

`--methods` mixes groups and bare names (`--methods agents,geo,qwen4b`);
`full` is the union of every group. Methods run grouped by embedding provider,
one provider resident at a time, and interleave per question inside a group.

```bash
# 1. Check the database and price the grid. Nothing is loaded.
just full-comparison-dry-run                       # sweep store, shared-design DB
just full-comparison-dry-run agents,geo base       # a slice

# 2. Prerequisites the dry run will name:
export PAGES_DB_PATH=.local/db/pages-shared.db CHROMA_PATH=.local/chroma-sweep
just relabel tok256 && just relabel tok512 && just relabel tok512ov && just relabel tok1024
just full-comparison-prepare                      # page_locations into the sweep DB
just eval-index qwen v1 base                      # geo metadata: rebuild each (provider, variant); the dry run lists which

# 3. Launch from cron (survives logout), resumable by rerunning the same line.
just full-comparison
just full-comparison agents,judges base            # a slice, same checkpoint rules
```

The run writes `docs/reports/retrieval/retrieval-results-full.md` (rewritten every 30 s while
running), `<output>.checkpoint.json` (kept by default; `--discard-checkpoint`
removes it), `<output>.cells.csv` (one row per variant, method, metric, for
merging with other reports), and the `-judge-actions.json` and
`-agentic-diagnostics.json` sidecars. `--judge-equivalence` adds `judge_hit@K`
through the local equivalence judge (`--judge-model` overrides
`config.yaml:equivalence_judge_model`), cached per variant.

Resume rules: a checkpoint is resumed when the question sets, warmup, design,
span target and the retrieval fingerprint (reranker, fusion weights, judge
provider and model, geo settings) match; it is *extended* when only variants or
methods were added; anything else starts fresh and says which key differed.
`--catch-up-only` runs newly added cells up to each variant's existing frontier.

Gate any agentic cell on the judge reliability probe (see above) before
quoting it, and read the `failures` column of the Coverage table: a cell whose
judge failed is scoring fallback under the method's name.

The legacy default (`--methods legacy-default`) is the 2026-08 primary
benchmark: sparse rerank, Qwen4B hybrid rerank, Nemotron vector and hybrid
rerank, the Qwen and Nemotron hybrid agentic methods, and the tool-using
`qwen_hybrid_agentic_tools` paired against `qwen_hybrid_agentic`. When any agent
calls a page tool the report adds a `Page tool usage` table, and per-question
action sequences and search terms land in the `-agentic-diagnostics.json`
sidecar:

```bash
uv run python experiments/indexing/compare_qwen_modes.py
```

### Judge reliability probe and detached launches

`agentic_judge_provider` in `src/retrieval/config.yaml` selects the judge backend
for the whole agentic family: `azure` (the Azure AI Foundry DeepSeek deployment
from `.env`, no GPU) or `ollama` (the local settings below it). The probe and the
notes in this section are about the `ollama` backend.

Every agentic method depends on a local model returning one Pydantic schema, and
the working structured-output method is a property of the (model, schema) pair:
`gemma4:31b` needs `json_schema` for `ChunkSufficiency` and `function_calling`
for `CorpusAction`, and `gpt-oss:20b` is the other way round on the first. A
wrong pairing does not raise; the retriever falls back and the cell silently
scores BM25 or `expand` under the agent's name. Measure before changing either:

```bash
export CHROMA_PATH=.local/chroma-sweep PAGES_DB_PATH=.local/db/pages-shared.db
# Smoke test: one short synthetic prompt, repeated.
uv run python experiments/indexing/probe_structured_output.py --models gemma4:31b --samples 5
# The number that matters: the first N benchmark questions, real first-stage
# retrieval, the exact prompt each agent would send on its first attempt.
uv run python experiments/indexing/probe_structured_output.py --models gemma4:31b \
    --schemas ChunkSufficiency --methods json_schema --from-eval 50 \
    --output .local/reports/schema-reliability-real.md
```

The report gives first-attempt success, the exception histogram, prompt length in
tokens (bucketed, with success per bucket), the Ollama stop reason and the size
of the thinking block. `done_reason=length` on a failure means the output cap
(`agentic_judge_num_predict`) was spent on thinking; `--num-predict` overrides it
for a re-measurement. Synthetic prompts are ~250 tokens, real ones 5-10k, which
is how a 10/10 probe preceded a run with 136 judge failures.

Gate a rerun on the real-prompt number: below ~95% on `ChunkSufficiency`, every
`*_agentic*` cell is measuring fallback. The judge also carries a circuit breaker
(`agentic_judge_max_failure_rate`) that raises `JudgeCircuitOpen` and stops the
run instead of scoring fallback for hours.

Launch anything longer than a few minutes through the detached launcher, not a
shell or agent background task; both end with their session:

```bash
experiments/indexing/launch_detached.sh gemma-suite \
    uv run python experiments/indexing/compare_chunkings.py --design shared \
    --variants base --limit 500 --agentic-diagnostics --keep-checkpoint \
    --methods qwen_hybrid_rerank,qwen_hybrid_agentic,qwen_hybrid_agentic_tools \
    --output docs/reports/agentic/agentic-gemma-$(date +%F).md
grep -cE 'judge failed|corpus agent failed|JudgeCircuitOpen' .local/logs/gemma-suite.log
```

It exports the shared-design stores, writes to `.local/logs/<name>.log`, and
prints the pid; confirm `PPID 1` and `TT ?` before walking away. Check the
failure count before quoting any number from the log.

### Evidence-equivalence judge

Strict chunk metrics remain the primary retrieval metrics. A separate
post-hoc judge can audit strict misses to determine whether the retrieved
evidence nevertheless contains the same answer-bearing facts as the golden
chunks. It reports both collective evidence equivalence and cases containing an
individually near-duplicate retrieved chunk. This exposes alternate-source,
translation, and chunk-boundary duplication without silently changing qrels.

Keep the completed retrieval checkpoint, then run the audit:

```bash
uv run python experiments/indexing/compare_qwen_modes.py   # keeps its checkpoint
uv run python experiments/indexing/judge_retrieval_equivalence.py \
  --checkpoint docs/reports/retrieval/retrieval-results-full.md.checkpoint.json \
  --variant base \
  --output docs/reports/retrieval/retrieval-equivalence-judge.md -k 5
```

Alternatively, run the audit as part of the comparison. This adds a
`judge_hit@K` column to the overall table and writes a resumable judgment cache
beside the comparison report:

```bash
uv run python experiments/indexing/compare_qwen_modes.py \
	--judge-equivalence --judge-k 10
```

Integrated judging runs immediately after each method's retrieved ranking.
Exact strict hits do not require an LLM call; strict misses are sent to the
equivalence judge and persisted before retrieval continues. On resume, any
older checkpointed predictions missing judgments are backfilled first.

The speed table also reports `chunk expansions`: the number of times an agentic
retriever increased its requested chunk count between attempts. Query
reformulations at the same chunk limit are not counted as expansions.

The judge runs on the local Ollama model from `--model` (default
`gpt-oss:20b`). Results are
checkpointed after every judgment next to the output report, so interrupted
runs resume without repeating model calls. `--limit` provides an inexpensive
calibration sample per method. Manually review a stratified sample before using
judge-adjusted rates for conclusions.

The Qwen and Nemotron methods retrieve the same canonical SQLite chunks from
independent embedding collections. `sparse_rerank` is the embedding-independent
lexical baseline. Use `--methods agents` for the agentic suite beside its baselines.

Agentic methods normally report the standard top-10 metrics. If an agentic
retriever expands beyond rank 10, the report also adds hit, recall, and MRR at
the deepest observed rank. Those expanded columns contain values only for
agentic methods; non-agentic baselines are shown as `-` because they were not
retrieved beyond the standard cutoff.

OKF is no longer part of the comparison (measured out; see
`docs/reports/retrieval/retrieval-results-comprehensive-2026-09-07.md` §3 and
`experiments/indexing/compare_okf_rag.py` for the standalone benchmark).

Concept-level OKF scoring (`--methods okf-only`) was removed from
`compare_qwen_modes.py` with the OKF cells; `compare_okf_rag.py` is the only
remaining OKF benchmark.

## OKF versus chunk-RAG benchmark protocol

A fair comparison freezes one `data/db/pages.db` snapshot and records its hash,
selected page IDs, source bytes, languages, and page count. Scraping and
Markdown extraction are shared preparation and are reported once. Quality,
cost, build efficiency, and online efficiency remain separate result families;
there is no meaningful single “OKF versus RAG” score.

The protocol follows BEIR-style frozen queries and qrels, ANN benchmark
quality-efficiency frontiers, Ragas-style separation of context and answer
quality, and production guidance to retain percentile latency, throughput,
throttling, and client-versus-service timing.

### Clean and incremental builds

For a clean track, delete only derived outputs and independently build:

1. RAG chunks, embedding inputs, embeddings, and the persisted vector index.
2. OKF discovery inventory, canonical catalog, complete concepts, indexes, and
	validated manifest.

Report end-to-end and per-stage wall time, pages and MiB per second, time to the
first queryable artifact, successful coverage, failures, retries, API calls,
tokens, estimated cost, peak RSS, CPU time, artifact size, and storage
amplification. Include p50, p95, and maximum item latency.

For incremental evaluation, apply deterministic 1%, 10%, and 25% update sets
containing additions, modifications, and removals where supported. Measure
changed-page throughput, unchanged artifacts rewritten, API usage, time until
queryability, stale artifacts, interrupted-run recovery, and no-change resume
idempotence. Label unavoidable full rebuilds explicitly.

### Online retrieval and navigation

Use a held-out query set, excluded warmups, randomized order, and at least three
measured repetitions. Run concurrency 1 for intrinsic latency and fixed load
levels such as 1, 4, and 8 for throughput and saturation. Report client p50,
p95, and p99 latency; questions per second; failures, timeouts, and throttles;
context bytes/tokens; model calls; retrieval/navigation steps; time to first
evidence; and end-to-end answer latency. Keep service timing separate from
client round-trip timing when available.

### Representation-neutral effectiveness

Keep existing chunk qrels for native RAG evaluation. Only OKF metrics project
relevance onto the OKF concept ontology:

1. A gold chunk maps to its source page.
2. The page maps to every concept listing it in `source_page_ids`; these are the
	question's golden concepts.
3. OKF results are already ranked concepts: cited concepts first, then other
	visited concepts.

Reports use the shared columns `hit@K`, `recall@K`, and `mrr@K`. For OKF rows,
these values use concept relevance; for RAG rows, they use chunk relevance. A
sibling chunk therefore does not receive strict RAG credit merely because its
page maps to the same concept. No fuzzy answer-token or substring matching
expands qrels at evaluation time: phrase occurrence does not prove that a page
expresses the same entity or supports the answer. The optional
evidence-equivalence audit is reported beside, never merged into, these strict
metrics. Exact duplicates in different concepts instead indicate a
canonicalization issue to fix in the OKF bundle. Concept retrieval also remains
separate from answer correctness and citation support.

### Answer quality

Use the same answer model, settings, context budget, and blinded randomized
system order. Score factual correctness, groundedness, relevance, context
recall and precision, citation validity and entailment, and abstention quality.
Deterministically check citation existence and retrieval/visit membership.
Calibrate semantic judges against a manually reviewed stratified sample and use
human pairwise preference for material conclusions.

### Controls and statistics

Every published run records the Git commit and dirty state, database hash,
hardware and operating system, Python lock, model deployment identifiers,
configuration and command line, UTC timestamps, concurrency, retries, timeout,
cold/warm state, GPU state, seeds, and query order. Avoid unrelated machine
load.

Use at least five repetitions for inexpensive local timing and three for costly
full index builds when budget permits; label fewer runs exploratory. Report
median and p95 latency, mean throughput and run-to-run variation, and paired
bootstrap 95% confidence intervals over question-level quality differences.
Publish win/tie/loss judgments and every tested configuration, highlighting the
non-dominated quality-cost-latency frontier.

Recommended primary scorecard:

| Family | Primary metric | Guardrail |
|---|---|---|
| Clean build | pages/hour | successful page coverage |
| Build economics | cost per 1,000 pages | token and retry counts |
| Storage | output/source byte ratio | validated artifact count |
| Incremental update | changed pages/hour | stale artifact count |
| Online efficiency | p95 latency | error rate and context tokens |
| Retrieval | page-level `nDCG@10` | page-level `Recall@10` |
| Answers | factual correctness | groundedness and citation entailment |
| Overall | quality-cost-latency frontier | no composite score |

Large or credential-adjacent traces belong under `.local/`. Commit only
reviewed aggregate reports. A complete runner should retain raw per-page and
per-query observations, resource samples, token/cost accounting, paired answer
outputs, and statistical summaries so results can be recomputed.

The original 19-question pilot was recovered from ignored local report artifacts
and preserved in
[docs/reports/okf/comprehensive-okf-results.md](../../docs/reports/okf/comprehensive-okf-results.md).
Treat it as historical evidence only: it is not directly comparable with the
later broad chunk evaluation.

## Chunk size sweep

`compare_chunkings.py` varies the chunking instead of the retriever. Every
published number in this project was measured on one cutting — 1,800 target
characters, no overlap, character sized — so chunk size has never been a
measured choice.

Every provider now reads its model's full context, so no chunk size can be
truncated and chunk size is the only variable that moves. Each variant is paired
automatically with the shortest configured sequence length that reads its chunks
whole, and any substitution is reported under `## Sequence Length Fit` — pairing
a 1,024-token cut with a 512-token limit would measure truncation and report it
as a chunk-size effect.

To measure the historical 512 cap deliberately, disable that pairing:

```bash
just chunk-sweep --variants base --methods qwen,qwen_s512 --no-fit
```

Same chunks, same labels, 22% of the corpus unread versus none.

Labels are per variant on shared question texts, so one method's row is
comparable straight across variants. Prepare a variant end to end first:

```bash
just chunk-sweep-db
export PAGES_DB_PATH=.local/db/pages-variants.db
just chunk-variant tok512 512 0 tokens qwen
just relabel tok512
just eval-index qwen v1 tok512

just chunk-sweep --variants base,tok512 --methods qwen,qwen_hybrid_rerank
```

The report puts each cell's retrieval quality beside that variant's truncation
at each provider's limit, because a variant whose inputs exceed the limit is
being silently cut and its scores describe a corpus the embedder never fully
read. Two kinds of cell are left out, and both are listed rather than dropped
quietly: one whose vector index is not built, and one that would duplicate the
same model at a shorter sequence length that already truncates nothing.

The recommendation refuses to rank variants scored on different numbers of
questions — that means relabelling is incomplete, not that one variant won.

### Adding an embedder or a reranker to an existing sweep

`prepare_variants.sh` builds a sweep from nothing: snapshot, cut each variant,
generate its questions, index. Do not use it to add a provider — it re-cuts
chunks, and `page_chunks` cascades into `eval_relevant_chunks`, so re-cutting a
variant that already has its own generated questions deletes their gold links.
Only the last of its four stages is needed:

```bash
just index-variants "qwen4b qwen8b nemotron8b"
just sweep-overview --dry-run --design per-variant \
	--variants base,tok256,tok512,tok512ov,tok1024 --methods large-embedders
```

One provider per process, because the indexers memoize their loaded model and two
of these checkpoints are 15 GB in bf16.

Named method groups for the scale question:

| group | rows | what it isolates |
|---|---|---|
| `embedder-ladder` | `qwen`, `qwen4b`, `qwen8b`, `nemotron`, `nemotron8b` | bare dense only, so the embedder is the sole variable |
| `large-embedders` | the three new embedders, bare and hybrid+reranked | the same models at both rungs |
| `reranker-ladder` | `qwen8b_hybrid_rerank` against `qwen8b_hybrid_rerank_4b` | the cross-encoder, first stage held fixed |
| `symmetric` | every embedder on one hybrid+rerank pipeline | chunking, with the retrieval stack held fixed |

**Which cross-encoder reranks is now a method choice.** `src/retrieval/config.yaml`
has a `rerankers:` map; each name in it generates `{provider}_rerank_{name}` and
`{provider}_hybrid_rerank_{name}` methods that inherit every `reranker_*` default
and override only what they name. The unnamed default keeps its own method names
and scores, so adding a rung cannot move a published number. `reranker_dtype` is
there because `CrossEncoder` otherwise loads float32 whatever the checkpoint says,
which would put 16 GB of 4B reranker beside a 16 GB embedder on one 48 GB card.

Price a reranker rung before running it. `--dry-run` charges reranking per
candidate token, at a rate that is per reranker — the 0.6B's 0.00132 s/token would
understate a 4B row several-fold.

### Why the headline metric is character overlap

`hit@k` and `recall@k` count whole chunks, and a chunk twice as long is roughly
twice as likely to contain any given answer. Ranking chunkings on them therefore
reproduces chunk size rather than retrieval quality, so the sweep leads with the
span metrics from `src/eval/metrics.py` and keeps the chunk metrics below them for
continuity with earlier reports.

The target region needs no model. Every approved question was generated *from* a
`base` chunk, so that chunk's span is ground truth by construction and covers all
3,476 questions for free. `load_span_labels(variant)` defaults to it.

Measured with retrieval held perfect — every variant handed the chunk that best
covers the target, so the only thing varying is chunk size:

| Variant  | Chunks | char_recall@1 | char_precision@1 | budget_recall@4000 |
|---|---:|---:|---:|---:|
| base     |   726 | 1.000 | 1.000 | 1.000 |
| tok1024  |   618 | 0.857 | 0.637 | 0.865 |
| tok512ov | 1,271 | 0.694 | 0.907 | 0.935 |
| tok512   | 1,220 | 0.694 | 0.913 | 0.958 |
| tok256   | 2,609 | 0.424 | 0.981 | 0.972 |

Every one of those rows scores `hit@1 = 1.0`, so the chunk metrics cannot tell them
apart at all. `char_recall@1` falls with chunk size because one small chunk cannot
cover a 1,466-character region; `char_precision@1` rises for the same reason. The
two are traded off by `budget_recall@4000`, which is the question that decides the
indexing choice — given 4,000 characters of context, which cutting reconstructs
more of the answer region? Small chunks win it: three `tok256` chunks fit where one
`base` chunk does.

**`base` is the ruler, not a contestant.** The target *is* a base chunk, so `base`
scores 1.000 by definition and its row carries no information. Variants are
comparable to each other; putting `base` on the same footing needs the tighter
target below.

### What the recall cost

Character overlap fixes half the problem. `char_recall@k` still hands k slots of
1,024 tokens four times the text that k slots of 256 tokens get, so a cut can buy
coverage with size, and until 2026-08-20 nothing in the report charged it for the
text it returned. `## Retrieval Cost (share of the index)` does:

| Variant  | Chunks | Indexed chars | store_share@1 | store_share@10 | recall_per_share@10 at equal recall |
|---|---:|---:|---:|---:|---:|
| base     |   726 | 1,064,470 | 0.235% | 1.167% | 0.771 |
| tok256   | 2,003 | 1,077,031 | 0.046% | 0.471% | 1.909 |
| tok512   |   990 | 1,078,330 | 0.099% | 1.066% | 0.844 |
| tok512ov | 1,049 | 1,151,174 | 0.093% | 0.951% | 0.947 |
| tok1024  |   501 | 1,079,332 | 0.280% | 2.045% | 0.440 |

Measured on `.local/db/pages-variants.db`, with `char_recall@10` held at 0.9 for
every row so the only thing varying is what each cut had to return. `tok1024`
reads 4.3x more of its index than `tok256` for the same answer coverage. **Lower
is better for `store_share@k`, and its table bolds the smallest cell.**

Note `tok512ov`'s indexed size: 1,151,174 characters of the same pages the
others store in about 1,078,000. Overlap is 6.8% more text embedded, stored and
searched, and the metric charges it — chunk lengths are summed unmerged against a
denominator that holds every copy. Characters of *reading*, where a reader sees
overlapped text once, are what `char_precision@k` already measures.

Read this beside `budget_recall@4000`, which fixes the cost instead of pricing
it. Agreement between the two is the strong result; a variant that wins
`char_recall@10` while losing both of these won on size.

### Why each variant's question count scales with its chunk size

Per-variant question generation removes one bias and, in its original form,
introduced another. One question per type per chunk probes a variant once per
*chunk*, so over identical pages the small cut is probed far more densely — on
the sweep database, 17.6 questions per 10,000 indexed characters for `tok256`
against 4.2 for `tok1024`, 7,806 questions against 1,456. And the five questions
squeezed out of a 256-token chunk reach further down it for facts to ask about
than five out of 1,024 tokens, so the large cut gets the more salient questions
as well as the smaller sample. `tok1024` winning every row of the merged
2026-08-18 report cannot be separated from that.

`generate_dataset.py --density` scales the count with the chunk: one question per
256 of its own tokens, so 256 gets one, 512 two, 1,024 four, and every variant
covers the corpus with the same number of questions at the same density. The
budget comes from each chunk's real token count, not its variant's nominal size,
so a page's short tail chunk is asked for one question rather than four. Question
types rotate across a variant's chunks instead of being exhausted within one,
which keeps the category breakdown balanced when a chunk only gets one question,
and the prompt asks for the facts to be spread over consecutive stretches of the
chunk — four questions clustered in a 1,024-token chunk's opening would probe the
same text one 256-token question does and leave three quarters of it untested.

```bash
export PAGES_DB_PATH=.local/db/pages-variants.db
uv run python -m src.eval.generate_dataset --variant tok1024 --density --workers 8
uv run python -m src.eval.generate_dataset --variant tok1024 --density --fill-missing
uv run python -m src.eval.sweep_status   # q/10k ch should match across variants
just sweep-overview --design per-variant-density --variants tok256,tok512,tok1024 ...
```

Pass `--density` to `--fill-missing` as well. Without it the top-up offers every
type for every chunk, which restores the per-type design and leaves the counts
looking right.

**What it costs.** The question model answers reliably for one shape only — the
multi-field type object through `json_schema` — so density mode changes which of
those fields it asks for rather than the schema, and inherits a ceiling of five
questions per chunk. Measured over 163 real chunks on this corpus:

| variant | requested | delivered | per chunk |
|---|---:|---:|---:|
| tok256  |  49 |  49 | 1.00 |
| tok512  | 114 | 112 | 1.87 |
| tok1024 | 215 | 172 | 4.06 |

Two things to read there. The 1,024-token cut loses 20% of what was asked for
against tok256's nothing, because one call asked for five pairs fails more often
than one asked for one — the run prints the shortfall, `--fill-missing --density`
closes it, and `sweep_status`'s `q/10k ch` column is where an unclosed gap shows.
And 3 of its 53 chunks hit the five-question cap, which binds above about 1,280
tokens; a cut larger than `tok1024` is probed below the requested density and the
run says so rather than applying the cap quietly.

Declare it as `--design per-variant-density`. The sweep tells the designs apart
by whether one question is labelled in more than one variant rather than by
counting rows, because matching row counts are exactly what density mode
produces — the old count-based check would have called these a shared question
set and published per-variant columns under a shared-design heading.

**What density normalisation does not fix.** Each variant's questions were still
written from its own chunks, with its own boundaries in view. Only pooling
removes that: anchor the questions (`src/eval/anchors.py`), project them onto
every variant (`src/eval/relabel.py`), and score every variant on the union as
`--design shared`. Equal counts are what make that pool balanced instead of
dominated by whichever cut produced the most questions, so the two changes
compose rather than compete.

`base` cannot join a density grid without regenerating its questions: its 3,476
approved rows are the legacy per-type set, and a chunk that already has questions
is skipped. The design check refuses the mixed grid rather than publishing it.

### The optional anchor target

`load_span_labels(variant, target="anchor")` uses the span of the quoted sentence
that states the answer (`src/eval/anchors.py`) instead of the whole paragraph. It
costs a model pass of about 75 minutes and lands for roughly six questions in ten,
and it buys exactly one thing: a target that is not a base chunk, so `base` can be
ranked alongside the variants. On 57 anchored questions `base` scores
`char_precision@1` 0.139 against `tok256`'s 0.314.

Never mix the two targets in one table — a paragraph-sized target and a
sentence-sized one put recall and precision on different scales.
