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

# Paired legacy/v2 reranker and agent comparisons
just eval-index nemotron v2
uv run python experiments/indexing/compare_qwen_modes.py \
	--methods phase2-nemotron \
	--output docs/retrieval-results-phase2-nemotron.md
```

`phase2-nemotron` and `phase2-azure` compare each provider's v1 and v2
hybrid-reranked baseline and agent. `phase2` includes both provider groups. The
v2 indexes are separate derived artifacts, so existing collections, retrieval
method names, reports, and checkpoints remain valid.

When a selected agentic method and its matching reranked baseline are both in
the run, the report adds paired agentic diagnostics at the configured category
hit cutoff. For example, `nemotron_hybrid_agentic_v2` is paired with
`nemotron_hybrid_rerank_v2`. The summary reports retry/rewrite/expansion counts,
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
normal indexing path. Azure requires `.env` values and asks for typed approval
before building embeddings.

`config.yaml` sets the default warmup count. Warmup queries are excluded from
timing.

By default, `compare_qwen_modes.py` runs the primary chunk benchmark: sparse
rerank, Qwen4B hybrid rerank, Nemotron vector and hybrid rerank, Azure hybrid
rerank, and the Azure and Nemotron hybrid agentic methods. It retains the
historical `docs/retrieval-results-chunks-okf.md` output name so existing chunk
checkpoints continue to resume:

```bash
uv run python experiments/indexing/compare_qwen_modes.py
```

### Evidence-equivalence judge

Strict chunk metrics remain the primary retrieval metrics. A separate
post-hoc judge can audit strict misses to determine whether the retrieved
evidence nevertheless contains the same answer-bearing facts as the golden
chunks. It reports both collective evidence equivalence and cases containing an
individually near-duplicate retrieved chunk. This exposes alternate-source,
translation, and chunk-boundary duplication without silently changing qrels.

Keep the completed retrieval checkpoint, then run the audit:

```bash
uv run python experiments/indexing/compare_qwen_modes.py --keep-checkpoint
uv run python experiments/indexing/judge_retrieval_equivalence.py \
  --checkpoint docs/retrieval-results-chunks-okf.md.checkpoint.json \
  --output docs/retrieval-equivalence-judge.md -k 5
```

Alternatively, run the audit as part of the comparison. This adds a
`judge_hit@K` column to the overall table and writes a resumable judgment cache
beside the comparison report:

```bash
uv run python experiments/indexing/compare_qwen_modes.py \
	--keep-checkpoint --judge-equivalence --judge-k 10
```

Integrated judging runs immediately after each method's retrieved ranking.
Exact strict hits do not require an LLM call; strict misses are sent to the
equivalence judge and persisted before retrieval continues. On resume, any
older checkpointed predictions missing judgments are backfilled first.

The speed table also reports `chunk expansions`: the number of times an agentic
retriever increased its requested chunk count between attempts. Query
reformulations at the same chunk limit are not counted as expansions.

The judge defaults to Azure Foundry through `AZURE_AI_*`. Use
`--provider local --model <ollama-model>` for local inference. Results are
checkpointed after every judgment next to the output report, so interrupted
runs resume without repeating model calls. `--limit` provides an inexpensive
calibration sample per method. Manually review a stratified sample before using
judge-adjusted rates for conclusions.

The Qwen and Nemotron methods retrieve the same canonical SQLite chunks from
independent embedding collections. `sparse_rerank` is the embedding-independent
lexical baseline. Use `--methods all-agentic` for the previous agentic-only
default suite.

Agentic methods normally report the standard top-10 metrics. If an agentic
retriever expands beyond rank 10, the report also adds hit, recall, and MRR at
the deepest observed rank. Those expanded columns contain values only for
agentic methods; non-agentic baselines are shown as `-` because they were not
retrieved beyond the standard cutoff.

The retired OKF experiment used the following protocol. It remains documented
for interpreting historical findings, but has no active runner.

## Historical OKF versus chunk-RAG benchmark protocol

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

Keep existing chunk qrels for native RAG evaluation. The shared track projects
both relevance and rankings onto the OKF concept ontology:

1. A gold chunk maps to its source page.
2. The page maps to every concept listing it in `source_page_ids`; these are the
	question's golden concepts.
3. RAG results map from ranked chunks to pages to concepts, preserving first
	occurrence and removing duplicates.
4. OKF results are already ranked concepts: cited concepts first, then other
	visited concepts.

Reports label the resulting metrics `concept_hit@K`, `concept_recall@K`, and
`concept_mrr@K`. Thus an English or Slovenian sibling page receives full credit
when both pages belong to the castle concept. No fuzzy answer-token or substring
matching expands qrels at evaluation time: phrase occurrence does not prove that
a page expresses the same entity or supports the answer. The optional
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
cold/warm state, Azure region, pricing date, seeds, and query order. Avoid
unrelated machine load and repeat remote runs at comparable times.

Use at least five repetitions for inexpensive local timing and three for costly
full Azure builds when budget permits; label fewer runs exploratory. Report
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
[docs/archived_okf-rag-results-2026-07-15.md](../../docs/archived_okf-rag-results-2026-07-15.md).
Treat it as historical evidence only: it is not directly comparable with the
later broad chunk evaluation.
