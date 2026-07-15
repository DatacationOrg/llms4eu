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

just okf-benchmark 19
```

`evaluate_embedding_models.py` compares the embedding providers listed in
`src/indexing/config.yaml`. Missing vector collections are built through the
normal indexing path. Azure requires `.env` values and asks for typed approval
before building embeddings.

`config.yaml` sets the default warmup count. Warmup queries are excluded from
timing.

`compare_qwen_modes.py` compares qwen vector search, sparse hybrid search,
reranking, and hybrid plus reranking.

These scripts remain retrieval experiments. Their chunk qrels cannot be used
directly for OKF because concepts and chunks have different granularity.

`compare_okf_rag.py` is a coverage-matched evidence-acquisition pilot. It selects
only approved questions whose gold source page occurs in the current OKF bundle,
then compares Qwen hybrid RAG, agentic Qwen hybrid RAG, and OKF navigation. Its
speed table reports `seconds`, `ms/query`, `queries/query`, and total `queries`;
it also reports source-page hit and MRR. RAG queries count retrieval attempts,
while OKF queries count Azure navigation actions. The pilot does not replace the
blinded end-to-end answer benchmark.

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
2. OKF discovery inventory, canonical catalog, enriched concepts, indexes, and
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

Keep existing chunk qrels for native RAG evaluation and report `nDCG@10`,
`MRR@10`, `Recall@10`, `Precision@10`, and hit rates. Do not score OKF concept
IDs against chunk IDs.

For a shared page-evidence track, map RAG chunks to source pages and OKF
concepts to their `source_page_ids`. Rank RAG pages by their highest-ranked
chunk and OKF pages by evidence order, de-duplicating while preserving rank.
Human reviewers should label pages not, partially, or fully relevant. Both
systems can then use page-level `nDCG@10`, `Recall@10`, and success metrics.

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
