# OKF versus chunk-RAG benchmark protocol (archived)

The OKF experiment ended and its code was removed; this is the measurement
protocol that was designed for it, kept as a record. Results are in
`comprehensive-okf-results.md`; the generator design is in
`okf-method-notes.md`.

---


A fair comparison freezes one `$LLMS4EU_DATA/db/pages.db` snapshot and records its hash,
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
[docs/archived_okf-rag-results-2026-07-15.md](../../docs/archived_okf-rag-results-2026-07-15.md).
Treat it as historical evidence only: it is not directly comparable with the
later broad chunk evaluation.
