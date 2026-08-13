# Architecture Decisions

Long-lived context for retrieval, indexing, eval, and data artifact decisions.

## Retrieval Ownership

Retrieval/RAG owns reusable ranking methods. Eval owns measurement.

- `src.retrieval.base` defines `RankedChunk` and the `Retriever` protocol.
- `src.retrieval.methods` is the public method catalog:
  `list_retrievers`, `build_retriever`, `ensure_retriever_ready`.
- `src.eval` owns datasets, labels, metrics, timing, and reports.

Reason: eval should compare application retrieval methods, not own them.

## Retrieval Names

User-facing names describe method intent, not storage details.

- `sparse`: lexical retrieval, currently BM25 internally.
- `english`, `qwen`, `qwen4b`, `nemotron`: dense/vector providers.
- `*_hybrid`: dense provider plus sparse retrieval.
- `*_rerank`: rerank candidates from one base retriever.
- `*_hybrid_rerank`: rerank hybrid candidates.

No `_chunk` suffix in public method names because chunk retrieval is current
target. No `sparse_hybrid`; hybrid means dense plus sparse.

## Hybrid History

RRF was tested as hybrid fusion. Weighted normalized score fusion performed
better in measured runs and is current active hybrid strategy.

Historical read:

- Azure `embed-v-4-0` weighted hybrid was the strongest method ever measured,
  but that provider has been removed (see "Local-Only Inference" below).
- Qwen 4B weighted hybrid is the strongest remaining method.
- Default hybrid weights are 70% vector, 30% sparse.

Keep RRF as historical context in research docs unless new eval justifies
active support.

## Rerank History

Reranking retrieves `N` candidates, then returns final top `M`. Current defaults:
30 candidates, 10 final results.

Measured Qwen3 reranking was harmful and slow in our setup. Keep rerank methods
available for experiments, but do not treat rerank as default until model usage,
prompt, and input formatting are diagnosed.

## Geo-Aware Ranking

Places can carry a `latitude`/`longitude` pair. `just geocode` derives them
once per place: a local LLM names the one real-world place the text
describes, and that name is geocoded via OpenStreetMap Nominatim
(`src.shared.geocode`). Coordinates are stored on the `places` row itself the
same way Chroma vectors are a derived cache — `just init` does not generate
them, `just geocode` does, and existing coordinates are left alone on rerun
unless `--force` is passed.

At query time, `geo.enabled` in `src/rag/config.yaml` gates a second local
LLM call that asks whether the question names a place; if so, that name is
geocoded the same way and every retrieved place's score is rescaled by
distance (`src.rag.geo.apply_geo_boost`):

```text
final_score = semantic_score * ((1 - weight) + weight * exp(-distance_km / decay_km))
```

A place with no coordinates, or a question with no detected location, is
left at its plain semantic score rather than penalized — missing geocoding
data should never push an otherwise relevant place out of the answer. The
multiplicative form also means distance can only pull a place's rank down
toward `(1 - weight)` of its semantic score, never to zero, so a strong text
match still beats a mediocre one that merely happens to be closer. This
mirrors the weighted-fusion shape already used for hybrid vector+sparse
scoring (see "Hybrid History" above) rather than a hard radius filter, which
risks dropping the correct place on a wrong or missing geocode.

Distinct from the unused `geo_filter` on `AgentSearchState`
(`src/rag/retry.py`): that scaffold widens a discrete NUTS2-region/
country-code filter during agentic retries and is never currently populated.
This feature is a continuous per-query distance signal computed from
coordinates, not a categorical filter; the two can be unified later if a real
regional-filter need appears, but there was no reason to force them together
now.

Reason `geo.enabled` defaults to `false`: enabling it adds a local LLM call
and an external Nominatim lookup to every query. Turning it on is only
useful once `just geocode` has populated real coordinates, which the tracked
dummy fixture data cannot produce (its place text never names a real place).

Reason for choosing Nominatim over the Google Maps Geocoding API: no API key,
account, or billing risk, at the cost of being a new external network
dependency either waay

## Chunk Storage

SQLite is source of truth for page chunks. Chroma is derived vector cache.

- `page_chunks.text` stores canonical chunk text.
- Chroma stores embeddings plus minimal ids.
- Vector retrieval hydrates chunk text from SQLite.
- Chroma readiness means collection exists and count matches `page_chunks`.

## Chunk Summaries

Chunk summaries were useful experiment artifacts, but are not part of
steady-state DB or retrieval code.

Historical result: chunk+summary often measured well, but summary generation
added extra derived content and operational cost. Current steady-state indexes
title, heading path, and chunk text.

## Indexing Boundary

`src.indexing` orchestrates rebuilds. `src.vector_store` owns Chroma mechanics
and query-time vector search.

Provider-shaped embedding implementations live in `src.shared.indexers`.
`src.indexing/config.yaml` owns which embedding providers are enabled for this
project. Retrieval builds `VectorChunkRetriever` instances from the enabled
provider list and checks Chroma readiness without owning provider definitions.

## Config Boundary

Experiment/runtime policy belongs in YAML config, not hidden in constructors.

- Retrieval limits, hybrid weights, reranker settings:
  `src/retrieval/config.yaml`.
- Indexing providers, default provider, model batch settings:
  `src/indexing/config.yaml`.
- Eval result limits and metric cutoffs: `src/eval/config.yaml`.

Implementation classes may keep defensive library defaults only when they are
not project policy.

## Data Artifacts

Tracked durable artifacts use descriptive paths:

- `data/db/pages.db`: page SQLite DB.
- `data/cache/chroma/`: regenerable shared vector cache.

`.local/` is private scratch and can be overridden through `.env`.

`just eval-index qwen` rebuilds the default production chunk-vector provider.

## Open Knowledge Format

The OKF experiment is a second knowledge representation over the same canonical
raw pages, not another chunk retriever.

- `page_metadata` and `page_markdown_content` are its read-only source.
- The local model discovers conservative page proposals, resolves them against a
  global canonical catalog, and enriches the resulting fixed concepts.
- Existing durable concepts seed canonicalization; exact normalized titles are
  resolved before model-assisted batched resolution.
- Generated concepts use minimal OKF frontmatter, retain source URLs in the
  standard body citations section, and keep source page IDs only for evaluation.
- Discovery inventory, canonical catalog, and enrichment checkpoints stay under
  `.local/okf/`; the validated bundle lives in `data/okf/tourism/`.
- Online answering follows bundle indexes and reads complete concept files; it
  does not reopen raw database pages or select source-text windows.
- Normal generation resumes incrementally. A deliberate clean rebuild deletes
  the bundle and all private OKF state first, preventing stale concepts and old
  generated prose from seeding a redesigned catalog.
- Shared retrieval evaluation projects gold and retrieved chunks through their
  source pages to OKF concept IDs. This credits translated and duplicate sibling
  pages when canonicalization assigns them to the same concept.

Reason: exact-page scoring penalizes valid alternate sources, while direct
concept-to-chunk comparison conflates representation granularity.

## OKF Versus RAG Benchmarking

Benchmarking is split into clean build, incremental update, online
retrieval/navigation, shared page-evidence retrieval, and end-to-end answer
quality. There is no composite OKF-versus-RAG score.

- Both representations use one frozen and hashed raw-page snapshot.
- Shared scraping and Markdown extraction are excluded from representation
  build time.
- RAG chunk qrels remain valid for RAG-only retrieval studies.
- Shared retrieval derives golden concepts from existing chunk qrels and OKF
  `source_page_ids`, then projects every method to the same concept ontology.
- Build results include wall time, throughput, coverage, retries, model usage,
  cost, memory, artifact size, and storage amplification.
- Query results include p50/p95/p99 latency, throughput, failures, context use,
  calls, and steps.
- Answer results include factual correctness, faithfulness, citation validity
  and entailment, abstention, and blinded paired judgments.
- Raw per-item observations are retained and paired bootstrap confidence
  intervals accompany material quality claims.

Reason: industry IR and search benchmarks report effectiveness together with
latency, throughput, build cost, and storage. The complete protocol and sources
are in [`experiments/indexing/README.md`](../experiments/indexing/README.md).

## Agentic Page Tools

`*_hybrid_agentic_tools` extends the agentic loop with two read-only tools over
the source page behind a retrieved chunk: `list_sections(page_id)` for a table
of contents and `search_in_page(page_id, term)` for sibling chunks. Both return
real `page_chunks.id` values, so anything the agent finds is scored against the
existing chunk qrels. Found chunks are inserted directly after the ranked chunk
of the same page, because the anchor's rank carries the retrieval system's
confidence in that page.

It is a separate method rather than a change to `*_hybrid_agentic`, so the
plain agent stays a live baseline in the same run.

Measured motivation, over 300 questions with `qwen_hybrid_rerank`: 75.3% already
hit@10, **13.3% miss while the gold chunk's page is nevertheless ranked** (what
these tools can recover), and 11.3% miss with the page absent (out of reach).

The prompt, not the plumbing, decides whether tools are used at all. Over the 40
recoverable misses:

| prompt | search_in_page | reformulate | recovered |
|---|---|---|---|
| "do these chunks answer the query?" | 1 | 37 | 2/40 |
| literal-answer gate + unseen-chunk counts | 42 | 0 | 5/40 |

Three properties earn their keep and should survive edits:

- **Sufficiency requires a literal answer**, not topical relevance. The first
  prompt declared `sufficient` on 25 of 40 questions that were all misses, which
  blocked every tool call downstream.
- **Each chunk reports `showing N of M chunks from this page`.** Without it the
  agent cannot know unread material exists on a page it already has.
- **`reformulate` is the last resort**, because it discards a page that was
  already correct.

Repeated identical tool calls are answered from the prompt instead of rerunning
the query: the first version burned its whole budget re-searching one term.

## Local-Only Inference

Every model call in the repo runs on local hardware. There is no hosted-model
or credential path in the tree. Geo-aware ranking's Nominatim lookup (see
"Geo-Aware Ranking" above) is the one exception, and it is a plain geocoding
HTTP call, not a model call or credential — no API key involved.

- Embeddings and reranking run through sentence-transformers; the agentic
  sufficiency judge, the OKF navigator and generator, the evidence-equivalence
  judge, and eval question generation all run through Ollama behind the
  `StructuredLlm` protocol in `src/shared/llm.py`.
- Agent model is `gpt-oss:20b` at low reasoning effort, set by
  `agentic_judge_model` in `src/retrieval/config.yaml`, `model` in
  `src/okf/config.yaml`, and `DEFAULT_LOCAL_MODEL` in the equivalence judge.
- Removed: the Azure Foundry chat client (`DeepSeek-V4-Pro`), the Azure
  embedding provider (`embed-v-4-0`, published as `embed_v4_*`), and the
  Azure-hosted Cohere reranker (`*_cohere*`).

Reason for going local: the hosted path cost quota, imposed a ~125k tok/min
rate limit on the agentic judge, hit content filters on tourism source text,
and made runs non-reproducible for anyone without the credentials.

Reason for `gpt-oss:20b` specifically: it was chosen on measured structured-output
reliability, not on reputation. Over 10 real sufficiency prompts (8k-20k chars)
built from live `qwen_hybrid_rerank` results:

| model | mode | valid verdicts | median |
|---|---|---|---|
| `gpt-oss:20b` (low) | tool call | 10/10 | 1.9s |
| `gemma4:31b` | tool call | 10/10 | 7.8s |
| `gemma4:26b` | tool call | 5/10 | 2.7s |
| `gemma4:26b` | json_schema | 0/10 | - |

Every `gemma4:26b` failure was a Slovenian question, where it returns no tool
call at all; in `json_schema` mode with reasoning off it answers in prose
instead of JSON on all inputs. `gemma4:31b` is reliable but 4x slower for a
judge called once per retrieval attempt.

Two consequences for the LLM layer:

- Structured output uses `method="function_calling"` (tool calls), not
  `json_schema`. `structured_local_model` takes a `method` argument so callers
  that still work with `json_schema` are unaffected.
- `LocalOllamaStructuredLlm.structured_output` treats a `None` result as a
  failed attempt and retries, then raises. Tool-call mode returns `None` when
  the model answers without calling the tool, and LangChain's `with_retry` does
  not catch that because it is not an exception.

Consequence: agentic and `embed_v4_*` numbers in existing `docs/` reports were
produced with the hosted models and are not directly comparable to new runs.
Label them as historical rather than re-baselining old reports.
