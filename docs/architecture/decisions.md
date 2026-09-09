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

## Direct Corpus Interaction

`dci` gives the agent the corpus as files instead of a vector index: BM25 picks
a bounded working set of at most K pages, `src/retrieval/workspace.py`
materializes them as line-numbered Markdown, and the agent explores with
`search`, `read` and `toc` until it can name chunk ids. Sweep K with
`dci_k10` / `dci_k50` / `dci_k200`, since larger K is not monotonically better.

Every chunk is written behind a `<!-- chunk: <id> | <heading> -->` marker, so a
grep hit at a line maps back to exactly one `page_chunks.id`. That marker is the
whole reason the method is scoreable against the existing chunk qrels rather
than needing its own ground truth.

**No `bash()`.** The papers hand the agent a general shell. This gives it three
bounded tools; `search` invokes ripgrep through a fixed argument vector, never a
shell string, so model output is always a search pattern and never a command. A
benchmark does not need arbitrary command execution to measure retrieval.

Two invariants protect the scores:

- A cited chunk id that does not exist is dropped, never ranked. A hallucinated
  id would otherwise be scored as a confident wrong answer.
- The BM25 shortlist is appended below the agent's citations, so a ranking that
  stops after two cited chunks cannot score worse than BM25 merely for being
  short.

Sized for a corpus far larger than today's 176 pages: pages stream out of SQLite
one at a time, each page carries a content hash so a rebuild rewrites only what
changed, and path/page/chunk lookups are materialized once per load instead of
per query.

`CorpusAction` needs `method="function_calling"` (3/3, against 0/3 for
`json_schema` at every `num_predict`, K and reasoning level tried) — the
opposite of `ToolAction` in the tool agent. The right structured-output mode is
a per-schema measurement, not a per-model setting.

## Local-Only Inference

Every model call in the repo runs on local hardware. There is no hosted-model
or credential path in the tree.

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
