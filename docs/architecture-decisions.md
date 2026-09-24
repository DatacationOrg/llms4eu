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
- `qwen`, `qwen4b`, `qwen8b`, `nemotron`, `nemotron8b`: dense/vector providers,
  one entry each under `providers` in `src/indexing/config.yaml`.
- `*_hybrid`: dense provider plus sparse retrieval.
- `*_rerank`, `*_rerank_4b`: rerank candidates from one base retriever with the
  Qwen3 0.6B or 4B cross-encoder (`rerankers` in `src/retrieval/config.yaml`).
- `*_hybrid_rerank`, `*_hybrid_rerank_4b`: rerank hybrid candidates.

No `_chunk` suffix in public method names because chunk retrieval is current
target. No `sparse_hybrid`; hybrid means dense plus sparse.

## Hybrid History

RRF was tested as hybrid fusion. Weighted normalized score fusion performed
better in measured runs and is current active hybrid strategy.

Historical read:

- Azure `embed-v-4-0` weighted hybrid was the strongest method ever measured,
  but that provider has been removed (see "Local-Only Inference" below).
- Qwen 4B weighted hybrid was the strongest remaining method as of July 2026,
  before the reranker fix below.
- Default hybrid weights are 70% vector, 30% sparse.

Keep RRF as historical context in research docs unless new eval justifies
active support.

## Rerank History

Reranking retrieves `N` candidates, then returns final top `M`. Current defaults:
30 candidates, 10 final results.

July finding, kept as a record: measured Qwen3 reranking was harmful and slow
in our setup, so rerank methods stayed experimental until model usage, prompt,
and input formatting were diagnosed.

Diagnosis: the July "reranking is harmful" result was an input-formatting bug: the Qwen3
reranker needs its chat template from the model cache, and without it scores
collapse (hit@5 0.774 against 0.881 when the file went missing again on
2026-09-07). With it, the reranker is the dominant factor. On the August sweep
(3,471 questions, base chunks, hit@5): `qwen8b_hybrid_rerank_4b` 0.950 is the
best pipeline; `nemotron8b` alone reaches 0.940 at 11 ms/query, above every
0.6B-reranked pipeline (0.88-0.92), and the 0.6B reranker lowers it to 0.917.
`nemotron8b_hybrid_rerank_4b` was never measured. Source:
`docs/reports/retrieval/retrieval-results-comprehensive-2026-09-07.md`.

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

`src.indexing` owns the whole indexing side: `embedders` builds the provider,
`cache` memoizes vectors, `store` owns Chroma mechanics and query-time search,
and `__main__` is the rebuild CLI.

`src/indexing/config.yaml` owns which embedding providers are enabled for this
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

- `$LLMS4EU_DATA/db/pages.db`: page SQLite DB.
- `$LLMS4EU_DATA/chroma/`: regenerable shared vector cache.

`LLMS4EU_DATA` defaults to `/data/llms4eu` and is set in `.env`. Point it at a
private path when a run should not touch the shared store.

`just index qwen` rebuilds the default chunk-vector provider.

## Open Knowledge Format (removed 2026-09-14, kept as a record)

> The code described below was deleted. Nothing in `src/` implements it
> today; the results are in `docs/reports/okf/comprehensive-okf-results.md`.

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
latency, throughput, build cost, and storage. The protocol that ran these is
archived in [`okf-benchmark-protocol.md`](okf-benchmark-protocol.md).

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

## Direct Corpus Interaction (removed 2026-09-24, kept as a record)

Measured out: -18.4pp hit@5 overall and -50.5pp cross-lingual against the
baseline (`docs/reports/agentic/agentic-retrieval-report-2026-09-01.md`). The
code stayed on main at the #18 merge; what it was:

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

## Geographic Scope, full version (superseded 2026-09-24, kept as a record)

The implementation of #18 below; today's code keeps only its measured core
(see "Geographic Scope" at the end). Its design documents are in
`docs/reports/geo/`.


Pages carry where they are about; questions resolve to a scope; retrieval
re-scores on it (soft, the default) or filters, widens and boosts (strict, for
measurement). Standards first, model second.

**Data model** (`sql/eval.sql` `page_locations`, 0..n rows per page). Coordinates
(WGS84) are the canonical fact. ISO 3166-1 country codes and Eurostat NUTS-2 /
NUTS-3 codes are *derived* from them by point-in-polygon against the GISCO 2024
boundaries (`src/shared/nuts.py`), so a NUTS revision is `just locate-pages
--recompute-codes`, not a re-extraction. Names are display labels and are never
filtered on: they are ambiguous and multilingual (Štajerska / Styria /
Steiermark), and Slovenia's statistical regions are not administrative units, so
Wikidata's `P131` chain skips them. NUTS is the one region hierarchy that is
consistent across EU countries, which is what an EU-wide corpus needs. One
`primary` row per page is denormalised into chunk metadata; `mentioned` rows stay
in SQLite for the agent's geo tools.

**Enrichment** (`src/preprocess/locations.py`, read-only until `--apply`) in
three tiers, each recorded in `method`: Wikidata for Wikipedia pages (`P625`,
`P605`, `P300`, with `P31` telling a person or a concept from a place, and a
country or administrative parent required, because a language item can carry
coordinates); a configured Wikidata QID per single-site source
(`src/preprocess/config.yaml` `source_locations`); and, last, an LLM naming the
place a page is about, resolved through Nominatim and kept only when the hit's
name matches what the model said. The model never produces coordinates
(Hu et al. 2024). Adding rows never touches `page_chunks`, so the approved
labels are safe.

**Retrieval** (`src/retrieval/retrievers/geo.py`, methods `*_hybrid_geo`,
`*_hybrid_rerank_geo`, `*_hybrid_rerank_geo_strict`). A `GeoScope` is resolved
once per question (`src/shared/geo_resolver.py`: the judge LLM names the
anchoring place and its width, a gazetteer places it; corpus names, then NUTS
names, then Wikidata, then Nominatim; cached in `geo_scope_cache`, but only a
settled answer is cached: a failed extraction or a place the gazetteer could not
find is retried next time rather than frozen into "no scope").

*Decision 2026-09-09: geography re-ranks; it does not pre-filter.* The first
measured run (2026-09-08, 500 questions) filtered the stage to the scope and lost
15 of the 72 scoped questions against the text baseline, winning one. Every loss
was a gold page with no footprint (biographies, Celeja, national-scope pages)
that the hard filter removed, and widening never fired because five located
pages always supplied chunks. With 89 of 176 pages located, any hard filter is a
coverage lottery. So the default path over-fetches `limit * geo_overfetch`
chunks unfiltered, reranks as the baseline does, and recombines each score as
`(1 - w) * text + w * s_geo` with both in [0, 1] (`w = 0.3`): `s_geo` is
`exp(-km / decay)` to a point scope, in/out of a region scope, and **1.0 for a
page with no location**. Unknown footprint is neutral, never "outside". A region
level is applied only when it keeps at most `geo_max_scope_share` (90%) of the
located pages; with 85 of 89 in one NUTS-3, "Slovenija" and "Posavska" are
no-ops on this corpus. The convex combination of normalised scores is the
fusion Bruch et al. (TOIS 2023) found beats rank fusion; the neutral-for-unknown
rule and the selectivity gate are the corpus facts from
[geo-improvement-plan.md](reports/geo/geo-improvement-plan.md) §1 and §3.

The strict shape survives as `*_hybrid_rerank_geo_strict` for measurement, with
two changes from the 2026-09-08 run: unlocated pages pass the filter
(`geo_include_null: true`) and widening counts located in-scope chunks rather
than all returned chunks, so the filter widens when *it* contributed too little.
The scope reaches the stage through constructor fields, `VectorChunkRetriever.where`
(Chroma `where`) and `SparseRetriever.allowed_page_ids`, so the `Retriever`
protocol and the eval harness are unchanged. This is the Spatial-RAG shape (Yu
et al. 2025), and the agent's `pages_in_region` tool keeps it because there the
user asked for containment.

Geo is payload metadata rather than a bespoke scoring pass so the planned Chroma
-> Qdrant move keeps it: Qdrant has native `geo_radius` filters, Chroma only a
`$gte/$lte` bounding box on the stored latitude and longitude.

**Agent tools** (`src/retrieval/retrievers/agentic_tools.py`): `find_pages_near
(place, radius_km)` and `pages_in_region(nuts_or_iso_code)` return each located
page's first chunk id *of the chunk variant being scored*, so what the agent
finds stays scoreable, like `list_sections` and `search_in_page`. They exist
only on the `*_geo` tool agent; the plain `*_hybrid_agentic_tools` keeps the
tool set its published numbers were measured with.

Not done: the legacy `places` collection behind `src.rag.agent_search` carries no
location payload; `WiderGeoFilter` now widens a real `GeoScope`, but the
places search it schedules does not filter. The page-chunk stack is where the
filter applies.

## Local-Only Inference

> Still true for what remains: embeddings, reranking and question
> generation. The agentic sufficiency judge, the OKF navigator and the
> evidence-equivalence judge named below have since been deleted.

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

## Corpus Expansion (2026-09-13)

The seed-URL tooling (`seeds.yaml`, `seed_urls.py`, `seed_quality.py`) stayed on
main at the #18 merge; its output lives under `/data/llms4eu` (`data/seeds/`,
`v2/seeds/corpus-v2.json`) and is fetched with `just fetch-pages <file>`.

The corpus grew from one locality to ten. Brestanica (176 pages, four Slovenian
sources, 1.08M chars) stays as it was fetched and labelled and is the gold
standard; nine clusters of the same shape were added around it:
`src/scraping/seeds.yaml` declares, per locality, the attraction's own site, the
town's site, the national biographical lexicon and the national-language
Wikipedia, and `src/scraping/seed_urls.py` discovers, verifies and writes the
seed files. Result after quality pruning: 1,338 pages, 14.5M chars, 11,261
base chunks, 38 sources, 8 languages, 690 pages located in 8 countries.

Why: the September geo runs found the spatial signal a constant (85 of 89 located
pages in one NUTS-3, so region scopes were no-ops), and the chunk-size sweep
found a 1,024-token cut returning a few percent of the whole store at k=10, so
size alone could buy recall. Both need a corpus spread over regions and about
an order of magnitude larger. Now eight NUTS-3 regions hold 58 or more located
pages each (SI036, SI032, HR064, AT224, HU222, CZ064, SK022, DE214), the
Croatian cluster is 35 km from Brestanica across a border, the two Czech
clusters are 8 km apart, and a 1,024-token cut at k=10 reads about 0.3% of the
store instead of 3.8%.

Decisions:

- **Lexicon entries come from Wikidata, not lexicon search pages.** People born
  in the town or its district with the lexicon's identifier property, most
  linked first, URL from the property's formatter. The lexicon search pages are
  JavaScript-rendered and yield nothing. Slovakia has no such property, so its
  people come from the Wikipedia category instead.
- **The reference is never touched.** Language detection was applied with the
  four Brestanica sources excluded (`--exclude-source`), locations with only
  the new sources selected (`--source`, now repeatable), and the seed builder
  drops any URL in `data/brestanica.json`. The last rule exists because two
  Ptuj wikilinks (Slovenija, Statistični urad) were already reference pages;
  the fetch upsert moved them to the new source and refreshed their text
  before this was caught, and they were restored from a pre-fetch copy.
- **Quality is measured against the reference of the same kind**
  (`src/scraping/seed_quality.py`, report in `docs/reports/scraping/`). The
  Brestanica castle and town sites themselves have a third stub pages, which
  sets the bar. Two cuts were applied to new sources only: 151 pages with under
  300 characters of prose (mostly Czech lexicon index records, which exist for
  people whose entry is not yet written) and 91 pages whose content repeated
  within the source (cookie notices and navigation blocks the extractor fell
  back to on Desinić, Bojnice, visitptuj and Miramare). The seed builder now
  rejects pages with under 300 extracted characters at seed time. One source
  stays flagged: Deutsche Biographie entries are real but short (median 753
  prose chars against 2,502 for Slovenska biografija).
- **Geo tiers 1 and 2 only, so far.** 602 of 1,159 new pages got a primary
  location from Wikidata or the source default; the LLM + Nominatim tier has
  not run on them, and `geo_country_hint` is still `si`, which mis-geocodes
  foreign place names (see the 2026-09 country-hint note). Miramare's point
  falls outside every NUTS polygon (it sits on the coast), so those 35 pages
  have coordinates but no region code.

Consequence: every report in `docs/` dated before 2026-09-13 was measured on the
176-page corpus. The approved questions and labels still cover only that part;
new questions for the added pages are a separate step.
