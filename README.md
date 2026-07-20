# LLMs4EU - Tourism - RAG

Local RAG for tourism places. SQLite stores canonical content, and Chroma stores
derived vector indexes. Both run entirely inside the Python environment.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [just](https://just.systems/)
- [Ollama](https://ollama.com/) — only needed for `just ask`, `just scrape`,
  and `just eval-generate`

## Setup

```bash
uv sync --extra dev   # install all dependencies
cp .env.example .env  # set local paths (defaults work out of the box)
```

```bash
ollama pull gemma4:e4b
# Optional for eval question generation:
ollama pull gemma4:26b-a4b-it-q4_K_M
```

## Usage

```bash
just init        # load seed data into SQLite
just index       # embed places and rebuild Chroma
just ask What place is best for a quiet forest walk near water?  # retrieve + LLM answer
just scrape https://example.com  # crawl a site, ingest into SQLite, reindex
just scrape-web  # start the local scraping UI
just eval-index qwen  # rebuild the default qwen chunk vector index
just test        # run the non-LLM test suite
```

Build an OKF v0.1 knowledge bundle from complete scraped pages with the existing
Azure Foundry credentials:

```bash
just okf-pilot     # small resumable source pilot
just okf-generate  # resume across all eligible pages
just okf-refresh-retrieval  # re-enrich existing assignments with retrieval metadata
just okf-validate  # check document conformance and links
```

OKF generation is a resumable three-phase pipeline: complete pages produce a
discovery inventory, proposals are resolved into a global canonical catalog,
and only then are fixed concepts enriched and indexed. Private phase state lives
under `.local/okf/`; the validated, portable bundle lives under
`data/okf/tourism/`.

See [src/okf/README.md](src/okf/README.md) for the complete generation workflow
and operational reference.

The OKF and RAG representations must not be compared by matching concept IDs to
chunk IDs. See [experiments/indexing/README.md](experiments/indexing/README.md)
for the clean-build, incremental-update, retrieval, latency, cost, and
answer-quality protocol.

### Improved OKF retrieval

OKF now has two independently measurable retrieval modes:

- `okf` keeps pure progressive navigation through the generated hierarchy.
- `okf_search` first applies local BM25-style search to concept metadata
    (title, description, aliases, tags, search terms, and language), then lets the
    same Azure navigator inspect the shortlisted **complete concepts**. It does
    not retrieve page chunks.

Both modes now use an independent evidence-sufficiency check before accepting
an answer, controlled backtracking with budgets of 12 actions and 8 complete
concepts, query-aware ordering of visited concepts and their source-page
provenance, and a per-question action trace. Future enrichment preserves richer
`search_terms` and `source_evidence` metadata and explicitly asks Azure not to
drop small supported facts. New bundle generation extracts atomic facts,
source-specific summaries, multilingual aliases, and realistic named and vague
retrieval questions for every page; these are preserved deterministically in
frontmatter and a source-evidence section. Generated indexes include aliases,
search terms, retrieval hints, and tags;
collections larger than 20 entries are split into bounded browse indexes, and
root collection descriptions list representative entities instead of only an
entry count.

Rebuild the deterministic indexes after changing or importing a bundle:

```bash
uv run python -c "from pathlib import Path; from src.okf.bundle import regenerate_indexes; regenerate_indexes(Path('data/okf/tourism'))"
just okf-validate
```

Compare the pure and metadata-assisted variants without overwriting an older
checkpoint:

```bash
uv run python experiments/indexing/compare_qwen_modes.py \
    --methods sparse_rerank,qwen_hybrid_rerank,okf,okf_search \
    --okf-bundle data/okf/tourism --limit 100000 --warmup 5 \
    --output .local/retrieval-results-okf.md \
    --checkpoint .local/retrieval-results-okf.checkpoint.json
```

When an OKF method is included, comparison eligibility and OKF page rankings
automatically use only retrieval-ready source pages: a page must have generated
atomic facts or retrieval questions in `source_evidence`. Failed enrichment
pages therefore cannot become gold pages or consume top-result positions. The
current bundle exposes 171 retrieval-ready pages and 2,377 eligible questions.

The companion `*-judge-actions.json` file contains candidates, visited files,
citations, evidence decisions, stop reasons, ranked page IDs, and Azure query
counts. Diagnose wrong routing, rejected evidence, or exhausted budgets from
these traces before increasing limits further. The synthetic benchmark still
favours retrieval of the source chunk from which each question was generated;
page recall should therefore be interpreted separately from answer correctness
and evidence support.

Search defaults to top 10 final results. Override per query:

```bash
uv run python -m src.rag.search "lake picnic" --limit 5
```

Experimental retrieval eval over scraped Markdown pages:

```bash
just eval-chunks
just eval-index qwen
just eval-generate 10
just eval qwen,sparse
just eval qwen4b_rerank,qwen4b_hybrid,qwen4b_hybrid_rerank
```

## Shape

```text
data/           tracked seed fixture
sql/            one-table schema, portable to SQLite and Postgres
src/db/         SQLite initialize and place queries
src/preprocess/ rebuild derived data from SQL rows
src/indexing/   provider-shaped vector indexing
src/vector_store/  Chroma collection, upsert, vector search
src/retrieval/  chunk retrieval methods and catalog
src/rag/        place search and answer scripts
src/eval/       chunked raw-page retrieval evaluation
src/okf/        complete-page OKF generation, validation, and navigation
src/scraping/   crawler, transform, ingest, scraping UI
src/shared/     schema, embeddings, env, LLM helper
tests/          data contract, retrieval, scrape transform
```

Durable reference databases live under `data/db/` with descriptive names such
as `pages.db`. Regenerable vector cache artifacts live under
`data/cache/chroma/`. SQLite page chunks are the source of truth for chunk text;
Chroma collections are derived indexes over those chunks. Use `.env` overrides
for private scratch paths under `.local/`.

---

## Roadmap

This repo focuses on building a clean, portable place database as the foundation
for a larger RAG system.

**Current:** SQLite + Chroma, everything local, no services needed.

**Next step:** swap storage backends for larger shared runs:
- SQLite → **Supabase** (free tier, under 500 MB, shared across teams)
- Chroma → **Qdrant** (free tier, managed vector index)

The SQL schema and module boundaries are designed to make that swap small.
Before scaling up, we need to align with other teams on what data collection
tools and shared infrastructure are available.

## Docs

- [docs/architecture-decisions.md](docs/architecture-decisions.md): durable decisions and why they matter.
- [src/okf/README.md](src/okf/README.md): complete OKF generation and navigation workflow.
- [experiments/indexing/README.md](experiments/indexing/README.md): retrieval experiments and fair OKF-versus-RAG protocol.
- [docs/retrieval-results-agentic.md](docs/retrieval-results-agentic.md): current agentic retrieval report.
