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
ollama pull gpt-oss:20b
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

- [docs/README.md](docs/README.md): the docs index, what lives where.
- [docs/architecture/decisions.md](docs/architecture/decisions.md): durable decisions and why they matter.
- [docs/architecture/geo-retrieval.md](docs/architecture/geo-retrieval.md): how geo-aware retrieval finds its answer.
- [experiments/indexing/README.md](experiments/indexing/README.md): retrieval experiments and evaluation protocol.
- [docs/reports/agentic/agentic-findings.md](docs/reports/agentic/agentic-findings.md): what agentic retrieval was tried and what it measured.
- [docs/reports/chunking/sweeps/](docs/reports/chunking/sweeps/): chunk-variant sweeps; [the merged table](docs/reports/chunking/chunk-size-sweep-merged-2026-08-18.md) is the current head.

Result reports are regenerable and mostly untracked, so a link to one may point at
a file you have to produce. Anything dated before 2026-08-11 was measured at the
old 512-token sequence cap, where 22% of the corpus never reached the embedder;
see `docs/architecture/decisions.md` before comparing it with a new run.
