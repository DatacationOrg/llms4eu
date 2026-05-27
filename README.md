# LLMs4EU - Tourism - RAG

Local RAG for tourism places. SQLite is the source of truth,
Chroma is the vector index. Both run entirely inside the Python environment.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [just](https://just.systems/)
- [Ollama](https://ollama.com/) — only needed for `just ask` and `just scrape`

## Setup

```bash
uv sync --extra dev   # install all dependencies
cp .env.example .env  # set local paths (defaults work out of the box)
```

```bash
ollama pull gemma4:e4b
```

## Usage

```bash
just init        # load seed data into SQLite
just index       # embed places and rebuild Chroma
just ask What place is best for a quiet forest walk near water?  # retrieve + LLM answer
just scrape https://example.com  # crawl a site, ingest into SQLite, reindex
just scrape-web  # start the local scraping UI
just test        # run the non-LLM test suite
```

Search defaults to top 3 results. Override per query:

```bash
uv run python -m src.rag.search "lake picnic" --limit 5
```

Experimental retrieval eval over scraped Markdown pages:

```bash
just eval-chunks
just eval-index
just eval-generate 10
just eval vector,bm25,vector_bm25
just eval bm25,vector_bm25,qwen3_rerank_hybrid
```

## Shape

```text
data/           tracked seed fixture
sql/            one-table schema, portable to SQLite and Postgres
src/db/         SQLite initialize and place queries
src/preprocess/ rebuild Chroma from SQL rows
src/vector_db/  Chroma collection, upsert, vector search
src/rag/        search and answer scripts
src/eval/       chunked raw-page retrieval evaluation
src/scraping/   crawler, transform, ingest, scraping UI
src/shared/     schema, embeddings, env, LLM helper
tests/          data contract, retrieval, scrape transform
```

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
