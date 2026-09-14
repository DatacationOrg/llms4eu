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
cp .env.example .env  # set LLMS4EU_DATA, the shared artifact store
```

All generated data lives outside the repo in one place, `LLMS4EU_DATA`
(default `/data/llms4eu`): the page database, the OKF bundle, the Chroma
vector cache, embedding caches, benchmark checkpoints and judge caches.
The repo tracks code, SQL schema and the small seed fixtures only.

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
data/           tracked seed fixtures (places.jsonl, brestanica.json)
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

Durable and regenerable artifacts alike live under `LLMS4EU_DATA`
(`db/pages.db`, `okf/tourism/`, `chroma/`, `embeddings/`, `checkpoints/`,
`judge-cache/`). SQLite page chunks are the source of truth for chunk text;
Chroma collections are derived indexes over those chunks.

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
- [experiments/indexing/README.md](experiments/indexing/README.md): retrieval experiments and evaluation protocol.
- [docs/](docs/): retrieval, OKF and agentic result reports.
