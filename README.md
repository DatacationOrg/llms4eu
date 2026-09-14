# LLMs4EU - Tourism - RAG

Local retrieval research over Slovenian tourism pages. SQLite stores canonical
content, Chroma stores derived vector indexes, and everything runs inside the
Python environment.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [just](https://just.systems/)
- [Ollama](https://ollama.com/) — needed for `just eval-generate` and for the
  agentic retrieval methods

## Setup

```bash
uv sync --extra dev   # install all dependencies
cp .env.example .env  # set LLMS4EU_DATA, the shared artifact store
```

All generated data lives outside the repo in one place, `LLMS4EU_DATA`
(default `/data/llms4eu`): the page database, the Chroma vector cache,
embedding caches, benchmark checkpoints and judge caches. The repo tracks
code, SQL schema and the source URL list only.

```bash
ollama pull gemma4:e4b
# Optional for eval question generation:
ollama pull gemma4:26b-a4b-it-q4_K_M
ollama pull gpt-oss:20b
```

## Usage

```bash
just fetch-pages   # fetch the Slovenian source URLs into the page database
just chunk         # split fetched Markdown into heading-aware page chunks
just index qwen    # embed those chunks into a Chroma collection
just test          # run the non-LLM test suite
```

Retrieval evaluation over the scraped pages:

```bash
just eval-generate 10        # generate labelled questions
just eval qwen,sparse        # compare named retrieval methods
just eval-report             # full benchmark report into docs/
just eval-inspect            # look at the labelled dataset
```

## Shape

```text
data/           brestanica.json, the tracked Slovenian source URLs
sql/            page and eval schema, portable to SQLite and Postgres
src/scraping/   fetch pages, extract Markdown, store in SQLite
src/db/         page-database connection and schema helpers
src/preprocess/ heading-aware page chunking
src/indexing/   provider-shaped vector indexing
src/vector_store/  Chroma collection, upsert, vector search
src/retrieval/  chunk retrieval methods and catalog
src/eval/       retrieval evaluation over labelled questions
src/shared/     schema, embeddings, env, LLM helper
experiments/    benchmark orchestration over the method catalog
tests/          schema contract, retrieval, extraction
```

Durable and regenerable artifacts alike live under `LLMS4EU_DATA`
(`db/pages.db`, `chroma/`, `embeddings/`, `checkpoints/`, `judge-cache/`). SQLite page chunks are the source of truth for chunk text;
Chroma collections are derived indexes over those chunks.

---

## Roadmap

This repo focuses on a clean, portable page corpus and a measured retrieval
pipeline over it.

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
- [docs/](docs/): retrieval, OKF and agentic result reports, including the
  findings from experiments whose code has since been removed.
