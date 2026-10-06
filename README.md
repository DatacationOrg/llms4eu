# LLMs4EU - Tourism - RAG

Local retrieval research over Slovenian tourism pages. SQLite stores canonical
content, Chroma stores derived vector indexes, and everything runs inside the
Python environment.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [just](https://just.systems/)
- [Ollama](https://ollama.com/) — needed for `just eval-generate`

## Setup

```bash
uv sync --extra dev   # install all dependencies
cp .env.example .env  # set LLMS4EU_DATA, the shared artifact store
```

All generated data lives outside the repo in one place, `LLMS4EU_DATA`
(default `/data/llms4eu`): the page database, the Chroma vector cache,
embedding caches and benchmark checkpoints. The repo tracks
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
just locate-pages  # Wikidata point per page, for the *_geo methods
just index qwen    # embed those chunks into a Chroma collection
just rechunk       # after changing chunk_size/overlap: rechunk, move labels
just test          # run the non-LLM test suite
```

Retrieval evaluation over the scraped pages:

```bash
just eval-generate 10                 # generate labelled questions
just eval-evidence                    # anchor answers to quotes (before rechunk)
just eval --methods qwen,sparse       # compare named retrieval methods
just eval-all                         # whole catalog, resumable
just eval-inspect                     # look at the labelled dataset
```

## Shape

```text
data/           brestanica.json, the tracked Slovenian source URLs
sql/            page and eval schema, portable to SQLite and Postgres
src/scraping/   fetch pages, extract Markdown, store in SQLite
src/db/         the wiki places dataset (wiki_qa.py, schemas/); legacy/ the old page DB
src/preprocess/ heading-aware page chunking, page locations
src/indexing/   embedding providers, embedding cache, Chroma collections
src/retrieval/  chunk retrieval methods and catalog
src/eval/       retrieval evaluation over labelled questions
src/shared/     env and artifact paths, prompt loading, LLM helper
prompts/        LLM prompt templates, loaded by src.shared.prompts
tests/          schema contract, retrieval, extraction
```

Durable and regenerable artifacts alike live under `LLMS4EU_DATA`
(`db/pages.db`, `chroma/`, `embeddings/`, `checkpoints/`). SQLite page chunks
are the source of truth for chunk text;
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
- [docs/README.md](docs/README.md): index of the dated research reports, including
  experiments whose code has since been removed. Start with
  [docs/reports/all-runs-unified-2026-09-14.md](docs/reports/all-runs-unified-2026-09-14.md).
