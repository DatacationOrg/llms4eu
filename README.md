# LLMs4EU - Tourism - RAG

Local retrieval research over Wikipedia pages of EU places (castles, lakes, caves,
...; 24 languages). The data is Parquet files with one pydantic model each, read
through `src/db/dataset.py`; embeddings are stored next to them as `.npy`.
Everything runs inside the Python environment.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [just](https://just.systems/)
- [Ollama](https://ollama.com/) — needed for the geo retriever's place lookup

## Setup

```bash
uv sync --extra dev   # install all dependencies
cp .env.example .env  # set LLMS4EU_DATA, the shared artifact store
```

The dataset lives in `DATASET_DIR` (default `/data/llms4eu/wiki`, see
`src/db/README.md`). Caches, checkpoints and scraped pages live in `LLMS4EU_DATA`
(default `/data/llms4eu`). The repo tracks code only.

## Usage

```bash
just chunk                          # cut the pages into chunks.parquet, all sizes
just index qwen                     # embed every chunk size (GPU)
just index nemotron --api           # the same on OpenRouter's free endpoint
just eval --methods qwen,sparse     # compare named retrieval methods
CHUNK_SIZE=<size> just eval         # on another chunk size
just chunk-compare --methods qwen   # every chunk size in one table
just eval-all                       # whole catalog, resumable
just fetch-pages                    # scrape source URLs into a JSONL of pages
just test                           # run the non-LLM test suite
```

## Shape

```text
data/           brestanica.json, tracked source URLs
src/db/         the dataset: dataset.py loads it, schemas/ one model per file
src/scraping/   fetch pages, extract Markdown, append to a JSONL
src/preprocess/ heading-aware chunking into chunks.parquet
src/indexing/   embedding providers, embeddings/<provider>/<size>.npy, vector search
src/retrieval/  chunk retrieval methods and catalog
src/eval/       retrieval evaluation on the wiki QA questions
src/shared/     env and artifact paths, prompt loading, LLM helper
datagen/        how generated columns were made; not used by the pipeline
prompts/        LLM prompt templates, loaded by src.shared.prompts
tests/          data contract, chunking, retrieval, extraction
```

The first Slovenian corpus (an SQLite page database) is not in this format; a
later step could convert it.

---

## Roadmap

**Current:** Parquet + `.npy` vectors, exact vector search, everything local.

**Next step:** an approximate vector index (FAISS or Qdrant) built from the stored
vectors, for fast queries over the larger chunk sizes and more models.

## Docs

- [docs/architecture-decisions.md](docs/architecture-decisions.md): durable decisions and why they matter.
- [docs/README.md](docs/README.md): index of the dated research reports, including
  experiments whose code has since been removed. Start with
  [docs/reports/all-runs-unified-2026-09-14.md](docs/reports/all-runs-unified-2026-09-14.md).
