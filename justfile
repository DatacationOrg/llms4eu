set dotenv-load
set windows-shell := ["pwsh", "-NoLogo", "-Command"]

# Show the pipeline, in the order you run it.
default:
    @just --list --unsorted

# Fetch source URLs and append the pages to $LLMS4EU_DATA/scraped/<source>.jsonl.
fetch-pages SOURCE="data/brestanica.json":
    uv run python -m src.scraping.fetch_pages {{SOURCE}}

# Indexing, retrieval and eval work on one chunk size, CHUNK_SIZE (see src/db/README.md).

# 0. Turn the scraped pages.jsonl into wikipages.parquet (keeps existing summaries).
pages:
    uv run python -m src.preprocess.pages

# 1. Cut the pages into chunks.parquet at every size. On a rechunk, unchanged chunks
# keep their vectors and roles.
chunk:
    uv run python -m src.preprocess.chunker

# 2. Embed the chunks of every size with one provider; only missing rows. Resumable.
# Nemotron also runs on OpenRouter's free endpoint: just index nemotron --api
index METHOD="qwen" *ARGS:
    uv run python -m src.indexing --method {{METHOD}} {{ARGS}}

# 3. Score retrieval methods on the wiki QA test questions. Takes any evaluate flag:
# just eval --methods all --limit 500
eval *ARGS="--methods qwen":
    uv run python -m src.eval.evaluate {{ARGS}}

# The whole catalog, resumable: finished methods are skipped on a re-run.
eval-all:
    uv run python -m src.eval.evaluate --methods all --checkpoint

# Score the same methods on every chunk size, in one table. Resumable.
chunk-compare *ARGS="--methods qwen":
    uv run python -m src.eval.compare_chunkings {{ARGS}}

test:
    uv run --extra dev pytest

# Lint and format, the same checks CI would run.
lint:
    uv run ruff check --fix .
    uv run ruff format .
