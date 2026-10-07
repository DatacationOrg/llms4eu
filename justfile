set dotenv-load
set windows-shell := ["pwsh", "-NoLogo", "-Command"]

# Show the pipeline, in the order you run it.
default:
    @just --list --unsorted

# 1. Fetch source URLs and append the pages to $LLMS4EU_DATA/scraped/<source>.jsonl.
fetch-pages SOURCE="data/brestanica.json":
    uv run python -m src.scraping.fetch_pages {{SOURCE}}

# Every chunk step, index and eval works on one chunk variant: `base`, or
# CHUNK_VARIANT=<name> from src/preprocess/config.yaml (CHUNK_VARIANT=c900 just chunk).

# 2. Split the fetched Markdown into chunks.
chunk:
    uv run python -m src.preprocess.chunker

# Rechunk every page with the current chunk config and move the eval labels onto
# the new chunks. Needs an evidence quote per question (just eval-evidence).
rechunk:
    uv run python -m src.preprocess.chunker --rebuild
    uv run python -m src.eval.evidence relabel

# Chunk, label, index and score every listed variant, in one table. Resumable.
chunk-compare VARIANTS="base,c900,c900ov,c3600" *ARGS="--methods qwen":
    uv run python -m src.eval.compare_chunkings --variants {{VARIANTS}} {{ARGS}}

# Look up where each page is about on Wikidata, for the *_geo methods.
locate-pages:
    uv run python -m src.preprocess.locations

# 3. Embed the chunks into one provider's Chroma collection.
index METHOD="qwen":
    uv run python -m src.indexing --method {{METHOD}}

# 4. Generate eval questions from unlabelled chunks.
eval-generate LIMIT="10":
    uv run python -m src.eval.generate_dataset --limit {{LIMIT}}

# Anchor each question's answer to a verbatim quote, so its label survives a rechunk.
eval-evidence *ARGS:
    uv run python -m src.eval.evidence backfill {{ARGS}}

# 5. Score retrieval methods. Takes any evaluate flag: just eval --methods all --limit 50
eval *ARGS="--methods qwen":
    uv run python -m src.eval.evaluate {{ARGS}}

# The whole catalog, resumable: finished methods are skipped on a re-run.
eval-all:
    uv run python -m src.eval.evaluate --methods all --checkpoint

# Read generated questions back out of the database.
eval-inspect LIMIT="20":
    uv run python -m src.eval.inspect_dataset --limit {{LIMIT}}

test:
    uv run --extra dev pytest

# Lint and format, the same checks CI would run.
lint:
    uv run ruff check --fix .
    uv run ruff format .
