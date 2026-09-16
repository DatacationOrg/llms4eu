set dotenv-load
set windows-shell := ["pwsh", "-NoLogo", "-Command"]

# Show the pipeline, in the order you run it.
default:
    @just --list --unsorted

# 1. Fetch the Slovenian source URLs into the page database.
fetch-pages SOURCE="data/brestanica.json":
    uv run python -m src.scraping.fetch_pages {{SOURCE}}

# 2. Split the fetched Markdown into chunks.
chunk:
    uv run python -m src.preprocess.chunker

# 3. Embed the chunks into one provider's Chroma collection.
index METHOD="qwen":
    uv run python -m src.indexing --method {{METHOD}}

# 4. Generate eval questions from unlabelled chunks.
eval-generate LIMIT="10":
    uv run python -m src.eval.generate_dataset --limit {{LIMIT}}

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
