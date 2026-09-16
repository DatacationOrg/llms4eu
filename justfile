set dotenv-load
set windows-shell := ["pwsh", "-NoLogo", "-Command"]

# Fetch the Slovenian source URLs into the page database.
fetch-pages SOURCE="data/brestanica.json":
    uv run python -m src.scraping.fetch_pages {{SOURCE}} --workers 4

# Rebuild page chunks from the fetched Markdown.
chunk:
    uv run python -m src.preprocess.chunker

# Rebuild one provider's chunk vector index.
index METHOD="qwen":
    uv run python -m src.indexing --method {{METHOD}}

test:
    uv run --extra dev pytest

eval METHODS="qwen":
    uv run python -m src.eval.evaluate --methods {{METHODS}}

eval-limit LIMIT="100":
    uv run python -m src.eval.evaluate --limit {{LIMIT}}

eval-generate LIMIT="10":
    uv run python -m src.eval.generate_dataset --limit {{LIMIT}}

eval-inspect LIMIT="20":
    uv run python -m src.eval.inspect_dataset --limit {{LIMIT}}

# Long run over the whole catalog; --checkpoint skips finished methods on resume.
eval-all:
    uv run python -m src.eval.evaluate --methods all --checkpoint
