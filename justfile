set dotenv-load
set windows-shell := ["pwsh", "-NoLogo", "-Command"]

# Fetch the Slovenian source URLs into the page database.
fetch-pages SOURCE="data/brestanica.json":
    uv run python -m src.scraping.fetch_pages {{SOURCE}} --workers 4

# Rebuild page chunks from the fetched Markdown.
chunk:
    uv run python -m src.preprocess.chunks

# Rebuild one provider's chunk vector index.
index METHOD="qwen":
    uv run python -m src.indexing.chunks --method {{METHOD}}

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

# Full benchmark report over the method catalog.
eval-report METHODS="all" OUTPUT="docs/retrieval-results.md":
    uv run python experiments/indexing/compare_qwen_modes.py --methods {{METHODS}} --output {{OUTPUT}}

eval-equivalence OUTPUT="docs/retrieval-equivalence-judge.md" K="5":
    uv run python experiments/indexing/judge_retrieval_equivalence.py --output {{OUTPUT}} -k {{K}}
