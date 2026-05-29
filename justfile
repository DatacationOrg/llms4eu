set dotenv-load
set windows-shell := ["pwsh", "-NoLogo", "-Command"]

scrape *URLS:
    uv run python -m src.scraping.scrape --urls {{URLS}}

fetch-pages SOURCE="data/brestanica.json" DB=".local/raw_pages.db":
    uv run python -m src.scraping.fetch_pages {{SOURCE}} --db {{DB}} --workers 4

init:
    uv run python -m src.db.initialize

scrape-web:
    uv run python -m src.scraping.web.main

index:
    uv run python -m src.preprocess.index

ask *question:
    uv run python -m src.rag.answer "{{question}}"

test:
    uv run --extra dev pytest

eval-chunks:
    uv run python -m src.preprocess.chunks

eval-index METHOD="qwen" CONTENT_MODE="chunk_summary":
    uv run python -m src.indexing.chunks --method {{METHOD}} --content-mode {{CONTENT_MODE}}

eval-generate LIMIT="10":
    uv run python -m src.eval.generate_dataset --limit {{LIMIT}}

eval-generate-model LIMIT MODEL:
    uv run python -m src.eval.generate_dataset --limit {{LIMIT}} --model {{MODEL}}

eval METHODS="qwen_chunk_summary":
    uv run python -m src.eval.evaluate --methods {{METHODS}}

eval-inspect LIMIT="20":
    uv run python -m src.eval.inspect_dataset --limit {{LIMIT}}

eval-import PATH:
    uv run python -m src.eval.import_questions {{PATH}}

eval-reset:
    uv run python -m src.eval.reset_dataset
