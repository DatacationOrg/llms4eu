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
