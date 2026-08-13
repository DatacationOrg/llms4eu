set dotenv-load
set windows-shell := ["pwsh", "-NoLogo", "-Command"]

init:
    uv run python -m src.db.initialize

index:
    uv run python -m src.preprocess.index

ask *question:
    uv run python -m src.rag.answer "{{question}}"

scrape *URLS:
    uv run python -m src.scraping.scrape --urls {{URLS}}

scrape-web:
    uv run python -m src.scraping.web.main

fetch-pages SOURCE="data/brestanica.json" DB="data/db/pages.db":
    uv run python -m src.scraping.fetch_pages {{SOURCE}} --db {{DB}} --workers 4

test:
    uv run --extra dev pytest

eval-chunks:
    uv run python -m src.preprocess.chunks

eval-index METHOD="qwen" VERSION="v1":
    uv run python -m src.indexing.chunks --method {{METHOD}} --chunk-version {{VERSION}}

eval-phase2 METHODS="phase2-nemotron" OUTPUT="docs/retrieval-results-phase2.md":
    uv run python experiments/indexing/compare_qwen_modes.py --methods {{METHODS}} --output {{OUTPUT}}

eval-generate LIMIT="10":
    uv run python -m src.eval.generate_dataset --limit {{LIMIT}}

geocode-pages *FLAGS:
    uv run python -m src.eval.geocode_pages {{FLAGS}}

eval METHODS="qwen":
    uv run python -m src.eval.evaluate --methods {{METHODS}}

eval-agentic:
    uv run python -m src.eval.evaluate --agentic-only

eval-agentic-limit LIMIT="100":
    uv run python -m src.eval.evaluate --agentic-only --limit {{LIMIT}}

eval-agentic-report OUTPUT="docs/retrieval-results.md":
    uv run python experiments/indexing/compare_qwen_modes.py --methods all-agentic --output {{OUTPUT}}

eval-equivalence CHECKPOINT="docs/retrieval-results-chunks-okf.md.checkpoint.json" OUTPUT="docs/retrieval-equivalence-judge.md" K="5":
    uv run python experiments/indexing/judge_retrieval_equivalence.py --checkpoint {{CHECKPOINT}} --output {{OUTPUT}} -k {{K}}

eval-inspect LIMIT="20":
    uv run python -m src.eval.inspect_dataset --limit {{LIMIT}}

okf-pilot SOURCE="castle_rajhenburg" LIMIT="2":
    uv run python -m src.okf.generate --source {{SOURCE}} --limit {{LIMIT}}

okf-generate:
    uv run python -m src.okf.generate

okf-rebuild:
    uv run python -m src.okf.generate --clean

okf-index:
    uv run python -c "from pathlib import Path; from src.okf.bundle import regenerate_indexes; regenerate_indexes(Path('data/okf/tourism'))"

okf-validate:
    uv run python -m src.okf.validate

okf-ask *question:
    uv run python -m src.okf.answer "{{question}}"

okf-benchmark LIMIT="19":
    uv run python experiments/indexing/compare_okf_rag.py --limit {{LIMIT}}
