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

# --- scraper arena (research/scrapers) ---
# Extractor libraries are layered on with `uv run --with` so the project env and
# uv.lock stay untouched: scrapling pins playwright, which would otherwise fight
# the pins chromadb/sentence-transformers/docling already impose.
arena-env := "--with resiliparse --with readability-lxml --with goose3 --with html-text --with scrapling --with markdownify"

# Fetch the 100-page corpus once: raw HTML, rendered DOM, wikitext.
arena-snapshot:
    uv run python -m research.scrapers.snapshot

# Re-render only the pages still missing a rendered DOM.
arena-render:
    uv run python -m research.scrapers.snapshot --render-only

# Report any page missing a raw snapshot, rendered DOM or wikitext.
arena-verify:
    uv run python -m research.scrapers.snapshot --verify

# Run every HTML entrant over every page, timed (warmup + median of 3).
arena-extract *ARGS:
    uv run {{arena-env}} python -m research.scrapers.run_extractors {{ ARGS }}

# Re-measure extraction speed on one pinned CPU; run it on an idle machine.
arena-time:
    uv run {{arena-env}} python -m research.scrapers.run_extractors --timing-only

# Wikiextractor-V2 in its own Python 3.10 (it only imports there: inline regex flags).
arena-wiki:
    uv run --python 3.10 --no-project --with pylatexenc --with babel \
        --with beautifulsoup4 --with pyyaml --with python-dotenv \
        python research/scrapers/wiki_pass.py

# Compare fetch layers (ours vs Scrapy vs Scrapling) over the same URLs.
arena-fetchers:
    uv run --no-project --with 'scrapling[fetchers]' --with scrapy \
        --with pyyaml --with python-dotenv --with httpx --with lxml \
        python research/scrapers/fetch_layer.py

# Serve the labelling arena. Offline-safe: no CDN, every asset off local disk.
arena PORT="8090":
    uv run python -m research.scrapers.arena.app --port {{PORT}}

# Live statistics on a second port, read-only: never hands out matchups.
arena-stats PORT="8091":
    uv run python -m research.scrapers.arena.app --port {{PORT}} --stats-only

# Dump the results page to .local/scraper-arena/report.html
arena-report:
    uv run python -m research.scrapers.report

# Delete this reviewer's votes but keep snapshots, extractor runs and the LLM pool.
arena-reset-votes:
    uv run python -c "from research.scrapers import store; store.reset_votes('human')"

# Delete the automated pool only.
arena-reset-llm-votes:
    uv run python -c "from research.scrapers import store; store.reset_votes('llm')"

# Write blinded briefs for automated judging (see the scraper-arena-judge skill).
arena-judge-prepare COUNT="12":
    uv run python -m research.scrapers.judge prepare --count {{COUNT}}

# Rewrite unjudged briefs from current output (after an extractor adapter fix).
arena-judge-refresh:
    uv run python -m research.scrapers.judge refresh

# Vote counts per judge pool.
arena-judge-status:
    uv run python -m research.scrapers.judge status

# Human-vs-panel agreement on the comparisons both pools judged.
arena-agreement:
    uv run python -m research.scrapers.judge agreement

# One pair's head-to-head record, split by index-like vs article pages.
arena-duel A="ours" B="trafilatura":
    uv run python -m research.scrapers.judge duel --a '{{ A }}' --b '{{ B }}'

# Standings for the ours / trafilatura / wikiextractor-v2 set on their 47 shared pages.
arena-wiki3:
    uv run python -m research.scrapers.judge focus --entrants ours,trafilatura,wikiextractor-v2

# Standings for any closed set of two or more entrants, comma-separated.
arena-focus ENTRANTS="ours,trafilatura,wikiextractor-v2":
    uv run python -m research.scrapers.judge focus --entrants '{{ ENTRANTS }}'

# How different are the tools before anyone votes: char means plus the
# word-token-cosine and byte-exact similarity matrices, for both layers.
arena-similarity:
    uv run {{arena-env}} python -m research.scrapers.similarity research/scrapers/deck_data.json

# Rebuild the self-contained presentation from deck.src.html + figures/.
arena-deck:
    uv run python -m research.scrapers.build_deck
