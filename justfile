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

# Rechunk every page with the current chunk config and move the eval labels onto
# the new chunks. Needs an evidence quote per question (just eval-evidence).
rechunk:
    uv run python -m src.preprocess.chunker --rebuild
    uv run python -m src.eval.evidence relabel

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

# Wikipedia places corpus, kept apart under /data/llms4eu/wiki.
wiki := "LLMS4EU_DATA=/data/llms4eu/wiki"

# List EU castles, parks, caves, …, each with its article in the local language.
wiki-collect:
    {{wiki}} uv run python -m src.data_prep.wiki_places collect

# Background fetch (safe to disconnect); re-run to resume, a lock stops doubles.
wiki-fetch:
    {{wiki}} setsid nohup flock -n /data/llms4eu/wiki/fetch.lock nice -n 19 \
      uv run python -m src.scraping.fetch_pages /data/llms4eu/wiki/urls.jsonl \
      --skip-done >> /data/llms4eu/wiki/fetch.log 2>&1 &

# How far the background fetch got, per language.
wiki-status:
    @{{wiki}} uv run python -m src.data_prep.wiki_places status

# Fetched pages with their metadata, as /data/llms4eu/wiki/pages.jsonl.
wiki-export:
    {{wiki}} uv run python -m src.data_prep.wiki_places export

test:
    uv run --extra dev pytest

# Lint and format, the same checks CI would run.
lint:
    uv run ruff check --fix .
    uv run ruff format .
