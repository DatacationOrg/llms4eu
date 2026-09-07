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

# Copy the durable page database to a private scratch DB for chunk-variant work.
# Variant chunking rewrites page_chunks, whose delete cascade would take the
# approved eval labels with it, so experiments never run against data/db.
chunk-sweep-db SOURCE="data/db/pages.db" TARGET=".local/db/pages-variants.db":
    uv run python -m src.db.snapshot {{SOURCE}} {{TARGET}}

eval-index METHOD="qwen" VERSION="v1" VARIANT="base":
    uv run python -m src.indexing.chunks --method {{METHOD}} --chunk-version {{VERSION}} --variant {{VARIANT}}

# Detect each page's language. Read-only; pass --apply to write.
detect-languages *ARGS:
    uv run python -m src.preprocess.languages {{ARGS}}

# Cut one chunk variant. Needs PAGES_DB_PATH pointed at a `just chunk-sweep-db` copy.
chunk-variant VARIANT SIZE="512" OVERLAP="0" UNIT="tokens" PROVIDER="qwen" STRATEGY="markdown" EXTRA="":
    uv run python -m src.preprocess.chunks --variant {{VARIANT}} --strategy {{STRATEGY}} \
        --size {{SIZE}} --overlap {{OVERLAP}} --unit {{UNIT}} --provider {{PROVIDER}} {{EXTRA}}

# Point the approved questions at one variant's chunks.
relabel VARIANT *ARGS:
    uv run python -m src.eval.relabel --variant {{VARIANT}} {{ARGS}}

chunk-sweep *ARGS:
    uv run python experiments/indexing/compare_chunkings.py {{ARGS}}

# Snapshot for the shared-question design, where one question set is projected
# onto every variant. Kept apart from the per-variant DB because both designs
# write into eval_relevant_chunks and a mixed database scores rows that are not
# comparable; `compare_chunkings.py --design` refuses to run against a mismatch.
chunk-sweep-shared-db SOURCE="data/db/pages.db" TARGET=".local/db/pages-shared.db":
    uv run python -m src.db.snapshot {{SOURCE}} {{TARGET}}

# Anchor each answer to a verbatim quote's character span. One model pass for the
# whole benchmark however many variants follow, and the prerequisite for both the
# shared design's labels and the `anchor` span target. Resumable: already-anchored
# questions are skipped.
anchor-answers *ARGS:
    uv run python -m src.eval.anchors {{ARGS}}

# The comprehensive variant grid: span and chunk metrics, latency, category split,
# and optionally judge_hit@K and agentic diagnostics. Always pass --dry-run first;
# a reranked cell costs roughly 0.00132s per median chunk token per question, and
# an agentic cell 5.1s per question.
#
# This runs against whatever PAGES_DB_PATH and CHROMA_PATH are set to, which for
# the durable store means base only. Use `sweep-overview` for the variant grid.
chunk-overview *ARGS:
    uv run python experiments/indexing/compare_chunkings.py {{ARGS}}

# `chunk-overview` against the variant store, so the five cuttings are visible.
# Same reason as prepare_variants.sh and sweep-status: `set dotenv-load` has
# already exported the durable CHROMA_PATH, and pointing a variant grid at it
# would silently find only base and prune every other cell.
sweep-overview *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    export CHROMA_PATH="${SWEEP_CHROMA_PATH:-.local/chroma-sweep}"
    export PAGES_DB_PATH="${SWEEP_PAGES_DB_PATH:-.local/db/pages-variants.db}"
    echo "store: $CHROMA_PATH | db: $PAGES_DB_PATH"
    uv run python experiments/indexing/compare_chunkings.py {{ARGS}}

# The same grid against the shared-question database, where one question set is
# projected onto every variant. Needs `anchor-answers` and `relabel` to have run.
shared-overview *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    export CHROMA_PATH="${SWEEP_CHROMA_PATH:-.local/chroma-sweep}"
    export PAGES_DB_PATH="${SHARED_PAGES_DB_PATH:-.local/db/pages-shared.db}"
    echo "store: $CHROMA_PATH | db: $PAGES_DB_PATH"
    uv run python experiments/indexing/compare_chunkings.py --design shared {{ARGS}}

audit-chunk-tokens PROVIDERS="qwen,qwen4b,nemotron" VERSIONS="v1,v2" OUTPUT="docs/chunk-token-audit-$(date +%F).md":
    uv run python experiments/indexing/audit_chunk_tokens.py --providers {{PROVIDERS}} --versions {{VERSIONS}} --output {{OUTPUT}}

eval-phase2 METHODS="phase2-nemotron" OUTPUT="docs/retrieval-results-phase2.md":
    uv run python experiments/indexing/compare_qwen_modes.py --methods {{METHODS}} --output {{OUTPUT}}

eval-generate LIMIT="10":
    uv run python -m src.eval.generate_dataset --limit {{LIMIT}}

# Give one chunk variant its own question set, generated from its own chunks, so
# it is not measured on questions written from base's cutting. Point
# PAGES_DB_PATH at the sweep copy first — this writes questions and gold labels.
#
# Density-normalised: one question per 256 of a chunk's own tokens, so 256 gets
# one, 512 two and 1,024 four and every variant covers the corpus with the same
# number of questions. Sweep the result as `--design per-variant-density`. Pass
# DENSITY="" for the legacy one-question-per-type sets, which probed the smallest
# cut four times more densely than the largest (`--design per-variant`).
variant-questions VARIANT WORKERS="8" MINCHARS="300" DENSITY="--density":
    uv run python -m src.eval.generate_dataset \
        --variant {{VARIANT}} --workers {{WORKERS}} --min-chars {{MINCHARS}} \
        {{DENSITY}}

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

# Relabel and index every chunk variant for the size sweep. Hours, resumable.
prepare-variants:
    bash experiments/indexing/prepare_variants.sh

# Index providers across variants in an existing sweep, without re-cutting chunks.
# `prepare-variants` re-cuts, and page_chunks cascades into eval_relevant_chunks,
# so using it to add a provider would delete the gold labels. One provider per
# process, so two 15 GB checkpoints are never resident together.
index-variants PROVIDERS="qwen4b qwen8b nemotron8b" VARIANTS="base tok256 tok512 tok512ov tok1024":
    PROVIDERS="{{PROVIDERS}}" VARIANTS="{{VARIANTS}}" \
        bash experiments/indexing/index_variants.sh

# Progress of the chunk-variant preparation: what is running, how far labelling
# has got, and which collections exist.
sweep-status LOG=".local/logs/prepare-variants.log":
    #!/usr/bin/env bash
    # Same reason as prepare_variants.sh: dotenv-load has already exported the
    # runtime CHROMA_PATH, so a `:-` default would report the wrong store.
    export CHROMA_PATH="${SWEEP_CHROMA_PATH:-.local/chroma-sweep}"
    export PAGES_DB_PATH="${SWEEP_PAGES_DB_PATH:-.local/db/pages-variants.db}"
    echo "=== running ==="
    # Match only the worker processes; a bare pattern also matches the shell
    # that is running this recipe, and any pgrep pattern matches itself.
    pgrep -af "(python|uv) .*(eval\.relabel|indexing\.chunks|preprocess\.chunks)" \
      | grep -vE "shell-snapshots|sweep.status" || echo "  nothing running"
    echo
    uv run python -m src.eval.sweep_status
    if [ -f "{{LOG}}" ]; then echo; echo "=== last log lines ==="; tail -5 "{{LOG}}"; fi

# The 2026-08-18 completion: both scale rungs of each embedder family plus one
# larger reranker rung, over the five cuttings. Four sequential legs, ~17.5h,
# resumable per cell. RERANK is the variants the 4B-reranker leg covers — the
# default two answer the question for 8h where all five cost 20h.
#
#   just large-embedder-sweep                          # as launched
#   just large-embedder-sweep "base,tok1024" "base,tok256,tok512,tok512ov,tok1024"
#
# Re-running is safe and cheap: each leg keeps its own checkpoint, so widening
# RERANK re-measures only the reranker leg and the other three resume instantly.
large-embedder-sweep RERANK="base,tok1024" VARIANTS="base,tok256,tok512,tok512ov,tok1024":
    RERANK_VARIANTS="{{RERANK}}" VARIANTS="{{VARIANTS}}" \
        bash experiments/indexing/run_large_embedder_sweep.sh

# Join every chunk sweep run into one table per metric. Parses the reports rather
# than retyping them, and refuses to publish if the runs scored different samples.
# Add a run by adding it to SOURCES and ORDER in the script, not by editing docs.
merge-chunk-sweeps:
    uv run python experiments/indexing/merge_chunk_sweeps.py
