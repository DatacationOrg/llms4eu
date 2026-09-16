#!/usr/bin/env bash
# Build everything a chunk-size sweep needs: cut each variant, point the approved
# questions at it, then index it.
#
# Four stages, each resumable, so re-running after an interruption continues
# rather than starting over:
#   1. snapshot  - copy the durable page DB, once
#   2. chunk     - cut each variant (append-only; RECHUNK=1 to re-cut)
#   3. questions - generate each variant's own questions from its own chunks, so
#                  no variant is measured on questions written from another
#                  cutting (~3h across four variants at 8 workers)
#   4. index     - one Chroma collection per variant and provider
#
# Answer anchoring (`src/eval/anchors.py`) is deliberately not here. The span
# metrics default to the base chunk each shared question was generated from,
# which is ground truth by construction and costs no model time; an anchor only
# narrows that target from a paragraph to a sentence.
#
# Nothing here touches data/db/pages.db. Re-chunking cascades deletes into
# eval_relevant_chunks, so all work happens against a snapshot under .local/.
set -euo pipefail

# Deliberately NOT `${CHROMA_PATH:-...}`. The justfile sets dotenv-load, so .env's
# CHROMA_PATH is already exported and a `:-` default would never apply — the sweep
# would write variant collections into the production store and, worse, rebuilding
# `base`/`qwen` deletes and recreates `page_chunks_qwen_chunk` there. Override with
# SWEEP_CHROMA_PATH / SWEEP_PAGES_DB_PATH, never with the runtime variables.
export CHROMA_PATH="${SWEEP_CHROMA_PATH:-.local/chroma-sweep}"
export PAGES_DB_PATH="${SWEEP_PAGES_DB_PATH:-.local/db/pages-variants.db}"

# name:size:overlap, sized in the embedder's own tokens rather than characters so
# the budget means the same thing in every language. 256 is the smallest size
# MiniLM can read; 512 matches the historical cap; tok512ov adds the conventional
# 15% overlap at that size; 1024 is roughly the p99 of `base`.
VARIANT_SPECS="${VARIANT_SPECS:-tok256:256:0 tok512:512:0 tok512ov:512:77 tok1024:1024:0}"

# Providers read their model's full context, so no chunk size overflows any of
# them and every variant gets the same set. `base` also gets the legacy 512
# provider: same chunks, same labels, so the difference against `qwen` measures
# exactly what the old cap was costing.
providers_for() {
  case "$1" in
    base) echo "qwen nemotron qwen_s512" ;;
    *)    echo "qwen nemotron" ;;
  esac
}

LOG="${LOG:-.local/logs/prepare-variants.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "--- run started $(date -Is) ---"
echo "pages db : $PAGES_DB_PATH"
echo "chroma   : $CHROMA_PATH"
echo "variants : $VARIANT_SPECS"

if [ "$PAGES_DB_PATH" = "data/db/pages.db" ]; then
  echo "!! refusing to run against the durable database" >&2
  exit 1
fi

# Indexing deletes and recreates a collection, so sharing a store with the running
# system would destroy its indexes. Compare canonical paths: the .env value may be
# quoted, relative, or spelled differently while naming the same directory.
canonical() {
  local value="$1"
  value="${value%\"}"; value="${value#\"}"
  value="${value%\'}"; value="${value#\'}"
  realpath -m -- "$value" 2>/dev/null || printf '%s' "$value"
}
runtime_chroma="$(
  grep -sE '^[[:space:]]*CHROMA_PATH[[:space:]]*=' .env |
    tail -1 | cut -d= -f2- | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
)"
if [ -n "$runtime_chroma" ] &&
   [ "$(canonical "$CHROMA_PATH")" = "$(canonical "$runtime_chroma")" ]; then
  echo "!! CHROMA_PATH resolves to the runtime store from .env ($runtime_chroma)." >&2
  echo "   Indexing deletes and rebuilds collections there. Aborting." >&2
  exit 1
fi

echo "=== snapshot ==="
if [ -f "$PAGES_DB_PATH" ]; then
  echo "reusing existing snapshot"
else
  uv run python -m src.db.snapshot data/db/pages.db "$PAGES_DB_PATH"
fi

# Anchors are placed by searching the page, so `base` needs no spans for that.
# Its spans are backfilled anyway: `base` is a variant like any other in the
# sweep, and labelling any variant is an overlap test against chunk spans.
echo "=== backfill base spans ==="
uv run python -m src.preprocess.chunks --variant base --backfill-spans

# Append-only: a variant that already has chunks is left alone. Re-cutting needs
# RECHUNK=1, and the order matters — page_chunks cascades into
# eval_relevant_chunks, so re-cutting a variant that already has its own generated
# questions deletes their gold links and orphans the questions. Cut first, then
# generate.
for spec in $VARIANT_SPECS; do
  IFS=: read -r variant size overlap <<<"$spec"
  echo "=== chunk $variant (${size} tokens, ${overlap} overlap) ==="
  uv run python -m src.preprocess.chunks \
    --variant "$variant" --strategy markdown --unit tokens --provider qwen \
    --size "$size" --overlap "$overlap" \
    ${RECHUNK:+--clean}
done

# Each variant gets questions written from its own chunks. A question generated
# from a `base` chunk is answerable from a `base` chunk by construction, so sharing
# one question set flatters `base` and any cutting close to it; this is what
# removes that. Resumable — a chunk that already has questions is skipped.
#
# Density-normalised by default: a chunk is asked for one question per 256 of its
# own tokens, so 256 gets one, 512 two and 1,024 four, and every variant covers
# the corpus with the same number of questions. The legacy alternative
# (QUESTION_DESIGN=per-type) asks one question of every type of every chunk
# whatever its size, which probed tok256 four times more densely than tok1024 over
# the same pages — 7,801 questions against 1,451 in the 2026-08-18 sweep — and
# reached further down each chunk for facts to ask about, so the large cut got the
# more salient questions as well as the smaller sample. Set it only to reproduce a
# report published before 2026-08-20.
#
# The confound neither setting removes: a variant with more chunks is a harder
# haystack, and each variant's questions were written with its own boundaries in
# view. Anchoring these questions and relabelling them onto every variant
# (`src/eval/anchors.py`, `src/eval/relabel.py`) pools them into one shared set and
# removes both; density normalisation is what makes that pool balanced rather than
# dominated by whichever cut produced the most questions.
QUESTION_DESIGN="${QUESTION_DESIGN:-density}"
case "$QUESTION_DESIGN" in
  density)  question_flags=(--density) ; sweep_design="per-variant-density" ;;
  per-type) question_flags=()          ; sweep_design="per-variant" ;;
  *) echo "!! unknown QUESTION_DESIGN '$QUESTION_DESIGN' (density|per-type)" >&2; exit 1 ;;
esac
echo "question design : $QUESTION_DESIGN (sweep with --design $sweep_design)"

# Two passes. The batch call asks a chunk for its whole budget at once, which is
# cheap but fails more often the more it asks for: measured 49 of 49 requested
# questions for tok256 against 172 of 215 for tok1024. The second pass asks for
# the leftovers one at a time, and needs the same flags — without --density it
# offers every type for every chunk and restores the per-type design with the
# counts still looking right.
for spec in $VARIANT_SPECS; do
  variant="${spec%%:*}"
  echo "=== generate $variant questions ($QUESTION_DESIGN) ==="
  uv run python -m src.eval.generate_dataset \
    --variant "$variant" --workers 8 "${question_flags[@]}"
  echo "=== fill $variant gaps ($QUESTION_DESIGN) ==="
  uv run python -m src.eval.generate_dataset \
    --variant "$variant" --workers 8 --fill-missing "${question_flags[@]}"
done

for spec in $VARIANT_SPECS base; do
  variant="${spec%%:*}"
  for provider in $(providers_for "$variant"); do
    echo "=== index $variant / $provider ==="
    # A provider that cannot load (no local weights, no GPU) is reported and
    # skipped; the sweep then lists its cells as "index not built".
    uv run python -m src.indexing.chunks \
      --method "$provider" --variant "$variant" 2>&1 | tail -1 \
      || echo "!! skipped $variant / $provider"
  done
done

echo "=== done $(date -Is) ==="
uv run python -m src.eval.sweep_status
echo
echo "Sweep this with: --design $sweep_design"
echo "A grid run under the wrong --design is refused before its first cell, but"
echo "only because the design is declared; nothing detects it from one cell."
