#!/usr/bin/env bash
# The 2026-08-18 sweep completion: the larger rungs of both embedder families, plus
# one larger reranker rung, on the same five cuttings and the same per-variant
# question sets as the 08-14 and 08-17 runs.
#
# Four invocations rather than one, for two independent reasons.
#
# VRAM. The indexers memoize their loaded model and never evict, so a single
# process running all three embedders holds 8.1 + 15.1 + 15.9 = 39.1 GB of weights
# before the reranker loads beside them. That does not fit on a 48 GB A6000 with
# any headroom, and an OOM twelve hours in costs the whole run.
#
# Checkpoints. `_load_state` includes the method list in its signature, so
# splitting one method list across invocations to bound VRAM would discard the
# checkpoint each time and re-measure everything. Separate outputs keep each
# provider independently resumable; `merge_chunk_sweeps.py` unions their cell CSVs
# back into one table, which is what the long-form CSV exists for.
#
# Sequential, not parallel. There is 48 GB for the two smallest runs together, but
# the report publishes a Speed table in ms/query and two runs contending for one
# GPU would make every latency figure a measurement of the other run.
set -euo pipefail

export CHROMA_PATH="${SWEEP_CHROMA_PATH:-.local/chroma-sweep}"
export PAGES_DB_PATH="${SWEEP_PAGES_DB_PATH:-.local/db/pages-variants.db}"

VARIANTS="${VARIANTS:-base,tok256,tok512,tok512ov,tok1024}"
# The reranker rung runs on two columns, not five. `base` is the reference cutting
# every earlier report used and `tok1024` is the variant that won every aggregate
# metric in the merged report, so between them they answer both questions the rung
# exists for: does scaling the dominant stage help, and does the chunking ordering
# survive a better reranker. The middle three would cost roughly 12 hours to
# confirm a trend the two ends already show. Widen deliberately if that trend
# turns out to be non-monotonic.
RERANK_VARIANTS="${RERANK_VARIANTS:-base,tok1024}"
DATE="${DATE:-2026-08-18}"
DESIGN="${DESIGN:-per-variant}"

LOG="${LOG:-.local/logs/large-embedder-sweep.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "--- run started $(date -Is) ---"
echo "pages db : $PAGES_DB_PATH"
echo "chroma   : $CHROMA_PATH"
echo "variants : $VARIANTS"
echo "design   : $DESIGN"

if [ "$PAGES_DB_PATH" = "data/db/pages.db" ]; then
  echo "!! refusing to run against the durable database" >&2
  exit 1
fi

# --keep-checkpoint on every leg: without it a completed leg deletes its own
# checkpoint, and a later interruption in that leg starts it over.
leg() {
  local name="$1" methods="$2" variants="$3"
  local output="docs/reports/chunking/sweeps/chunk-size-sweep-${DATE}-${name}.md"
  echo
  echo "=== leg $name ($methods) over $variants — $(date -Is) ==="
  local start=$SECONDS
  uv run python experiments/indexing/compare_chunkings.py \
    --design "$DESIGN" \
    --variants "$variants" \
    --methods "$methods" \
    --keep-checkpoint \
    --output "$output"
  echo "=== leg $name done in $(((SECONDS - start) / 60))m — $(date -Is) ==="
}

# Both rungs per provider in one leg: the bare-vector row is where the embedder is
# the only thing acting, and the reranked row is the published pipeline. One leg
# means one model load for both.
leg qwen4b     "qwen4b,qwen4b_hybrid_rerank"         "$VARIANTS"
leg qwen8b     "qwen8b,qwen8b_hybrid_rerank"         "$VARIANTS"
leg nemotron8b "nemotron8b,nemotron8b_hybrid_rerank" "$VARIANTS"

# Last, because it is the most expensive leg and the least load-bearing: the three
# above are the question that was asked. If this one is cut short, the sweep still
# answers it.
leg rerank4b "qwen8b_hybrid_rerank_4b" "$RERANK_VARIANTS"

echo
echo "=== all legs done $(date -Is) ==="
uv run python experiments/indexing/merge_chunk_sweeps.py || true
