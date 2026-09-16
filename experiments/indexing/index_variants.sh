#!/usr/bin/env bash
# Index an arbitrary set of providers across an arbitrary set of chunk variants.
#
# Split out of prepare_variants.sh because that script's job is to *create* a
# sweep (snapshot, cut, generate questions, index) and it hard-codes the two
# providers the first sweep used. Adding a provider to an existing sweep needs
# only the last of its four stages, and re-running the whole thing would re-cut
# chunks whose delete cascade takes the generated questions with it.
#
# One provider per `uv run` invocation, deliberately. The indexers memoize their
# loaded model, so indexing several providers in one process keeps every model
# resident: qwen8b alone is 15.1 GB and nemotron8b 15.9 GB in bf16, which does
# not fit beside each other and a reranker on one 48 GB card. A fresh process per
# provider gives the weights back to the driver between models.
#
# Resumable in the sense that indexing one (variant, provider) pair is
# idempotent: it deletes and rebuilds that one collection and touches no other.
# Re-run after an interruption and completed pairs are simply rebuilt, which is
# cheap when the embedding cache still holds their vectors.
set -euo pipefail

# Same reasoning as prepare_variants.sh, and the same deliberate absence of a
# `:-` default on the runtime variables: dotenv-load has already exported .env's
# CHROMA_PATH, so a default would never apply and the sweep would write into the
# production store, where rebuilding a collection destroys the running system's.
export CHROMA_PATH="${SWEEP_CHROMA_PATH:-.local/chroma-sweep}"
export PAGES_DB_PATH="${SWEEP_PAGES_DB_PATH:-.local/db/pages-variants.db}"

PROVIDERS="${PROVIDERS:-qwen4b qwen8b nemotron8b}"
VARIANTS="${VARIANTS:-base tok256 tok512 tok512ov tok1024}"

LOG="${LOG:-.local/logs/index-variants.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "--- run started $(date -Is) ---"
echo "pages db  : $PAGES_DB_PATH"
echo "chroma    : $CHROMA_PATH"
echo "providers : $PROVIDERS"
echo "variants  : $VARIANTS"

if [ "$PAGES_DB_PATH" = "data/db/pages.db" ]; then
  echo "!! refusing to run against the durable database" >&2
  exit 1
fi

# Byte-comparing the paths would let a quoted, relative, or differently spelled
# spelling of the same directory through, and indexing there deletes the running
# system's collections.
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

# Provider outer, variant inner: the model loads once per provider rather than
# once per cell, and a 15 GB checkpoint takes long enough to load that the order
# is worth fixing. Each variant is still its own process, so peak residency is
# one model either way.
for provider in $PROVIDERS; do
  for variant in $VARIANTS; do
    echo "=== index $variant / $provider ($(date -Is)) ==="
    start=$SECONDS
    if uv run python -m src.indexing.chunks \
        --method "$provider" --variant "$variant" 2>&1 | tail -2; then
      echo "    took $((SECONDS - start))s"
    else
      echo "!! skipped $variant / $provider after $((SECONDS - start))s"
    fi
  done
done

echo "=== done $(date -Is) ==="
uv run python -m src.eval.sweep_status
