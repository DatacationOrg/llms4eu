#!/usr/bin/env bash
# Temporary: Ling extra layers (gen_extra.py: meta, unanswerable, compare, variants; qrels runs in run_qrels.sh) so
# far), 24/7; every pass picks up the newly ok items. Usage: setsid nohup ./run_extra.sh >> gen_extra.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
export CLEAN_MODEL=ling
while true; do
  for s in meta unans compare variants; do
    echo "$(date +%T) $s"
    uv run --with bm25s --with langchain-openai python gen_extra.py $s --workers 150
  done
  sleep 1800
done
