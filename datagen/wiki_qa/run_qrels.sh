#!/usr/bin/env bash
# Temporary: Ling multi-page relevance (gen_extra.py qrels), 24/7; each pass picks up new ok questions and checked
# unanswerable items. Usage: setsid nohup ./run_qrels.sh >> gen_qrels.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
export CLEAN_MODEL=ling
while true; do
  uv run --with bm25s --with langchain-openai python gen_extra.py qrels --workers 400 --rpm 600
  sleep 1800
done
