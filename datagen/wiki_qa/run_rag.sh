#!/usr/bin/env bash
# Temporary: answers/evidence/cross-lingual (gen_rag.py) for newly kept items, 24/7: Ling generates and judges
# (answer_labels_ling), then waits 30 min for the corpus runners to keep more items. Bunny judges in run_rag_bunny.sh.
# Usage: setsid nohup ./run_rag.sh >> gen_rag.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
export CLEAN_MODEL=ling
while true; do
  uv run --with langchain-openai python gen_rag.py gen --workers 200
  uv run --with langchain-openai python gen_rag.py judge --out answer_labels_ling --workers 200
  sleep 1800
done
