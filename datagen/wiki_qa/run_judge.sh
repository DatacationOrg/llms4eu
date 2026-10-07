#!/usr/bin/env bash
# Temporary: Ling answer judge (gen_rag.py judge -> answer_labels_ling) every 30 min, so answers get judged while
# run_rag.sh is still in a long gen pass. Usage: setsid nohup ./run_judge.sh >> gen_rag_judge2.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
export CLEAN_MODEL=ling
while true; do
  uv run --with langchain-openai python gen_rag.py judge --out answer_labels_ling --workers 150
  sleep 1800
done
