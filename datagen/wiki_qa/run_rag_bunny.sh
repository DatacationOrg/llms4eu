#!/usr/bin/env bash
# Temporary: Bunny, the second judge of Ling's answers and translations (gen_rag.py judge -> answer_labels), 24/7;
# on a 429 waits 1 h, otherwise rechecks for new pages every 30 min. Usage: setsid nohup ./run_rag_bunny.sh >> rag_bunny.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
while true; do
  out=$(uv run --with langchain-openai python gen_rag.py judge --workers 20 2>&1 | tee -a /dev/stderr | tail -1)
  [[ $out == *"stopped on 429"* ]] && sleep 3600 || sleep 1800
done
