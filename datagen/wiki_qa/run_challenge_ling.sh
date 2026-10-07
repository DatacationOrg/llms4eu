#!/usr/bin/env bash
# Temporary: Ling as a second challenging-item generator (gen_challenge.py --part 1: half the queue by id hash, while
# Bunny takes the other half; both language-balanced), labelled by Ling too. Usage: setsid nohup ./run_challenge_ling.sh >> gen_challenge_ling.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
export CLEAN_MODEL=ling
while true; do
  uv run --with bm25s --with langchain-openai python gen_challenge.py --part 1 --rpm 600 --workers 300
  sleep 600
done
