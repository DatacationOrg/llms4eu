#!/usr/bin/env bash
# Temporary: farm challenging (BM25-hard) items with Space Bunny until the corpus is done; on a 429 wait 1 h.
# Usage: setsid nohup ./run_challenge.sh >> gen_challenge.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
while out=$(uv run --with bm25s --with langchain-openai python gen_challenge.py --part 0 --rpm 12 --workers 30 2>&1 | tee -a /dev/stderr | tail -1); [[ $out == *"stopped on 429"* ]]; do
  echo "$(date +%T) 429, sleeping 1h"; sleep 3600
done
