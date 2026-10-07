#!/usr/bin/env bash
# Temporary: run dense_rank.py once the GPU has been free (others < 2 GB) for 10 consecutive minutes.
cd "$(dirname "$0")"
free_for=0
while [ $free_for -lt 10 ]; do
  used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [ "$used" -lt 2048 ]; then free_for=$((free_for+1)); else free_for=0; fi
  sleep 60
done
echo "$(date +%T) GPU free for 10 min"
uv run --with sentence-transformers --with langchain-openai python dense_rank.py pages && \
uv run --with sentence-transformers --with langchain-openai python dense_rank.py rank
echo "$(date +%T) done"
