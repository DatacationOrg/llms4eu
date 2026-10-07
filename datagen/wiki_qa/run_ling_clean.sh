#!/usr/bin/env bash
# Temporary: Ling 3.1 Flash (Vercel, free; no rate limit since the account has credits) label -> repair -> verify on the corpus queue from the END, while
# the two Bunny runners work from the front; all steps resume from the DB. Every row stores the model id.
# Usage: setsid nohup ./run_ling_clean.sh >> ling_clean.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
export CLEAN_MODEL=ling
run() { uv run --with langchain-openai python "$@"; }
for c in $(ls -r queue/chunk_*); do
  echo "$(date +%T) chunk $c"
  run bunny_label.py --ids "$c" --workers 80
  run bunny_fix.py --ids "$c" --workers 80
  run bunny_verify.py --ids "$c" --workers 80
done
echo "$(date +%T) queue finished"
sleep 3600  # nothing new to do: avoid a watchdog restart every 5 min re-walking every finished chunk
