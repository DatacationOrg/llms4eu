#!/usr/bin/env bash
# Temporary: second Bunny runner (20 workers) over the corpus queue from the top while the main chain
# (run_bunny_clean.sh) is busy with the teacher sample; the main chain later skips what is done here.
# Usage: setsid nohup ./run_bunny_clean_b.sh >> bunny_clean_b.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
run() { uv run --with langchain-openai python "$@"; }
until_ok() {
  while out=$(run "$@" 2>&1 | tee -a /dev/stderr | tail -1); [[ $out == *"stopped on 429"* ]]; do
    echo "$(date +%T) 429, sleeping 1h"; sleep 3600
  done
}
for c in $(ls queue/chunk_* | awk "NR % 2 == 1"); do  # even chunks (000, 002, ...); the main chain does the odd ones
  echo "$(date +%T) chunk $c"
  until_ok bunny_label.py --ids "$c" --workers 20
  until_ok bunny_fix.py --ids "$c" --workers 20
  until_ok bunny_verify.py --ids "$c" --workers 20
done
echo "$(date +%T) queue finished"
sleep 3600  # nothing new to do: avoid a watchdog restart every 5 min re-walking every finished chunk
