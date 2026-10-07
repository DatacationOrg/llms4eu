#!/usr/bin/env bash
# Temporary: keep Space Bunny busy 24/7: label, repair and verify the teacher sample, then the corpus queue in chunks.
# One process at a time, so the 50 rpm pacing holds. On a 429 wait an hour and resume (scripts are resumable).
# Usage: setsid nohup ./run_bunny_clean.sh >> bunny_clean.log 2>&1 &
cd "$(dirname "$0")"
set -a; . ../.env; set +a
run() { uv run --with langchain-openai python "$@"; }
until_ok() {  # rerun a step until it finishes without a 429
  while out=$(run "$@" 2>&1 | tee -a /dev/stderr | tail -1); [[ $out == *"stopped on 429"* ]]; do
    echo "$(date +%T) 429, sleeping 1h"; sleep 3600
  done
}
until_ok bunny_label.py --ids teacher_ids.txt --workers 30
until_ok bunny_fix.py --ids teacher_ids.txt --workers 30
until_ok bunny_verify.py --ids teacher_ids.txt --workers 30
mkdir -p queue
[ -e queue/chunk_000 ] || split -l 2000 -d -a 3 corpus_queue.txt queue/chunk_
for c in $(ls queue/chunk_* | awk "NR % 2 == 0"); do  # odd chunks (001, 003, ...); runner B does the even ones
  echo "$(date +%T) chunk $c"
  until_ok bunny_label.py --ids "$c" --workers 30
  until_ok bunny_fix.py --ids "$c" --workers 30
  until_ok bunny_verify.py --ids "$c" --workers 30
done
echo "$(date +%T) queue finished"
sleep 3600  # nothing new to do: avoid a watchdog restart every 5 min re-walking every finished chunk
