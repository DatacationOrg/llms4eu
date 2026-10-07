#!/usr/bin/env bash
# Temporary: keep the free-model jobs (Bunny, Ling) running 24/7: every 5 min, restart any runner that is not running
# (all steps resume from the DB, so a restart never redoes finished work). Logs restarts to watchdog.log.
# Usage: setsid nohup ./watchdog.sh >> watchdog.log 2>&1 &
cd "$(dirname "$0")"
while true; do
  for r in run_bunny_clean.sh:bunny_clean.log run_bunny_clean_b.sh:bunny_clean_b.log run_challenge.sh:gen_challenge.log run_ling_clean.sh:ling_clean.log run_rag.sh:gen_rag.log run_rag_bunny.sh:rag_bunny.log run_challenge_ling.sh:gen_challenge_ling.log run_extra.sh:gen_extra.log run_qrels.sh:gen_qrels.log run_judge.sh:gen_rag_judge2.log; do
    s=${r%%:*}; log=${r##*:}
    if ! pgrep -f "bash ./$s" > /dev/null; then
      echo "$(date '+%F %T') restarting $s"
      echo "=== watchdog restart $(date +%T) ===" >> "$log"
      setsid nohup "./$s" >> "$log" 2>&1 < /dev/null &
    fi
  done
  sleep 300
done
