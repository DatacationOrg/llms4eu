#!/usr/bin/env bash
# Launch a long job so it outlives the shell, the SSH session, and any agent
# harness that started it.
#
# The 2026-09-01 gemma sweep died 9.5 hours in, at the minute the Claude Code
# session that had started it ended. It was a session leader with no controlling
# terminal, so SIGHUP was not the problem; the harness tears down the background
# tasks it tracks. A job launched here is reparented to init via setsid and a
# double fork, reads stdin from /dev/null, and writes to .local/logs/<name>.log.
#
# NOT ENOUGH ON THIS HOST for anything that must outlive your login: lingering
# is off for user accounts (Linger=no), so systemd stops the per-user manager
# when the last session closes and kills every process under it, PPID 1 or not.
# The 2026-09-03 sweep died that way at 17:01. Use launch_cron.sh for multi-hour
# jobs; this script is fine for jobs that end while you are still logged in.
#
# Usage:
#   experiments/indexing/launch_detached.sh <name> <command...>
#   experiments/indexing/launch_detached.sh gemma-suite \
#       uv run python experiments/indexing/compare_chunkings.py --design shared ...
#
# Defaults to the shared-design sweep stores; override SWEEP_CHROMA_PATH or
# SHARED_PAGES_DB_PATH to point elsewhere. Prints the pid and its ps line so
# you can confirm PPID 1 before walking away.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <name> <command...>" >&2
  exit 2
fi
name="$1"; shift

cd "$(dirname "$0")/../.."
mkdir -p .local/logs
log=".local/logs/${name}.log"

export CHROMA_PATH="${SWEEP_CHROMA_PATH:-.local/chroma-sweep}"
export PAGES_DB_PATH="${SHARED_PAGES_DB_PATH:-.local/db/pages-shared.db}"
# Server-side default for models whose client does not send keep_alive. The judge
# itself sends `agentic_judge_keep_alive` on every request; this covers the rest.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

{
  echo "launched : $(date -Is)"
  echo "command  : $*"
  echo "chroma   : $CHROMA_PATH"
  echo "pages db : $PAGES_DB_PATH"
  echo "----"
} > "$log"

# setsid gives the job its own session (no controlling terminal); nohup covers
# HUP; the trailing & plus disown detaches it from this shell, and when this
# script exits the job is reparented to init. `$!` is the job's pid because
# setsid execs in place when it is not already a process-group leader.
nohup setsid "$@" < /dev/null >> "$log" 2>&1 &
pid=$!
disown "$pid"
sleep 1
echo "log : $log"
echo "pid : $pid"
if ps -p "$pid" > /dev/null 2>&1; then
  ps -o pid,ppid,sid,tty,stat,etime,cmd -p "$pid" | cut -c1-140
  echo "(PPID becomes 1 once this script exits; TT must be ? and SID must equal PID)"
else
  echo "the job exited within a second; check the log" >&2
  exit 1
fi
