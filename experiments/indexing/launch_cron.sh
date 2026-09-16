#!/usr/bin/env bash
# Launch a long job from cron so it survives the end of the login session.
#
# `launch_detached.sh` (setsid, PPID 1) is not enough on this host: lingering is
# off for user accounts (Linger=no in the logind user record), so when the last
# SSH/VS Code session closes systemd stops the per-user manager and kills
# everything under it, detached or not. The 2026-09-01 and 2026-09-03 gemma
# sweeps both died that way, mid-cell, at the minute the session ended, leaving a
# clean Python shutdown warning as the last log line. Cron runs in the system
# slice and is unaffected. (If an administrator turns lingering on for the
# account, the plain launcher becomes sufficient.)
#
# Usage:
#   experiments/indexing/launch_cron.sh <name> <command...>
#
# Give it `.venv/bin/python ...`, NOT `uv run ...`: the uv snap re-homes its
# process into a transient scope under the per-user manager even when cron
# started it, which puts the job right back where logout kills it.
#
# Writes .local/run/<name>.sh (the job, with the sweep environment exported),
# installs a one-shot crontab line for the next minute that removes itself and
# runs the job, and logs to .local/logs/<name>.log. Verify a minute later with
#   pgrep -af <name>; cat /proc/<pid>/cgroup    # expect system.slice/cron.service
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <name> <command...>" >&2
  exit 2
fi
name="$1"; shift
root="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$root/.local/run" "$root/.local/logs"
job="$root/.local/run/${name}.sh"
log="$root/.local/logs/${name}.log"

printf -v quoted '%q ' "$@"
# Cron starts from an almost empty environment. Carry over the model-cache and
# device settings of the launching shell, or the reranker/embedder load fails
# with "couldn't find them in the cached files" (HF_HUB_CACHE lives in
# /home/share on this host, not under ~/.cache).
carried=""
for var in HF_HUB_CACHE HF_HOME HF_DATASETS_CACHE HF_HUB_OFFLINE TRANSFORMERS_CACHE \
           TORCH_HOME SENTENCE_TRANSFORMERS_HOME CUDA_VISIBLE_DEVICES OLLAMA_HOST; do
  if [ -n "${!var:-}" ]; then
    printf -v line 'export %s=%q\n' "$var" "${!var}"
    carried+="$line"
  fi
done
cat > "$job" <<JOB
#!/usr/bin/env bash
set -uo pipefail
cd "$root"
export CHROMA_PATH="${SWEEP_CHROMA_PATH:-.local/chroma-sweep}"
export PAGES_DB_PATH="${SHARED_PAGES_DB_PATH:-.local/db/pages-shared.db}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export PATH=/snap/bin:/usr/local/bin:/usr/bin:/bin
${carried}{
  echo "== launched \$(date -Is) pid \$\$ cgroup \$(cat /proc/self/cgroup)"
  echo "== command  : ${quoted}"
  echo "== chroma   : \$CHROMA_PATH | pages db : \$PAGES_DB_PATH"
  ${quoted}
  echo "== EXIT \$? \$(date -Is)"
} >> "$log" 2>&1
JOB
chmod +x "$job"

at_min="$(date -d '+1 min' +%M)"
at_hour="$(date -d '+1 min' +%H)"
marker="# one-shot:${name}"
{
  crontab -l 2>/dev/null | grep -vF -- "$marker" || true
  echo "$at_min $at_hour * * * crontab -l | grep -vF -- '$marker' | crontab - ; $job $marker"
} | crontab -
echo "job : $job"
echo "log : $log"
echo "cron: fires at ${at_hour}:${at_min}, then removes itself"
crontab -l | grep -F -- "$marker"
