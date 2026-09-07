# Gemma Judge Rerun — Handoff

## Goal

Rerun the agentic retrieval benchmark with `gemma4:31b` as the sufficiency judge in
place of `gpt-oss:20b`, and produce numbers comparable cell-for-cell against the
committed gpt-oss results. The first attempt (2026-09-01) failed for two
independent reasons that must both be fixed before relaunching: the judge is
unreliable on real prompts, and the run did not survive the session that started
it. Neither is a retrieval problem.

Do not start the rerun before completing Fix 1 and Fix 2. Fix 1 is roughly ten
minutes and decides whether the rerun is worth running at all.

## Current Behavior

### What is committed

`998dd60` — the gpt-oss baseline measurements. 495 questions, `base` variant,
shared design, `qwen` embedder, `gpt-oss:20b` judge at `reasoning: low`:

| method | hit@1 | hit@5 | hit@10 | mrr@10 | ms/query | LLM calls |
|---|---|---|---|---|---|---|
| `qwen_hybrid_rerank` | 0.723 | 0.881 | 0.907 | 0.791 | 675 | 0 |
| `qwen_hybrid_agentic` | 0.725 | 0.889 | 0.915 | 0.795 | 6,062 | 597 |
| `qwen_hybrid_agentic_tools` | 0.719 | 0.879 | 0.903 | 0.787 | 6,572 | 617 |
| `dci` | 0.576 | 0.697 | 0.723 | 0.632 | 22,084 | 2,898 |

Written up in `docs/agentic-retrieval-report-2026-09-01.md`. Its three headline
conclusions: the page tools give no benefit (95% of rankings untouched, 5
discordant pairs of 495); DCI fails badly and its `crosslingual` collapse
(0.857 → 0.352) is architectural rather than a bug; and the harness has a
run-to-run noise floor near 1pp, so the retry loop's +0.8pp is not established.

### What is uncommitted, and why

Everything below is in the working tree. It is entangled with roughly 3,400 lines
of unrelated in-progress chunking-variant work in the same files, which is why it
was not committed alongside the measurements. Do not sweep that in without asking.

| change | files | state |
|---|---|---|
| DCI citation resolution, answer retry, drop logging, `_signature` fix | `src/retrieval/retrievers/dci.py`, `src/retrieval/config.yaml`, `tests/test_retriever_dci.py` | done, 20 tests pass |
| `agentic_reasoning_levels` dimension (18 methods) | `src/retrieval/config.yaml`, `src/retrieval/methods.py`, `src/eval/agentic_diagnostics.py`, `tests/test_rag_methods.py`, `src/retrieval/README.md` | done, tests pass |
| Judge switched to `gemma4:31b` plus method pairing | `src/retrieval/config.yaml` | done |
| `reasoning` / `reasoning-dci` sweep presets | `experiments/indexing/compare_chunkings.py` (untracked) | done |
| Schema reliability probe | `experiments/indexing/probe_structured_output.py` (untracked) | **has the flaw in Fix 1** |
| `render_qwen_report.py` | committed in `998dd60` | done |

`src/eval/agentic_diagnostics.py` is the only shared file whose diff is entirely
from this work (18 insertions), so it can be committed cleanly on its own.

### Why the first gemma run failed

Launched 14:08, died 23:46 the same day, 9h37m in, having completed **cell 1 of 4**.

**Cause 1 — the judge is unreliable on real prompts.** `ChunkSufficiency` via
`json_schema` measured 10/10 on the probe and logged **136 hard failures**
(`OutputParserException: Invalid json output`) in the real cell, still climbing
when the process died. The probe used one short synthetic prompt; the benchmark
uses 495 real Slovenian prompts carrying retrieved chunk text. The probe did not
represent the workload. For contrast, gpt-oss on its correct pairing had **zero**
failures in 597 real calls.

Failures compound rather than merely costing time. `_judge_locally` returns
`sufficient=False` on failure — correct, since a broken judge must not be trusted
to say "good enough" — so every failure tells the agent to widen its search, which
issues another judge call, which may also fail. Cell 2 was budgeted at 1.9h and
ran 9.3h without finishing.

**Cause 2 — the run did not survive the session.** The process was a session leader
with no controlling terminal (`SID == PID`, `TT = ?`), so no dropped SSH or VS Code
connection could `SIGHUP` it. That analysis was correct and irrelevant: Claude Code
tears down its tracked background tasks when the session ends, and the log's final
line is a Python shutdown warning stamped at the minute the session ended. A long
job must be detached from the harness, not merely from a terminal.

### Measured facts about gemma worth keeping

Per-schema first-attempt success, 10 samples per cell,
`.local/reports/schema-reliability-2026-09-01.md`:

|  | ChunkSufficiency | ToolAction | CorpusAction |
|---|---|---|---|
| `gemma4:31b` function_calling | **0%** | 100% | 100% |
| `gemma4:31b` json_schema | 100% (see note) | 100% | **0%** |
| `gpt-oss:20b` function_calling | 100% | **0%** | 100% |
| `gpt-oss:20b` json_schema | 100% | 100% | **0%** |

Note: 100% on the synthetic probe prompt, but it fails repeatedly on real prompts.
This is the cell Fix 1 must re-measure.

Three things follow:

- The correct structured-output method is a property of **(model, schema)**, not of
  either alone, and the first column is *inverted between the two models*. A wrong
  pairing does not raise — it returns no structured output, the retriever falls
  back, and the cell silently scores something other than the method under test.
  `agentic_judge_structured_method` is therefore already switched to `json_schema`
  for gemma; leaving it at `function_calling` would have produced a full suite of
  meaningless numbers.
- `CorpusAction` is `function_calling`-only for **both** models. That pairing is the
  one the judge swap does not touch.
- Gemma costs ~12 s/call against gpt-oss's ~1 s, and 28 GB of VRAM against 13 GB, on
  a GPU shared with other users. Even at perfect reliability that makes the full
  suite a multi-day job.

Gemma declares `thinking` in its Ollama capabilities, but `reasoning: low` and
`reasoning: high` produced identical success rates *and* identical latencies
(11.7/11.7 s, 7.6/7.6 s) across all six cells. It appears to support thinking as a
boolean, not as a graded effort level; graded effort looks gpt-oss-specific.
`qwq:latest` declares no `thinking` capability at all. So the
`agentic_reasoning_levels` dimension is only meaningful on gpt-oss today, and the
`_high` method names are cosmetic on gemma. This was never confirmed by a direct
test: the definitive check (comparing `think` false/true/"low"/"high" and measuring
thinking-token counts) timed out against a busy GPU and should be run.

## Implementation

### Fix 1 — make the probe use real prompts (do this first)

`experiments/indexing/probe_structured_output.py` builds one synthetic prompt per
schema in `_cases()`. That is what made a 10/10 result meaningless. Replace it with
prompts built from the actual eval set.

1. Add `--from-eval N`, which samples N approved questions from `PAGES_DB_PATH`
   (deterministic order, e.g. `order by q.id limit N`), runs the real first-stage
   retriever for each, and builds each schema's prompt from the real query and real
   retrieved chunks via the existing `_sufficiency_prompt`, `_action_prompt` and
   `_corpus_prompt` helpers. Keep the synthetic path for a fast smoke test.
2. Report first-attempt success, the exception histogram, **and prompt length in
   tokens**, bucketed. The hypothesis worth testing is that failures correlate with
   prompt length against `agentic_judge_num_predict: 1024`; if so, raising that cap
   may be the actual fix and gemma may be usable after all.
3. Run 50 samples for `gemma4:31b` on `ChunkSufficiency`/`json_schema`,
   `ToolAction`/`json_schema` and `CorpusAction`/`function_calling`. Save to
   `.local/reports/`.

**Decision gate.** If real-prompt success for `ChunkSufficiency` is below ~95%,
gemma cannot be the sufficiency judge: `function_calling` is already 0% for that
schema, so there is no second option, and every `*_agentic*` method depends on it.
In that case either stop and report, or run only the tool agent, whose `ToolAction`
schema was robust either way. Do not spend GPU hours on a judge known to fail —
that is exactly what the first attempt did.

### Fix 2 — launch detached from the harness

Do not rely on the agent harness's background execution. Launch through a
double-forked wrapper (`setsid` plus output redirection to a log file and stdin from
`/dev/null`), so the process is reparented to init and outlives any session.

Verify before walking away: the pid must have `PPID 1`, or at least a non-`claude`
ancestor, and no controlling terminal. Check with
`ps -o pid,ppid,sid,tty,stat -p <pid>`.

Also set `OLLAMA_KEEP_ALIVE=24h` in the launch environment. Ollama drops an idle
model after about five minutes; between cells there is a model-loading gap, and if
gemma unloads while another user has taken the 28 GB it needs, the judge fails for
the remainder of that cell — silently.

### Fix 3 — fail fast instead of retrying identically

`LocalOllamaStructuredLlm.structured_output` retries up to `judge_retries: 3` at
`temperature=0`, appending a corrective instruction. The corrective prompt helps,
but the first attempt is deterministic, so a prompt that fails structurally tends to
keep failing. In the gpt-oss DCI run, 35 hard failures cost about 70 wasted
invocations; with gemma at 12 s/call the waste is far larger, and it is what
inflated cell 2 to 4.7× its budget.

Options, in order of preference: vary temperature on retry (cheapest, and keeps the
retry useful); or add a per-run failure-rate circuit breaker that aborts a cell once
failures exceed a threshold, so an unusable cell stops in minutes rather than hours.
The second is what would have saved the 9.5 hours.

### The rerun

Once Fix 1 clears the gate and Fix 2 is in place:

```
--variants base --design shared --limit 500 --agentic-diagnostics --keep-checkpoint
--methods qwen_hybrid_rerank,qwen_hybrid_agentic,qwen_hybrid_agentic_tools,dci
--output docs/agentic-gemma-<date>.md
```

Use the **unsuffixed** method names, not `_high`: reasoning effort is inert on
gemma, and the suffix would imply a contrast that does not exist. Keep
`qwen_hybrid_rerank` in the list as a control — it must reproduce hit@5 ≈ 0.881,
which is how you know the question sample matches the committed run. It has now
reproduced three times: 0.8808, 0.8808, 0.8828.

Cost at gemma's measured rates, assuming reliability is fixed:

| cell | LLM calls | estimate |
|---|---|---|
| `qwen_hybrid_rerank` | 0 | ~7 min |
| `qwen_hybrid_agentic` | ~597 | ~1.9h |
| `qwen_hybrid_agentic_tools` | ~705 | ~1.2h |
| `dci` | ~2,968 | ~6.3h |
| | | **~9.5h** |

Consider dropping `dci` from the first rerun. It is two thirds of the cost, and its
failure is architectural — a grep-only agent cannot match a Slovenian answer to an
English question — so a judge swap is not expected to move it. Running the first
three is about 3.2h and answers the question that matters: **were the page tools
inert because the tools are useless, or because gpt-oss could not drive them?**
Gemma handles `ToolAction` reliably, so if the tools have value this is the run that
shows it. If the tool agent is still flat at about 4.8% of rankings changed, that
result is much harder to explain away.

### Before reporting any number from the rerun

Check the judge-failure count per cell first:

```
grep -cE 'judge failed|corpus agent failed' .local/logs/<name>.log
```

A cell with hundreds of failures measured BM25 fallback, not the method named in its
row. Report that, not a hit@5. Getting this backwards is the mistake that put a 7%
caveat into the gpt-oss report only after the numbers had already been quoted.

Note also that the action log is still not persisted by the sweep harness, so
`retried`, `rewritten`, `expanded`, `tool_calls` and `tools_used` come back empty for
every question, and a `0` there is missing data rather than inaction. This is why the
gpt-oss report cannot say how often `search_in_page` was actually called. Fixing that
is worth doing before drawing a final conclusion about the page tools.

## Environment Traps

- `.env` sets `CHROMA_PATH=.local/chroma`, which has **no `qwen8b` collection** and a
  dead `english` collection at 0 rows; every dense method reports a missing index
  against it. Use `.local/chroma-sweep`, which holds all five providers.
- `data/db/pages.db` has **no `variant` column**, so the uncommitted variant
  threading makes any `dci*` method raise `no such column: variant` against the
  durable store. DCI runs only against `.local/db/pages-shared.db` (shared design,
  3,476 questions, 429 anchors) or `.local/db/pages-variants.db` (per-variant,
  20,218 questions). Use the shared one to match the committed numbers.
- Adding a method to a finished run's `--methods` **changes the checkpoint signature
  and discards every completed cell** (`compare_chunkings.py`, "starting fresh"). Use
  a separate `--output` and merge with `render_qwen_report.py <ckpt> <ckpt> ...`.
- `*.checkpoint.json` is gitignored, and `ecfbd68` removed one for adding 130k lines
  of churn. Commit the rendered reports, not the checkpoints.
- The GPU is shared with other users. The same baseline cell has taken 339s, 378s and
  713s depending on contention, so treat all latency figures as indicative and never
  compare timings across runs.
- The harness prices every agentic cell at a flat 5.1 s/query. Measured: 6.1 (retry
  loop), 6.6 (tools), 22.1 (DCI) with gpt-oss, and about 12 s/call with gemma. Ignore
  its cost estimate for anything involving DCI or gemma.

## Status 2026-09-03

### Fix 1 — done, gate cleared with a config change

`probe_structured_output.py --from-eval N` builds each schema's prompt from the first
N benchmark questions with real first-stage retrieval, and records prompt tokens,
Ollama stop reason and thinking size per call. Real prompts are 3.6k–10.5k tokens
(median ~6k); the synthetic prompt was 253. Reports in
`.local/reports/schema-reliability-real-2026-09-03-*.md`.

| cell (gemma4:31b, 50 real prompts) | first-attempt | s/call | failures |
|---|---|---|---|
| `ChunkSufficiency` json_schema, num_predict 1024 (as configured on 09-01) | **76%** (38/50) | 36 | 12, all `done_reason=length`, all exactly 1024 output tokens |
| `ChunkSufficiency` json_schema, num_predict 4096 | **98%** (49/50) | 61 | 1, `length` after an 11.9k-char thinking block |
| `ToolAction` json_schema, 4096 | queued in the chain below | | |
| `CorpusAction` function_calling, 4096 | queued in the chain below | | |

The failures do not correlate with prompt length (failed prompts span 3.6k–10k like
the successes); they correlate with thinking length (failed mean 3.4k chars vs 2.1k).
So the cause is the output cap being spent on thinking, and the fix is the cap:
`agentic_judge_num_predict` is now **4096** in `src/retrieval/config.yaml`. It is a
bound, not a budget — successful calls emit 26–256 output tokens.

The "definitive check" on thinking levels was run with direct `/api/chat` calls:
`think` None/True/"low"/"high" give byte-identical output and latency on gemma;
`think=False` with a JSON schema returns prose (`sufficient=true`). Thinking is a
boolean on gemma, and with `json_schema` it must stay on. Ollama rejects the string
`"false"` for `think` with HTTP 400, so the probe parses `false`/`true`/`none` into
the right types. A parallel probe launched today (`.local/logs/schema-probe-nothink.log`,
reports `*-fc-nothink.md`) measures `function_calling` with thinking **off** on real
prompts for all three schemas; on the synthetic prompt that pairing was 100% for
`ChunkSufficiency`. If it holds on real prompts it is a second, much faster judge
configuration for gemma, and the sweep config should be chosen on that evidence
before the rerun is started.

Latency is 36–61 s/call, not the 12 s measured on the synthetic prompt (longer
prompts, and a shared GPU at 99%). Rerun cost at ~50 s/call: ~8h for the retry loop,
~10h for the tool agent. DCI is dropped from the first rerun as the plan suggested.

### Fix 2 — done

`experiments/indexing/launch_detached.sh <name> <cmd...>`: setsid+nohup, logs to
`.local/logs/<name>.log`, exports the shared-design stores, prints the pid. Verified
PPID 1 / TT ? on every launch. `agentic_judge_keep_alive: 24h` is sent by the judge
on every request (the server env var alone would not reach an already-running
Ollama).

### Fix 3 — done

`LocalOllamaStructuredLlm`: first attempt at temperature 0, retries at
`agentic_judge_retry_temperature` (0.3). Circuit breaker
`agentic_judge_max_failure_rate: 0.2` after `agentic_judge_breaker_min_calls: 20`
raises `JudgeCircuitOpen` (not a RuntimeError, so the retrievers cannot swallow it)
and aborts the run; completed cells stay in the checkpoint. Tests in
`tests/test_shared_llm.py`.

### Action log — fixed

`compare_chunkings._diagnostics` passed the action log without `rankings`, so the
diagnostics aligned zero actions to every question and reported 0 retries / 0 tool
calls for agents that made hundreds. Fixed with a regression test in
`tests/test_compare_chunkings.py`. The 2026-09-03 run is the first whose
`retried`/`tool_calls` columns mean anything.

### Judge configuration — decided on real prompts

Thinking off + `function_calling`, 50 real prompts each, 2026-09-03
(`.local/reports/schema-reliability-real-2026-09-03-*-fc-nothink.md`):

| schema | first-attempt | s/call | thinking on + json_schema (4096) for contrast |
|---|---|---|---|
| `ChunkSufficiency` | **100%** (50/50) | 9.5 | 98%, 61 s/call |
| `ToolAction` | **100%** (50/50) | 8.9 | not measured (probe cancelled as moot) |
| `CorpusAction` | **100%** (50/50) | 13.6 | 100% with thinking on, 7.6 s/call (synthetic) |

So the 2026-09-01 grid's "gemma function_calling 0% on ChunkSufficiency" was a
thinking-on artefact. `src/retrieval/config.yaml` now has
`agentic_judge_reasoning: false`, `agentic_judge_structured_method: function_calling`,
`agentic_tools_structured_method: function_calling`, `agentic_judge_num_predict: 4096`.
The `_high`/`_low` methods turn thinking back on and are unusable on gemma.

### The rerun — died once more; relaunched from cron without the uv snap

Attempt 2 started 2026-09-03 16:06 (setsid, PPID 1, thinking-off function_calling
config) and **died at 17:01:06 after cell 1 of 3**: no traceback, last log line a
Python shutdown warning, the same signature as 09-01. Cell 1 is safe in
`docs/agentic-gemma-2026-09-03.md.checkpoint.json`: `qwen_hybrid_rerank` hit@5
**0.8808**, 495 questions, 0 judge failures. GPU was free; no reboot.

Cause: the user journal shows the uv snap's transient scope for the sweep
(`user@1061.service/app.slice/snap.astral-uv.uv-*.scope`) ending at 17:01:06 with no
stop job logged. The logind user record has `Linger=no` and yesterday's login
session is gone, so when the last session closed systemd stopped the per-user
manager and everything under it. `setsid`/PPID 1 detaches from the shell, not from
the user slice. Fix 2 as written above was necessary but not sufficient.

Two more facts learned on 2026-09-04 while relaunching:

- A job started from **cron** but run through **`uv run`** still lands in
  `user@1061.service/.../snap.astral-uv.uv-*.scope`: the snap re-homes its process
  under the per-user manager regardless of who started it. Verified on pid 2261892
  and killed.
- Running `.venv/bin/python` directly from cron stays in
  `system.slice/cron.service`. `.env` is loaded by `src/shared/env.py` itself, so
  nothing from `uv run` is lost.

Attempt 3 therefore uses `experiments/indexing/launch_cron.sh` semantics: a one-shot
crontab line that removes itself and runs `.local/run/gemma-suite.sh`, which resumes
the checkpoint, refuses to start beside another `compare_chunkings`, prints the judge
config it saw, and calls `.venv/bin/python experiments/indexing/compare_chunkings.py`
with the arguments below. Same log file. Started 2026-09-04 10:40:05, pid 2266412,
cgroup `/system.slice/cron.service`, zero inherited session variables. **Stopped on
request at 11:45, one hour into cell 2, 0 judge failures; gemma unloaded from the
GPU.** Cell 1 remains in the checkpoint, so a relaunch of the same command resumes
at cell 2. Verify with
`cat /proc/$(pgrep -f 'compare_chunkings[.]py' | head -1)/cgroup`; it must say
`cron.service`. Having an administrator turn lingering on for the account would make
the plain setsid launcher sufficient again.

```
compare_chunkings.py --design shared --variants base --limit 500 --agentic-diagnostics
  --keep-checkpoint --methods qwen_hybrid_rerank,qwen_hybrid_agentic,qwen_hybrid_agentic_tools
  --output docs/agentic-gemma-2026-09-03.md
```

Expected cost at ~9 s/call: baseline ~7 min, retry loop ~1.5h, tool agent ~1.8h,
about 3.5h in all. Before quoting a number: `qwen_hybrid_rerank` hit@5 must be
≈0.881, and

```
grep -cE 'judge failed|JudgeCircuitOpen' .local/logs/gemma-suite-2026-09-03.log
```

must be small. A `JudgeCircuitOpen` traceback means the judge failed >20% of calls
and the run stopped itself; completed cells stay in the checkpoint. This is the
first run whose diagnostics columns (`retried`, `tool_calls`) are populated, so the
page-tools question — inert tools, or a judge that could not drive them — can
finally be answered from the `qwen_hybrid_agentic_tools` row.

To stop it: `pkill -f 'gemma-suite[.]sh'; pkill -f 'compare_chunkings[.]py'` (the
bracket keeps pkill from matching its own command line).

## Status 2026-09-03

Fix 1, Fix 2 and Fix 3 are done; the rerun is launched. What the probe found changed
the plan: the fix is not a bigger output cap, it is turning thinking off.

**Fix 1 (real-prompt probe).** `probe_structured_output.py --from-eval N` builds each
schema's prompt from the first N benchmark questions with real first-stage retrieval,
and records prompt tokens, Ollama stop reason and thinking size per call. Reports in
`.local/reports/schema-reliability-real-2026-09-03-*.md`, 50 questions each:

| gemma4:31b configuration | ChunkSufficiency | ToolAction | CorpusAction | s/call |
|---|---|---|---|---|
| json_schema, thinking on, cap 1024 (the 09-01 config) | 76% | — | — | 36 |
| json_schema, thinking on, cap 4096 | 98% | — | — | 61 |
| **function_calling, thinking off** | **100%** | **100%** | see report | **9** |

Every failure at cap 1024 was `done_reason=length` with empty content: gemma's
scratchpad (2-12k chars on real prompts, ~850 on the synthetic one) ran past the cap
before the JSON started. Prompt length does not predict it; failures sit in the same
token bucket as successes. A brevity instruction in the prompt is ignored. `think` is a
switch on gemma (None/True/low/high are byte-identical), and with thinking off,
json_schema returns prose while function_calling parses every time. So the 09-01 grid's
"function_calling 0% for ChunkSufficiency" was a thinking-on result, and the pairing
inverts with thinking off.

**Config now** (`src/retrieval/config.yaml`): `agentic_judge_reasoning: false`,
`agentic_judge_structured_method: function_calling`,
`agentic_tools_structured_method: function_calling`, `agentic_judge_keep_alive: 24h`,
`agentic_judge_retry_temperature: 0.3`, `agentic_judge_max_failure_rate: 0.2` (circuit
breaker, `JudgeCircuitOpen`, not a RuntimeError so retrievers cannot swallow it). The
`_high` methods turn thinking back on and are unusable on gemma.

**Fix 2.** `experiments/indexing/launch_detached.sh <name> <cmd...>`: setsid+nohup,
shared-design stores exported, log in `.local/logs/<name>.log`. Verified PPID 1 / TT ?.

**Also fixed.** The sweep's agentic diagnostics reported zero retries and zero tool
calls because `compare_chunkings._diagnostics` passed the action log without the
rankings the builder zips it against. Regression test added; the smoke run now shows
retries. The 08-31 and 09-01 "tools inert" diagnostics tables are therefore missing
data, not evidence of inaction; the rerun will produce the real ones.

**Launched** (detached, `.local/logs/gemma-suite-2026-09-03.log`):
`--methods qwen_hybrid_rerank,qwen_hybrid_agentic,qwen_hybrid_agentic_tools`,
`--output docs/agentic-gemma-2026-09-03.md`, DCI dropped as suggested. At 9 s/call the
two agent cells are ~1.6h and ~1.8h. Before quoting anything:
`grep -cE 'judge failed|JudgeCircuitOpen' .local/logs/gemma-suite-2026-09-03.log`
and check `qwen_hybrid_rerank` reproduces hit@5 ≈ 0.881.

## Status 2026-09-07 — gemma parked, judge back on Azure DeepSeek

On request, the agentic judge is switched back to the Azure AI Foundry deployment the
family was first built on (`AZURE_AI_MODEL=DeepSeek-V4-Pro` in `.env`). Implemented,
not run:

- `src/shared/llm.py`: `AzureFoundryStructuredLlm` restored (it was removed in
  `38d79f7` when the local judges arrived), now with the same retry temperature and
  circuit breaker as the Ollama client. `from_env()` names any missing variable.
- `src/retrieval/retrievers/agentic.py`: `default_judge` reads
  `agentic_judge_provider` (`azure` | `ollama`). Every agent (`*_agentic*`,
  `*_agentic_tools*`, `dci*`) goes through it, so one switch moves the whole family.
- `src/retrieval/config.yaml`: `agentic_judge_provider: azure`. The Ollama block
  (gemma4:31b, thinking off, function_calling, cap 4096) stays as the measured-good
  local configuration but is inactive. `agentic_judge_num_predict` is the Azure
  `max_tokens`; `dci_num_predict` (4096) still applies to DCI.
- Tests: `tests/test_shared_llm.py` covers JSON extraction from prose, retry
  correction and sampling, breaker, missing env vars, key hidden from `repr`, and
  the provider switch.

Not changed: the reasoning-rung methods (`_low`/`_high`) still exist; on Azure the
rung is ignored, so they measure the same thing as the unsuffixed method. The
structured-output probe (`probe_structured_output.py`) is Ollama-only.

Nothing has been executed against Azure. The first real call should be a small
`--limit 20` sweep to confirm the endpoint, key and `json_object` response format
still behave, and to see DeepSeek's latency per judge call before budgeting a run.

### 2026-09-07 — DeepSeek benchmark launched

Connectivity confirmed with two real judge calls (1.4–1.8 s each, both parsed first
try; `from_env` now loads `.env` itself because nothing on the benchmark path did).
Full four-method run launched from cron as `deepseek-suite-2026-09-07`
(`.local/logs/deepseek-suite-2026-09-07.log`, output
`docs/agentic-deepseek-2026-09-07.md`). First launch died at startup: cron's bare
environment lacks `HF_HUB_CACHE=/home/share/cache/huggingface/hub`, so the reranker
could not be loaded offline. `launch_cron.sh` now copies the model-cache and device
variables of the launching shell into the job script.

Second launch died the same way with the cache variable set: the shared cache
`/home/share/cache/huggingface/hub` was pruned on 2026-09-04 14:42 (dir mtime, entries
owned by other users) and `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen3-Embedding-0.6B`,
`Qwen3-Embedding-4B` and the Nemotron embedder are gone; earlier logs had no
"audit: no local tokenizer" lines, this one has eight. Re-downloaded the two models
this run needs (reranker + qwen embedder, ~2.4 GB) into that cache and relaunched.
Anyone running the `qwen4b`, `nemotron` or `_high` cells will hit the same wall.

Third launch ran cell 1 in 23 s with hit@5 **0.774** (should be ≈0.881): my filtered
re-download had left out the reranker's `chat_template.jinja`, which the Qwen3
reranker uses to format its query/document prompt, so it scored noise. Killed the run,
downloaded the complete snapshots, and re-ran the control cell locally: hit@5 0.8848,
672 ms/query, 337 s — within the ~1pp noise band of the earlier controls (0.8808,
0.8808, 0.8828). Deleted the bad checkpoint and relaunched. Lesson: when restoring a
model to the cache, take the whole snapshot; a control cell that is suddenly fast is
a broken control, not a fast one.
