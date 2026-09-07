# Gemma as retrieval judge: what happened, what broke, where we are

**Status (7 Sep 2026):** All runs are stopped and the GPU is free. Nothing is committed yet. The benchmark is ready to relaunch and would take about 3.5 hours.

## The idea in one paragraph

Our retrieval benchmark uses a small local language model as a "judge". After a search, the judge looks at the retrieved text and says either "this is enough to answer the question" or "search wider". We had all our numbers with `gpt-oss:20b` as the judge. The goal was to rerun the same benchmark with `gemma4:31b` as the judge, so we can tell whether the disappointing results for the "page tools" agent were the agent's fault or the judge's.

## Attempt 1 (1 Sep): failed after 9.5 hours

The run finished only 1 of 4 cells. Two unrelated things went wrong.

**Gemma returned empty answers.** 136 times the judge gave back nothing usable. Every time that happens the code plays it safe and treats the answer as "not enough, search wider", which triggers another judge call, which can fail again. A cell budgeted at 2 hours ran 9 hours without finishing.

**The run died when the session closed.** The process stopped at the exact minute the terminal session that started it ended. No error, no crash: it was simply killed.

## What we found out this week

### Why gemma returned empty answers

Gemma "thinks" before it answers, and that thinking counts against the maximum number of words it is allowed to produce. On the short test prompt we had used, thinking was brief and everything looked fine (10 out of 10). On real prompts, which are 15 to 40 times longer, gemma often thought so long that it hit the limit before writing the actual answer. Result: empty output.

The old reliability test only used one short made-up prompt. That is why it said "100%" for a configuration that failed a quarter of the time in reality.

We rebuilt the test so it uses real questions with real retrieved text. On 50 real prompts:

| Configuration | Success | Speed |
|---|---|---|
| Thinking on, as configured on 1 Sep | 76% | 36 s per call |
| Thinking on, bigger output limit | 98% | 61 s per call |
| **Thinking off, tool-call output format** | **100%** on all three schemas | **9 s per call** |

So the fix is: turn thinking off and use the tool-call output format. Gemma is then reliable and about five times faster than the thinking version. We also confirmed that gemma's thinking is a simple on/off switch: the "low" and "high" effort settings that work on gpt-oss do nothing on gemma, so any "high effort" gemma results would be meaningless.

### Why the runs kept dying

This one took three tries to get right, so it is worth spelling out.

- **Try 1:** we assumed the chat tool that started the job killed it on exit. We launched the job fully detached from any terminal. It died anyway, at 17:01 on 3 Sep, again at the moment the login session ended.
- **The real cause:** the server is configured so that when a user logs out, *every* process belonging to that user is stopped. Detaching from the terminal does not help; the process is still "yours".
- **Try 2:** we started the job from cron instead, which runs outside user sessions. It still landed back inside the user session, because the `uv` tool we use to run Python moves itself there.
- **Try 3:** cron plus calling the project's Python directly, skipping `uv`. Verified working: the process sits outside the user session and would survive a logout.

Practical consequence for everyone on this server: **anything that must run overnight has to be started via cron and without `uv run`.** There is now a helper script for this (`experiments/indexing/launch_cron.sh`). The alternative is for an admin to enable "lingering" for the account, which would make the simple approach work again.

## Other things fixed along the way

- **Retries were wasted.** When the judge failed, the code retried the identical request three times with identical settings, so it failed identically. Retries now add a small amount of randomness, so a second attempt can actually succeed.
- **A circuit breaker.** If more than 20% of judge calls fail, the run now stops itself instead of burning GPU hours producing numbers that measure the fallback, not the method. This is what would have saved the 9.5 hours on 1 Sep.
- **A reporting bug.** The benchmark report was showing "0 retries, 0 tool calls" for every question, for every agent, in every report so far. This was a bug in how the log was passed to the report, not the agents doing nothing. It is fixed, so the next run is the first one where we can actually see how often the page tools were used.
- **Keep the model loaded.** The judge now tells the model server to keep gemma in memory for 24 hours, so it does not unload between cells and then fail to reload on a busy GPU.

## The rerun

Attempt 3 was launched on 4 Sep with the fixed configuration. The control cell (plain search, no agent) reproduced the committed number exactly (hit@5 = 0.881), which confirms we are benchmarking the same questions. There were zero judge failures. The run was stopped on request one hour into the second cell to free the GPU. The finished first cell is saved, so relaunching resumes from cell 2.

## Is gemma worth pursuing?

**Yes, for one specific question, and then decide.** Gemma is now reliable and as fast as it is going to get. Running the three cheap cells (about 3.5 hours) will tell us whether the page tools are ever actually used when a judge that provably can drive them is in charge. That is the question the gpt-oss report could not answer.

Reasons not to invest beyond that: gemma is still about nine times slower than gpt-oss per call, needs twice the GPU memory on a shared machine, cannot do graded reasoning effort, and nothing so far suggests the headline retrieval numbers will move. The DCI method was left out of the rerun on purpose: it fails for structural reasons (a keyword-only agent cannot match a Slovenian answer to an English question) and a judge swap will not change that.

**Recommendation:** run the three cells once via cron when the GPU is free, read the tool-usage column, and if the tools are still barely used, drop both the tool agent and the gemma track and spend the effort elsewhere.

## Where to look

- Detailed technical handoff with every measurement: `development_assets/plans/gemma-judge-rerun-handoff.md` (dated status section at the end)
- Reliability measurements on real prompts: `.local/reports/schema-reliability-real-2026-09-03-*.md`
- Run log: `.local/logs/gemma-suite-2026-09-03.log`
- Relaunch: `experiments/indexing/launch_cron.sh gemma-suite .venv/bin/python experiments/indexing/compare_chunkings.py --design shared --variants base --limit 500 --agentic-diagnostics --keep-checkpoint --methods qwen_hybrid_rerank,qwen_hybrid_agentic,qwen_hybrid_agentic_tools --output docs/agentic-gemma-2026-09-03.md`
- Before quoting any result: `grep -cE 'judge failed|JudgeCircuitOpen' .local/logs/gemma-suite-2026-09-03.log` must be close to zero.

## Update, later on 7 Sep: gemma parked

The judge has been switched back to the Azure DeepSeek model we started with. This is a
one-line config switch (`agentic_judge_provider: azure`); the gemma settings are kept
but inactive. No run has been started. The first thing to do when someone picks this up
is a small 20-question smoke run to confirm the Azure endpoint still answers and to see
how fast it is per judge call.
