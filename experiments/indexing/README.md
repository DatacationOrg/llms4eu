# Indexing Experiments

Benchmark orchestration over the retrieval catalog. These scripts read the
labelled eval questions in `$LLMS4EU_DATA/db/pages.db`, build retrievers from
`src/retrieval`, and score with metrics from `src/eval`.

## compare_qwen_modes.py

Runs a set of methods over the same questions, round-robin, and writes a
Markdown report.

```bash
just eval-report                                  # all 36 methods
uv run python experiments/indexing/compare_qwen_modes.py --limit 100
uv run python experiments/indexing/compare_qwen_modes.py --category crosslingual
uv run python experiments/indexing/compare_qwen_modes.py \
    --methods phase2-nemotron \
    --output docs/retrieval-results-phase2-nemotron.md
```

Method groups: `all` (the whole 36-method catalog) and `phase2`
(`nemotron_hybrid_rerank` against its `_v2` twin, isolating the chunk
representation). v2 indexes are separate derived artifacts, so existing
collections and reports stay valid — build one with `just index nemotron v2`.

Runs checkpoint into `$LLMS4EU_DATA/checkpoints/` and resume automatically. A
checkpoint only extends onto a compatible run; change the scoring shape and it
is rejected rather than silently mixed. Warmup queries are excluded from timing.

## judge_retrieval_equivalence.py

Asks a local LLM whether a method's retrieved evidence could answer the
question as well as the gold evidence, for cases scored as misses. Verdicts
cache in `$LLMS4EU_DATA/judge-cache/` so a re-run does not re-judge.

```bash
just eval-equivalence
```

`compare_qwen_modes.py --judge-equivalence` runs the same judge inline and adds
a `judge_hit@k` column.

Every model call is local, through Ollama.
