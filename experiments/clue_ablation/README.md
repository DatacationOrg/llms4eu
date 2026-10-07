# Clue ablation: how much does retrieval depend on the detail in a question?

*This file: one experiment, the source of truth for its findings and decisions. Cross-experiment summary and ratings: [overview](../README.md).*

## In short (2026-10-07)

**Status: parked after a 30-question pilot.** The tooling works; the full clue extraction is blocked on LLM throughput (the free API models are rate-limited, the local GPU route is not yet measured since a speed fix). Part of the question is now answered without an LLM by removing anchors (numbers, names) from questions: [language anchors, step 3](../language_anchors/README.md#step-3-anchors-removed-from-the-full-question).

**Insights (pilot, n = 30: signals, not results)**

1. **Hard questions carry many clues:** 5 on average (3-7), mostly location, type, feature, quantity and date. Fits the finding that they were generated to be unique, so details were stacked. *Evidence: gpt-oss:20b extraction, 98% of clues quoted literally from the question.*
2. **Clue counting with an LLM is feasible but fuzzy at the edges:** two LLMs agree within one clue on 18 of 20 questions; documentary clues (numbers, dates) are counted more consistently than clues in general. *Evidence: 20 questions, two extractors.*
3. **On these questions dense retrieval does far worse than BM25:** 33% vs 83% hit@10 at 512-token chunks, the same at 256; hybrid (0.7 dense) falls in between, 77%. Dense finds the right kind of place in the right region, not the right one ([why](../language_anchors/README.md#dense-failure-analysis-30-pilot-questions)). Later evidence suggests BM25's lead is inflated by the numbers the generator added.

**Next, when unparked:** measure local GPU speed with the context fix; human clue count on ~30 items; then phase A on 300-1,000 questions.

## Question and design

Treat **detail as an experimental variable** instead of labelling questions as realistic or not (tried; too ambiguous, see [audit finding 12](../../docs/reports/wiki-audit/audit-2026-10-06.md)): how does retrieval change when the same information need carries fewer clues, or other kinds of clues? This measures the effect of the amount and type of information, not vocabulary mismatch (the clues stay page wording) or real user behaviour.

**A clue** is one constraint on the place: one attribute with one value, **quoted literally** from the question, so it can be checked by string match. Attributes: `name`, `location`, `type`, `feature`, `quantity`, `date`, `event`, `other` ([prompt](../../prompts/clue_extract.md)). Per clue, computed rather than judged: lexical match with the gold page, selectivity (document frequency; metadata counts for location and type), documentary or not (follows from the attribute).

- **Phase A (correlational):** 1,000 dev hard questions on `balanced` pages; extract clues; fresh gold-page ranks (dense, BM25, hybrid; not the stored `rank_o`); relate hit@k to number, match, selectivity and type of clues, per language.
- **Phase B (causal, only if A shows a dependence):** compose variants from the extracted clues so phrasing stays constant (all clues / without quantity, date and record clues / location + type + one feature); score the gold rank plus an ambiguity bound from metadata; check that composed questions still fit the gold page.

Considered and dropped: a random-removal control (noisy), rewriting the original question (mixes style with detail), per-question realism labels (ambiguous).

## Blockers

- **API throughput:** the Vercel AI Gateway rejects most requests with an instant 503 for `inclusionai/ling-3.1-flash-free` (about 3 get through, then refusals, even at 5 per minute) and `poolside/laguna-s-2.1-free` (4 of 10; also ignores forced tool calls). The colleague who generated the data hits the same limits. Fixed on our side: pydantic's `$defs` schema caused a 400 with Ling, so the tool schema is inlined.
- **Local GPU speed:** the first gpt-oss run took ~20 s per question, but with Ollama's 131k-token context; the 4,096-token limit is in the code and not yet measured.
- **Embeddings:** Qwen is complete at 256 and 512 tokens, only 51% at 1024 and 8% at 2048, so larger chunk sizes cannot be compared fairly yet.

---

## Details

### What exists

| file | does |
|---|---|
| [`extract.py`](extract.py) | sample + clue extraction; backend `ollama` (local, default `gpt-oss:20b`, `--num-ctx 4096`) or `ling` (Vercel gateway, tool calling, answers cached in `out/ling-cache.jsonl` and retried in passes) |
| [`ranks.py`](ranks.py) | gold-page ranks for the same sample: `dense_qwen`, `bm25`, `hybrid_qwen` (`--methods`), chunk size from `CHUNK_SIZE` |
| [`prompts/clue_extract.md`](../../prompts/clue_extract.md) | the extraction prompt |
| `out/` (git-ignored) | `clues-*.jsonl`, `ranks-<size>.jsonl`, `ling-cache.jsonl` (12 of 30), `artifacts/` (local query-embedding cache; `/data` is read-only) |

```
uv run python -m experiments.clue_ablation.extract --n 30                       # local GPU
uv run python -m experiments.clue_ablation.extract --backend ling --n 30        # Vercel, needs VERCEL_API_KEY in .env
CHUNK_SIZE=512 uv run python -m experiments.clue_ablation.ranks --n 30
```

### Pilot numbers

Clue extraction (gpt-oss:20b, 30 questions): 151 clues, 148 quoted exactly (a literal substring check: copied, not paraphrased; split and classes not yet checked by a human). Attributes: location 41, type 30, feature 22, quantity 21, date 17, event 14, other 6. Two extractors on 20 other questions: total counts identical on 10, within one on 18; documentary counts identical on 14, within one on 19.

Retrieval on the same 30 hard questions, hit@10:

| retriever | 512-token chunks | 256-token chunks |
|---|---|---|
| dense, Qwen3-Embedding-0.6B | 33% | 33% |
| BM25 (pipeline) | 83% | 87% |
| hybrid (0.7 dense + 0.3 BM25) | 77% | 63% |

Dense on 30 easy questions: 30/30 in the top 10, so the setup is sound. On hard questions dense left 14 of 30 out of the top 100; BM25 ranked 22 first.

### Next steps (when unparked)

1. Measure gpt-oss speed with `--num-ctx 4096` on the 30 pilot questions (when `nvidia-smi` shows the GPU free).
2. Choose the sample size from that speed: 300 questions or 1,000 (overnight).
3. Human check: count clues on ~30 items, compare with the extraction.
4. Compute per-clue lexical match and selectivity; run phase A.
5. Decide on phase B; rerun ranks at 1024 / 2048 once embedded.
