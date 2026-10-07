# Clue ablation: how much does retrieval depend on the detail in a question?

**Status (2026-10-07): parked after the pilot.** Phase A tooling works on 30 questions; the full extraction is
blocked on LLM throughput (see [Blockers](#blockers)).

## Question

Hard questions in the wiki QA set describe a place without naming it, with about **5 clues each** (pilot). They were
generated to be unique among 133k pages, so the generator stacked details: exact years, areas, records
([audit findings 11-12](../../docs/reports/wiki-audit/audit-2026-10-06.md)). A tourist gives fewer, vaguer clues.

Instead of labelling questions as realistic or not (tried; too ambiguous), treat **detail as an experimental
variable**: how does retrieval change when the same information need carries fewer clues, or other kinds of clues?

What this measures: the effect of the **amount and type of information** in a question. What it does not measure:
vocabulary mismatch (the clues stay page wording) or real user behaviour.

## Design

**A clue** is one constraint on the place: one attribute with one value, **quoted literally** from the question
(checkable by string match). Attributes: `name`, `location`, `type`, `feature`, `quantity`, `date`, `event`,
`other` ([prompt](../../prompts/clue_extract.md)). Per clue, computed rather than judged:

- lexical match: do its content words occur in the gold page?
- selectivity: document frequency of its words; for location and type, metadata counts
- documentary or not: follows from the attribute (`quantity`, `date`, records)

**Phase A (correlational, cheap):**

1. Sample: 1,000 dev hard questions on `balanced` pages (<= 400 pages per language), fixed hash order.
2. Extract clues (LLM, literal quotes); check against a human count on ~30 items.
3. Fresh gold-page ranks with the pipeline's retrievers: dense (Qwen3-Embedding-0.6B), BM25, hybrid, on the
   512-token chunks (and other sizes once embedded). The stored `rank_o` dense ranks are not used (lead-only index,
   a third of the rows).
4. Relate hit@k to number of clues, number of lexically matching clues, selectivity and attribute class, per
   language.

**Phase B (causal, only if A shows a dependence):** compose variants **from the extracted clues** (so phrasing is
held constant): all clues / without `quantity` + `date` + record clues / location + type + one feature. Score the
gold rank plus an ambiguity bound from metadata (pages with the same type and location); judge fitting pages only
on a calibration sample. Guards: check the composed question still fits the gold page; stratify by language and
baseline rank.

Considered and dropped: a random-removal control (noisy, little insight); rewriting the original question (mixes
style with detail); classifying realism per question (ambiguous, see audit finding 12).

## What exists

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

## Results so far (pilot, n = 30: signals, not results)

**Clue extraction (gpt-oss:20b):**

- 98% of clues quoted exactly: 148 of 151, measured as a literal substring check. This says the model copied rather
  than paraphrased; split and classes are not checked yet.
- Mean 5.0 clues per question, range 3-7. Attributes: location 41, type 30, feature 22, quantity 21, date 17,
  event 14, other 6.
- Two LLMs' clue counts on 20 other questions agreed exactly on 10, within one on 18; documentary counts agreed
  better (14 exact, 19 within one). Segmentation is the main source of difference, hence the literal-quote rule.

**Retrieval on the same 30 hard questions, hit@10:**

| retriever | 512-token chunks | 256-token chunks |
|---|---|---|
| dense, Qwen3-Embedding-0.6B | 33% | 33% |
| BM25 | 83% | 87% |
| hybrid (0.7 dense + 0.3 BM25) | 77% | 63% |

- Dense on 30 **easy** questions: 30/30 in the top 10, so the setup is sound.
- On hard questions, dense left 14 of 30 out of the top 100; BM25 ranked 22 first.
- Hybrid is below BM25: the default weights favour dense.

**Inferences to test:** BM25 lives off the exact anchors that survived the banned-word step (years, numbers, names);
dense may fail because clues are spread over chunks (not supported so far: 256 = 512) or because many places look
alike. The pipeline's default hybrid weights may not suit described-place questions. Worth telling the team
before the chunk-size comparison, once confirmed on more questions.

## Blockers

- **API throughput.** The Vercel AI Gateway now rejects most requests with an instant 503 for both
  `inclusionai/ling-3.1-flash-free` (about 3 requests get through, then refusals, even at 5 per minute) and
  `poolside/laguna-s-2.1-free` (4 of 10; also ignores forced tool calls). The colleague who used it for generation
  hits the same limits now. Fixed on our side: pydantic's `$defs` schema caused a 400 with Ling, so the tool schema
  is inlined.
- **Local GPU speed.** The first gpt-oss run took ~20 s per question on the shared GPU, but it ran with Ollama's
  131k-token context. The 4,096-token limit is in the code now and **not yet measured**.
- **Embeddings.** Qwen is complete at 256 and 512 tokens, but only 51% at 1024 and 8% at 2048 (nemotron: 10% at
  1024, 83% at 2048). Larger chunk sizes cannot be compared fairly until they are complete.

## Next steps

1. Measure gpt-oss speed with `--num-ctx 4096` on the 30 pilot questions (run when `nvidia-smi` shows the GPU free).
2. Choose the sample size from that speed: 300 questions (about 1.5-2 h at the old speed) or 1,000 (overnight).
3. Human check: count clues on ~30 items, compare with the extraction.
4. Compute per-clue lexical match and selectivity; run phase A step 4.
5. Decide on phase B from the result; rerun ranks at 1024 / 2048 once embedded.
