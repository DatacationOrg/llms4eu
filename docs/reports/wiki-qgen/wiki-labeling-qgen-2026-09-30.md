# Wiki places corpus: page tags and RAG test questions (2026-09-29/30)

Goal: for every page of the Wikipedia places corpus (`/data/llms4eu/wiki/pages.jsonl`, 133,164
pages, 24 EU languages plus a few `tr`/`en`), (1) tag what kind of page it is and (2) generate up to
five search-test items: a question a chat user without the article would ask, its English
translation, 1-3 short facts that answer it, and English embedder keywords. Both with small local
models fine-tuned (LoRA r=8) on teacher labels, so the 133k pages run on the shared A6000.

Working files live in `tmp_labeling/` (git-excluded); see its `README.md`.

## Teacher data

- Teacher: `stealth/space-bunny-alpha` (OpenRouter free tier, LangChain `json_schema` structured
  output). Its free-request counter on the key never moved (`free_model_daily_requests.used` stayed 0),
  so budget by your own count; 1000/day is the documented cap.
- Tags: 547 labelled articles (437 train, 110 held-out), language-balanced.
- Questions: 1,028 teacher articles in three pools (260 first pool, 505 sqrt-language-balanced with
  ~10% machine-made Swedish lakes, 263 top-up so every language has >= 30 train articles).
- All question items were reviewed by Sonnet agents against the article text under
  `tmp_labeling/fix2/RULES.md`. The teacher's main faults, all fixed by the agents: facts written in
  English for non-English articles, garbled tokens (stray CJK/Cyrillic, "subsystems l Everywhere"),
  facts not in the text, answers leaking into queries, trivial register questions (basin codes,
  coordinates), and ambiguous names (hundreds of same-named lakes). Items dropped from ~4.8 to ~3.4
  per article; stubs got 1-2 questions. Result: 1,025 articles (905 train, 120 held-out) in
  `qg_clean.jsonl`.
- Agents running in parallel must keep scratch files in their own folder: several collided in a
  shared `/tmp` path (outputs were verified per-folder and by a fact/text word-overlap check: 41 of
  2,229 items below 40% overlap, all paraphrases or infobox numbers).

## Page tags (done for all 133,164 pages)

Eight tags from `tag_spec.py`: article_type, information_richness, tourist_appeal, primary_focus,
visitor_info, history, nature, culture. Agreement with the teacher on 110 held-out articles:

| model | type | richness | tourist | focus | visitor | history | nature | culture | mean |
|---|---|---|---|---|---|---|---|---|---|
| Laya-multilingual zero-shot | 0.33 | 0.31 | 0.18 | 0.30 | 0.70 | 0.67 | 0.51 | 0.81 | 0.476 |
| Laya + LoRA (6 epochs) | 0.91 | 0.81 | 0.65 | 0.81 | 0.86 | 0.86 | 0.83 | 0.89 | 0.827 |
| Gemma 4 E4B zero-shot | 0.65 | 0.67 | 0.78 | 0.81 | 0.87 | 0.87 | 0.78 | 0.87 | 0.789 |
| **Gemma 4 E4B + LoRA, 600 steps** | 0.94 | 0.84 | 0.81 | 0.90 | 0.90 | 0.88 | 0.85 | 0.92 | **0.880** |

Earlier, on 60 articles: Jev (API, not fine-tunable) 0.84. Decider-4B and GLiNER2.5 were not
fine-tuned (GLiNER far behind zero-shot; Decider needs one prefill per question, E4B answers all eight
in one ~60-token JSON). The E4B classifier ran over the corpus in ~5 h (0.04 s/page on short pages,
0.22 s on long ones), 0 errors, into `wiki_qa.db` table `labels`:

| tag | distribution |
|---|---|
| article_type | registry_stub 74,092 · descriptive 59,053 · list/disambiguation 19 |
| information_richness | low 69,510 · medium 39,098 · high 24,556 |
| tourist_appeal | low 87,580 · medium 30,892 · high 14,692 |
| primary_focus | data 74,236 · description 44,207 · history 12,976 · protection 1,399 · visiting 346 |
| visitor_info / history / nature / culture (true) | 26,123 / 35,367 / 88,933 / 26,190 |

## Question generation

Prompt and schema: `tmp_labeling/gen_questions.py` (`PROMPT`, `Questions`). Rules that mattered:
ask only what a real person would ask (never codes, IDs, coordinates), scale the count to the
article (stub 1-2, rich up to 5), name or describe the place so only it fits among 100k pages,
paraphrase, facts from the article, query from the question only. `question_en` was added so both
native and English queries can be tested.

Cheap automatic checks on 120 held-out articles (`qg_checks.py`; language by lingua, "leak" = a
fact word/number not in the question appears in the query):

| model (vLLM, json_schema) | items/art | question in article lang | English query | query leaks answer | s/article |
|---|---|---|---|---|---|
| cleaned teacher | 3.1 | 96% | 80% | 3% | - |
| Gemma 4 E4B zero-shot | 2.5 | 95% | 1% | 65% | 0.36 |
| Gemma 4 26B-A4B NVFP4 zero-shot | 2.6 | 95% | 91% | 29% | 0.49 |
| E4B + LoRA, 700 steps (545 train) | 3.2 | 96% | 80% | 4% | 0.36-0.56 |
| E4B + LoRA, 1000 steps (905 train) | 3.3 | 94% | 77% | 3% | 0.30 |
| **Qwen3.5-4B + LoRA, 1000 steps (905 train)** | 3.3 | 95% | 77% | 4% | 0.36 |

The numbers tie; reading 20 held-out articles in hard languages (el, lv, bg, lt, et, sl, lb, hr,
hu, ga, mt) decides it. **Qwen3.5-4B is the final model** (`qwen35_qg_s1000/`):

- E4B at 1000 steps: broken, repetitive Luxembourgish ("Wéi ass d'Buerg Hesper … ginn?" x5),
  garbled Lithuanian, hallucinated facts (Jakucs cave "in the Bakony", from a caving club's name;
  Schlass Bierg "230 m high"), duplicate questions, translation errors ("kreisajā" left bank as
  right bank; "romarska cerkev sv. Križa" as "Romanesque church of St. George").
- Qwen3.5-4B: fluent, varied Luxembourgish and Lithuanian, no hallucinations or duplicates found,
  better translations. Left: some Greek/Bulgarian queries keep native-script names, small slips
  ("Ποια βουνό", "Bunje" for Bunić, "sandbar" for wetland).

Training (transformers 5.17, peft 0.21.1, torch 2.14, batch 1, lr 2e-4 OneCycle, gradient
checkpointing, loss on answer tokens only): E4B ~1.5 s/step, 21 GB; Qwen3.5-4B 1.5-2.7 s/step,
13 GB, without the `flash-linear-attention`/`causal-conv1d` kernels (installing them should speed
training up; vLLM inference has its own kernels).

Speed of the final model on 100 random corpus pages (bf16, vLLM 0.30, 256 sequences): 0.36 s/page;
0.34 s/page on 1,000. So ~7-8 h for the 77k human-written pages, ~13 h for all 133k.

FP8 does not help on the A6000: vLLM 0.30's default FP8 path (CUTLASS W8A8) crashes on Ampere (no
FP8 tensor cores); forced weight-only Marlin (`kernel_config={"linear_backend": "marlin"}`) runs at
0.40 s/page vs 0.36 bf16, same quality (valid 100/100, 97% language, 2% vs 3% leaks; 213/353
questions word-identical).

## Dead ends and version notes

- **DiffusionGemma-26B-A4B** (block diffusion): fast on short stubs (~0.7 s/article at 10 parallel)
  but vLLM refuses JSON-schema decoding for diffusion models (vllm#45436 was closed by #45468, which
  rejects it with a clear error; real structured diffusion, #57250, merged after 0.30.0 and covers
  choice/yes-no/score only) and has no LoRA support. A LoRA trained in transformers (FP8 experts,
  ~16 s/article) fixed language (45% → 93% in article language) but kept corrupting words
  ("Tervajvi", "kilometä") and produced valid JSON for 14/20. Dropped.
- **Gemma 4 26B-A4B**: strongest zero-shot (91% English queries) at 0.49 s/article, but vLLM 0.30
  cannot serve a LoRA on the Gemma 4 MoE (`get_expert_mapping` not implemented; PRs #46772, #50252
  open) and bf16 weights (52 GB) exceed the card for training. Would need QLoRA, merge, re-quantize.
- vLLM was pinned at 0.24.0 by habit for most of the night; 0.30.0 (2026-09-22) was the latest and
  gave the same speed for E4B + LoRA (0.51 vs 0.56 s on 120 articles). None of the three blockers
  above is fixed in 0.30.
