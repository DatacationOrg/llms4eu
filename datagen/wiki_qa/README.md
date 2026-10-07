# Wiki QA generation scripts (record)

The scripts that built `qa/*.parquet`, copied as they ran from the git-excluded
workspace `tmp_labeling/` on thebeast; not maintained, not linted. Run from this
folder (they import each other by name and read `/data/llms4eu/wiki/pages.jsonl`).
The full method, models and settings per stage: `docs/reports/wiki-qgen/wiki-qa-dataset.md`;
the day-by-day decisions: `CLEANING_LOG.md`; the workspace's own file guide:
`README.tmp_labeling.md`.

`bunny.py` is the shared free-model client: structured output (JSON schema for
Space Bunny on OpenRouter, tool calling for Ling 3.1 Flash on Vercel), request
pacing, and 429/503 handling. Keys come from `.env` (`OPENROUTER_API_KEY`,
`VERCEL_API_KEY`). Results go to one working SQLite file, `wiki_qa.db`; every row
records the model that made it. `run_*.sh` loop each step under `watchdog.sh`.

| # | stage | scripts |
|---|---|---|
| 1 | page tags | `tag_spec.py`, `train_lora.py --task cls`, `run_vllm.py` |
| 2 | corpus questions | `gen_questions.py`, `train_lora.py --task qg`, `run_vllm.py` |
| 3 | rule checks | `rule_checks.py` |
| 4 | quality labels, repairs, relabels | `qspec.py`, `bunny_label.py`, `bunny_fix.py`, `bunny_verify.py`, `make_queue.py` |
| 5 | cross-page duplicates | `near_dups.py` |
| 6 | hard questions | `bm25.py`, `gen_challenge.py`, `dense_rank.py rank` |
| 7 | answers, evidence, cross-lingual | `gen_rag.py gen` |
| 8 | answer judges | `gen_rag.py judge`, `qspec.RAG` |
| 9 | extra layers | `gen_extra.py`, `tables.py` |
| 10 | multi-page relevance | `gen_extra.py qrels` |
| 11 | ranks, tags, splits | `dense_rank.py rag`, `build_clean.py` |
| 12 | export | `build_clean.py` |

Stage 0, the corpus, is `../corpus/wiki_places.py`. The LoRA adapters
(`qwen35_qg_s1000/`, `e4b_cls_s0600/`), training sets (`sft_*.jsonl`) and blind-check
folders (`xcheck/`, ...) are in `/data/llms4eu/wiki/datagen/wiki_qa/`.
