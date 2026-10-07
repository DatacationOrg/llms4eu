# tmp_labeling (git-excluded)

**The dataset is in `wiki_qa_dataset/`: start with its `README.md`** (Parquet files, guide, counts, checksums).
Everything else here is the workspace that builds it (scripts, runners, logs, `wiki_qa.db`, `backup/`).

Page tags and RAG test questions for `/data/llms4eu/wiki/pages.jsonl`. Findings:
`docs/reports/wiki-qgen/wiki-labeling-qgen-2026-09-30.md`.

| file | what |
|---|---|
| `wiki_qa.db` | results: `labels` (8 tags, all 133,164 pages), `questions` (corpus pass); `models` maps each row's `model` to a readable name and description |
| `run_qg_corpus.sh` | starts the corpus question pass (resumable, optional UTC deadline) |
| `run_vllm.py` | offline vLLM + LoRA with json_schema decoding, on a pool file or the whole corpus |
| `train_lora.py` | LoRA r=8 trainer (`--task qg` questions, `--task cls` tags) |
| `gen_questions.py` | question prompt + schema, corpus queue order, the OpenRouter teacher |
| `tag_spec.py` | the 8 tags: teacher prompt, classifier prompt, schema |
| `qg_checks.py` | automatic checks on generated questions (language, English query, answer leak) |
| `fix2_batches.py`, `fix2_check.py`, `fix2/RULES.md` | agent cleaning of new teacher items into `qg_clean.jsonl` |
| `qwen35_qg_s1000/` | final question LoRA (Qwen/Qwen3.5-4B, 1000 steps) |
| `e4b_cls_s0600/` | tag classifier LoRA (google/gemma-4-E4B-it, 600 steps) |
| `qg_clean.jsonl`, `labels.jsonl`, `label_pool*.json` | training data: cleaned question items, teacher tags, article texts |
| `questions_pool*.db` | raw teacher question output |
| `qg_test.json`, `qg_hard20.json`, `bench100.json` | eval pools: 120 held-out, 20 hard-language held-out, 100 random pages |

## Cleaning (from 2026-10-02; log of decisions and results: `CLEANING_LOG.md`)

Two free models share the work: Space Bunny (OpenRouter, `CLEAN_MODEL=bunny`) and Ling 3.1 Flash (Vercel AI Gateway,
`CLEAN_MODEL=ling`); every row stores its model. Originals in `questions` are never changed.

| file | what |
|---|---|
| `wiki_qa.db` new tables | `checks` (rule flags), `bunny_labels` (qspec labels per item), `repairs` (fixed/replaced items), `repair_labels` (relabel of the repaired page), `extra_items` (described/multi), `challenge_items` (BM25-hard items + answer + labels), `challenge_qwen` (LoRA attempt, unused), `answers` (answer, verbatim evidence, qtype, cross-lingual pair per kept item), `answer_labels` / `answer_labels_ling` (`qspec.RAG` criteria + x_ok, by Bunny / Ling), `rag_items` (superseded pilot) |
| `bunny.py` | Bunny client: pacing ≤50 rpm, 429 stop, 32k token cap, `pages()`, `run_all()` |
| `bunny_label.py`, `bunny_fix.py`, `bunny_verify.py` | label → repair (fix / replace) → verify, resumable from the DB |
| `run_bunny_clean.sh`, `run_bunny_clean_b.sh`, `run_ling_clean.sh`, `watchdog.sh` | the 24/7 runners (Bunny: odd / even corpus chunks of `corpus_queue.txt`; Ling: from the end) and their watchdog |
| `rule_checks.py`, `sample_ids.py`, `make_queue.py`, `label_dist.py` | free rule flags, teacher sample, priority queue, label distributions |
| `bm25.py` | BM25 over all pages: rank of a question's own page, distinctive words |
| `gen_challenge.py`, `run_challenge.sh`, `run_challenge_ling.sh` | challenging (vague, standalone) items with answers, labelled for filtering; Bunny from the front of the queue, Ling `--reverse` |
| `gen_rag.py`, `run_rag.sh`, `run_rag_bunny.sh` | answers, evidence spans, question type and a cross-lingual version per kept item (Ling), judged by Ling and Bunny |
| `dense_rank.py` | Nemotron 1B dense rank of the own page: `rank` (challenge items), `rag` (all answered questions + BM25 → `rag_rank.json`) |
| `compare_labellers.py` | two labellers against gold val |
| `gen_hard.py` | described / multi items (≤1k, training data only) |
| `build_clean.py` | exports (as `.parquet`; JSONL copies to `legacy_jsonl/`) `wiki_qa_clean` (per item the verified repair or the original, labels, provenance, `ambiguous_across_pages`), `wiki_qa_challenge`, `wiki_qa_rag` (the RAG test set, see `rag_rows`) |
| `gen_extra.py`, `run_extra.sh` | Ling layers: query variants + tags, unanswerable, two-page comparison, list/geo questions from metadata, computed table questions, multi-page qrels (tables `table_q`, `variants`, `unanswerable`, `compare`, `meta_q`, `qrels`) |
| `tables.py` | Markdown data-table finder and number parser for computed table questions (`gen_extra.py tables` → table `table_q`) |
| `wiki_qa_dataset/` | **the release**: `README.md` (user guide per question group: target, build, how to find the answer, caveats), `manifest.json` (counts), `SHA256SUMS`, the `wiki_qa_*.parquet` exports, `legacy_jsonl/`; current counts |
| `backup/` | DB and export snapshots taken before the 2026-10-04 extension |
| `build_sft.py` | LoRA sets `sft_qg` / `sft_fix` / `sft_challenge` (`train_lora.py --task file --data …`) |
| `gen_challenge_vllm.py` | challenge items from the Qwen LoRA (tried, not good enough) |

Reproduce the question LoRA: `uv run --with peft==0.21.1 --with transformers==5.17.0 --with torch==2.14.0
--with torchvision --with langchain-openai python train_lora.py --task qg --steps 1000 --save-every 500 --ckpt`.
