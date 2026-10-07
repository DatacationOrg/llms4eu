#!/usr/bin/env bash
# Corpus question pass: Qwen3.5-4B + qwen35_qg_s1000 over pages.jsonl in gen_questions.queue order (short pages
# first, SE lakes and machine-made pages last) into wiki_qa.db table questions. Resumable: rerun to continue.
# Usage: ./run_qg_corpus.sh [UTC deadline, e.g. 2026-10-01T06:00]   (background; progress in corpus_qg.log)
cd "$(dirname "$0")"
setsid nohup uv run --with vllm==0.30.0 --with langchain-openai python run_vllm.py \
  --qg qwen35_qg_s1000 --db wiki_qa.db --chunk 2000 ${1:+--deadline "$1"} >> corpus_qg.log 2>&1 &
echo "started pid $! (log: $PWD/corpus_qg.log)"
