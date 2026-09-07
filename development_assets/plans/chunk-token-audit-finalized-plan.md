# Chunk Token Audit Finalized Plan

## Goal

Add a reproducible, read-only audit that measures whether current character-sized Slovenian chunks are truncated by the exact locally cached Qwen 0.6B, Qwen 4B, and Nemotron embedding tokenizers. Scan every current chunk for operational risk, compare 30 matched Slovenian/English tourism passages if trustworthy reviewed translations can be supplied, and make an evidence-based recommendation without changing chunk boundaries or IDs.

## Current Behavior

- `src/preprocess/chunks.py::chunk_markdown` groups Markdown by heading and paragraph, targeting 1,800 characters with a 2,600-character long-paragraph threshold and 300-character minimum for multi-chunk pages.
- Splits prefer `. `, then whitespace, then a hard character split. There is no overlap and no token-aware splitting.
- Sub-300-character chunks are dropped when a page produces multiple chunks. This is content loss, separate from model truncation.
- V1 embedding input adds title and heading; V2 adds labeled metadata. Both can be longer than canonical chunk text.
- Qwen 0.6B and Qwen 4B use a 512-token configured cap. Nemotron uses 4,096.
- Rechunking is out of scope because positional chunk IDs are referenced by evaluation labels.

## Implementation

1. Expose the existing SQLite chunk loader in `src/vector_store/chunks.py` as a public read-only function. Continue using it in production indexing.
2. Add `src/indexing/token_audit.py` with typed records and functions for provider limits, per-text measurements, grouped distributions, truncation summaries, paired-language comparisons, fixture validation, and deterministic Markdown-ready data.
3. Reuse `build_indexer` and `chunk_text_representation`. Inspect each selected SentenceTransformer model's actual tokenizer and document prompt. Count prompt plus content plus special tokens with truncation disabled. Compare against configured/effective `max_seq_length`. Do not compute embeddings.
4. Aggregate the full SQLite corpus by provider, representation version, and source language. Report count, character/token p50/p90/p95/p99/max, tokens per character, over-limit count/rate, total/max lost tokens, overhead, and worst chunk IDs.
5. Add a checked-in JSONL fixture under `experiments/indexing/data/sl_en_tourism_pairs.jsonl` containing 30 representative Slovenian passages and English translations with unique IDs and source provenance only if translations can truthfully be described as reviewed. Never claim machine-generated translations are reviewed. If reviewed translations are unavailable, implement fixture format/validation and clearly leave the fixture/report portion blocked rather than fabricating review status.
6. Analyze language pairs without source-only metadata. Report per-provider paired token counts, characters per token/tokens per character, ratios, and distribution summaries. Explain that fewer characters per token means more tokens for a fixed character budget.
7. Add `experiments/indexing/audit_chunk_tokens.py`, following repository argparse/env/report patterns. Defaults: providers `qwen,qwen4b,nemotron`, versions `v1,v2`, current DB, checked-in pair fixture, explicit dated docs output. Support filters/output. Never download implicitly; fail clearly if local models are absent.
8. Add a `just` recipe using `uv`.
9. Generate `docs/chunk-token-audit-2026-08-11.md` if actual cached tokenizers and required fixture are available. Record model names, limits, chunk config, command, corpus tables, language tables, worst examples, and recommendation.
10. Update indexing/preprocessing READMEs with audit command and semantics.
11. Add `tests/test_chunk_token_audit.py` using fake tokenizer/model adapters for prompt/special token accounting, exact/over-limit behavior, lost tokens, missing-language grouping, paired ratios, deterministic ordering, and malformed fixtures.
12. Extend `tests/test_preprocess_chunks.py` to document no overlap, hard-split boundary behavior including possible one-character overshoot, and short-fragment dropping without changing production behavior.

## Recommendation Rules

- Keep current chunking if over-limit rates and lost-token tails are negligible.
- Recommend a lower shared character target only if one conservative threshold works across providers.
- Otherwise recommend a future versioned/provider-aware token-budget experiment.
- Treat no-overlap as a boundary-recall issue and short-fragment dropping as content loss, not tokenizer truncation.

## Scope Boundaries

Included: tokenizer efficiency, effective embedding-input length, implicit embedding truncation, V1/V2 overhead, current splitter characterization, and measured recommendation.

Excluded: changing chunk boundaries, adding overlap, migrating IDs/qrels, rebuilding Chroma, retrieval benchmarking, reranker/generator audits, runtime translation, automatic model downloads, commits, and pushes.

## Validation

1. `uv run --extra dev pytest tests/test_chunk_token_audit.py tests/test_preprocess_chunks.py`
2. `uv run ruff format` on touched Python files.
3. `uv run ruff check` on touched Python files.
4. `uv run --extra dev pytest`
5. Run the new audit recipe for all selected providers and both versions when local caches permit.
6. Verify worst IDs resolve in SQLite and reruns produce stable analytical tables apart from declared run metadata.
