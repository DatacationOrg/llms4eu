# Retrieval

Reusable chunk retrieval methods.

`methods.py` is the public catalog for eval and RAG callers:
`list_retrievers`, `build_retriever`, and `ensure_retriever_ready`.

`base.py` defines `RankedChunk` and the `Retriever` protocol.

`retrievers/` contains concrete methods:

Every method runs locally; no retrieval path calls a hosted API.

- `sparse`: local lexical retrieval over page chunks.
- `{english,qwen,qwen4b,nemotron}`: Chroma vector search over page chunks.
- `*_hybrid`: normalized weighted score fusion over one vector provider plus
  sparse retrieval.
- `*_rerank`: Qwen3 cross-encoder reranker over first-stage candidates.
- `*_hybrid_rerank`: Qwen3 reranker over hybrid candidates.
- `qwen_agentic`, `qwen_hybrid_agentic`, and `nemotron_hybrid_agentic`: the
  agentic sufficiency and query-reformulation loop over the provider's
  reranked chunks.
- `qwen_hybrid_agentic_tools` and `nemotron_hybrid_agentic_tools`: the same
  loop plus two page-navigation tools. Beyond `sufficient` / `reformulate` /
  `expand` the agent may call `list_sections(page_id)` to read a page's table
  of contents, and `search_in_page(page_id, term)` to pull sibling chunks the
  first stage missed. Chunks it finds are inserted directly after the ranked
  chunk from the same page, so the agent's choices are scored rather than
  parked at the end of the list. Tool calls are capped by
  `agentic_tools_max_tool_calls`.
- Every dense, hybrid, reranked, and agentic method also has a `_v2` variant,
  such as `nemotron_hybrid_rerank_v2` and
  `nemotron_hybrid_agentic_v2`. These use isolated metadata-context dense and
  sparse indexes. Unsuffixed names retain the legacy v1 behavior.
- `sparse_v2` and `sparse_rerank_v2` use the same metadata-context text as v2
  dense retrieval; v1 sparse methods continue to index raw chunk text.

Retrieval tuning lives in `config.yaml`. Enabled vector providers come from
`src/indexing/config.yaml`; eval defaults come from `src/eval/config.yaml`.
Agentic retrieval starts with 10 reranked chunks and expands in steps of 5 up
to the configured maximum when the judge requests more context.

The agentic sufficiency judge is a local Ollama model set by
`agentic_judge_model` in `config.yaml` (`gpt-oss:20b`, low reasoning effort,
structured output via tool calls). Ollama must be running and the model pulled
before any `*_agentic*` method works.

Reasoning effort is a benchmark dimension, not a fixed setting. Every name in
`agentic_reasoning_levels` generates a parallel method suffixed with it —
`qwen_hybrid_agentic_high`, `qwen_hybrid_agentic_tools_high`, `dci_high`,
`dci_k50_high` — running the same loop at that effort. The unsuffixed names keep
the configured default (`low`), so adding a rung cannot move a published number.
Sweep both rungs with `--methods reasoning` (agents) or `--methods reasoning-dci`.

This axis exists because the 2026-09-01 comparison ran the whole family at `low`
and its two worst results were a tool agent failing to emit valid JSON on 7% of
steps and a DCI agent exhausting its step budget without answering — failures
reasoning effort plausibly moves, which made "the tools do not help"
inseparable from "the judge was thinking as little as it is allowed to".

Sparse retrieval uses BM25. `sparse_k1` controls repeated-term saturation, and
`sparse_b` controls length normalization.

See [`experiments/indexing/README.md`](../../experiments/indexing/README.md) for workload,
latency, quality, and statistical requirements.
