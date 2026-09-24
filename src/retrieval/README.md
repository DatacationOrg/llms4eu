# Retrieval

Reusable chunk retrieval methods.

`methods.py` is the public catalog for eval and RAG callers:
`list_retrievers`, `build_retriever`, and `ensure_retriever_ready`.

`base.py` defines `RankedChunk` and the `Retriever` protocol.

`retrievers/` contains concrete methods. Every method runs locally; no
retrieval path calls a hosted API.

- `sparse`: local lexical retrieval over page chunks.
- `<provider>`: Chroma vector search over page chunks, one per entry in
  `providers` of `src/indexing/config.yaml`.
- `*_hybrid`: normalized weighted score fusion over one vector provider plus
  sparse retrieval.
- `*_rerank`, `*_rerank_4b`: Qwen3 0.6B or 4B cross-encoder over first-stage
  candidates; `*_hybrid_rerank*` over hybrid candidates. Each entry in
  `rerankers` of `config.yaml` adds one suffix.

- `*_hybrid_rerank_geo`: the hybrid-rerank stage re-scored by distance to the
  place the question names (`retrievers/geo.py`). The local model names the
  place and how wide it is; Wikidata gives the point; over 4x candidates each
  score becomes `0.7 * text + 0.3 * exp(-km / decay)`, and a page with no
  location scores 1.0 on geography. Places are cached per query in
  `geo_query_places`; a lookup failure leaves the query unscoped. Needs
  `just locate-pages`. Settings are the `geo_*` keys in `config.yaml`.

Hybrid + rerank is the strongest measured configuration. The agentic
sufficiency and query-reformulation loop was removed after it measured below
that baseline while spending an LLM call per attempt; the findings are in
[`docs/reports/agentic/agentic-findings.md`](../../docs/reports/agentic/agentic-findings.md).

Retrieval tuning lives in `config.yaml`. Enabled vector providers come from
`src/indexing/config.yaml`; eval defaults come from `src/eval/config.yaml`.

Sparse retrieval uses BM25. `sparse_k1` controls repeated-term saturation, and
`sparse_b` controls length normalization.
