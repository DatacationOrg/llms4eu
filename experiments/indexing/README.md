# Indexing Experiments

Small runnable comparisons for page-chunk retrieval. These scripts use the
approved eval questions in `data/db/pages.db`, reusable retrievers from
`src/retrieval`, and metrics from `src/eval`.

```bash
uv run python experiments/indexing/evaluate_embedding_models.py
uv run python experiments/indexing/evaluate_embedding_models.py --category crosslingual
uv run python experiments/indexing/evaluate_embedding_models.py --providers qwen --limit 20

uv run python experiments/indexing/compare_qwen_modes.py
uv run python experiments/indexing/compare_qwen_modes.py --category crosslingual
uv run python experiments/indexing/compare_qwen_modes.py --limit 100
```

`evaluate_embedding_models.py` compares the embedding providers listed in
`src/indexing/config.yaml`. Missing vector collections are built through the
normal indexing path. Azure requires `.env` values and asks for typed approval
before building embeddings.

`config.yaml` sets the default warmup count. Warmup queries are excluded from
timing.

`compare_qwen_modes.py` compares qwen vector search, sparse hybrid search,
reranking, and hybrid plus reranking.
