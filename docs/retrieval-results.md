# Retrieval Eval Results

Historical retrieval results over 726 page chunks and 3,476 approved questions.
Historical method names map to current vocabulary: `bm25` means `sparse`,
`*_chunk` maps to the provider name, and `*_bm25` maps to `*_hybrid`.

## Reads

- Azure weighted hybrid was strongest measured method.
- Qwen3-Embedding-4B was strongest measured local dense provider. It was
  feasible on RTX 5070 Ti 16 GB with batch size 1 and max sequence length 512.
- Qwen 4B nearly reached Azure on crosslingual questions.
- English MiniLM was too weak for this multilingual corpus.
- Weighted sparse+dense fusion beat RRF hybrid in measured runs. Default hybrid
  weight is 70% vector, 30% sparse.
- Summary-only was consistently weaker than chunk-only.
- Chunk+summary measured well, but added extra derived content and operational
  cost. Current steady-state indexes title, heading path, and chunk text.
- Qwen3 reranking was harmful and slow in the measured setup. A 50-candidate
  rerank sample still performed poorly, so candidate depth was not the obvious
  issue. Diagnose model usage, prompt, and input formatting before enabling
  rerank by default.

## Key Results

| historical method         | current equivalent | hit@1 | hit@5 | hit@10 | mrr@10 |
|---------------------------|--------------------|------:|------:|-------:|-------:|
| bm25                      | sparse             | 0.544 | 0.668 | 0.705  | 0.596  |
| qwen_chunk                | qwen               | 0.512 | 0.719 | 0.795  | 0.602  |
| qwen_chunk_summary_bm25   | qwen_hybrid        | 0.603 | 0.794 | 0.851  | 0.685  |
| qwen4b_chunk_summary      | qwen4b             | 0.641 | 0.837 | 0.891  | 0.726  |
| azure_chunk               | azure              | 0.669 | 0.861 | 0.905  | 0.752  |
| azure_chunk_summary_bm25  | azure_hybrid       | 0.726 | 0.893 | 0.930  | 0.797  |

Crosslingual subset, 711 questions:

| historical method        | current equivalent | hit@1 | hit@5 | hit@10 | mrr@10 |
|--------------------------|--------------------|------:|------:|-------:|-------:|
| bm25                     | sparse             | 0.145 | 0.250 | 0.297  | 0.190  |
| qwen4b_chunk_summary     | qwen4b             | 0.608 | 0.851 | 0.911  | 0.710  |
| azure_chunk_summary      | azure              | 0.599 | 0.871 | 0.928  | 0.715  |

Rerank sample, 50 questions:

| family       | best non-rerank hit@5 | best rerank hit@5 |
|--------------|----------------------:|------------------:|
| Qwen vector  | 0.680                 | 0.260             |
| Qwen hybrid  | 0.680                 | 0.240             |
| Azure vector | 0.880                 | 0.220             |
| Azure hybrid | 0.680                 | 0.220             |
