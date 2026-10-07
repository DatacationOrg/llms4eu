# Indexing

`python -m src.indexing --method <provider>` (`just index qwen`) embeds the chunks of
every size into `embeddings/<provider>/<size>.npy` under the dataset folder:
float16 unit vectors, row i = chunk i of `load(Chunk, size=size)`, NaN until
embedded. Only missing rows are embedded, so a stopped run resumes and a rechunk
re-embeds only the chunks that changed. `--api` embeds on OpenRouter's free
endpoint instead of the GPU, for providers listed under `api` in `config.yaml`
(Nemotron 3 Embed 1B, the same weights; 1000 requests a day, waited out).

Providers are the entries under `providers` in `config.yaml` (Qwen3-Embedding
0.6B/4B/8B, Nemotron 3 Embed 1B/8B), run through sentence-transformers; the folder
name is the provider name. Documents get the model's own `document` prompt (none for
Qwen3, `passage: ` for Nemotron), queries its `query` prompt.

`store.py` answers vector queries for the CHUNK_SIZE in use: exact dot-product search
over the stored vectors. An approximate index (FAISS, Qdrant) built from the same
files is the next step when query latency matters. Query embeddings are cached in
`$LLMS4EU_DATA/embeddings/cache.sqlite`.
