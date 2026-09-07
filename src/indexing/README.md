# Indexing

Builds local vector indexes for document chunks.

The indexer boundary is provider-shaped: English MiniLM, Qwen multilingual,
Qwen 4B, and Nemotron 3 Embed 1B all expose the same `embed_documents` /
`embed_query` methods from `src.shared.indexers`. Every provider runs locally
through sentence-transformers.

Chunk vector collections are derived state in Chroma. SQLite remains the source
of truth for page metadata, chunk text, and eval labels. `src.vector_store.chunks`
owns Chroma mechanics and query-time vector search; indexing only orchestrates
rebuilds.

Current chunk collections are storage names, not public retrieval method names:

```text
page_chunks_{english,qwen,qwen4b,qwen8b,nemotron,nemotron8b,qwen_s512}_chunk
```

Two independently stored chunk-representation versions are available:

- `v1` is the unchanged legacy `TitleHeadingChunkText` representation and keeps
	the existing `page_chunks_{provider}_chunk` collection names.
- `v2` uses `MetadataContextChunkText`: labeled title, heading, source language,
	source collection, page kind, and chunk content. It writes separate
	`page_chunks_v2_{provider}_chunk` collections. The retrieved evidence remains
	the original chunk text.

Both versions use the same chunk boundaries and IDs. The version changes only
the text indexed for dense and sparse retrieval, allowing paired comparisons.

## Chunk variants

A *version* is how a chunk becomes embedding text; a *variant* is how pages were
cut into chunks in the first place. They are independent, and both are collection
suffixes with the historical case left empty:

```text
page_chunks_{provider}_chunk              # v1, base
page_chunks_v2_{provider}_chunk           # v2, base
page_chunks_tok512_{provider}_chunk       # v1, tok512 variant
page_chunks_v2_tok512_{provider}_chunk    # v2, tok512 variant
```

```bash
uv run python -m src.indexing.chunks --method qwen --chunk-version v2 --variant tok512
```

Readiness is scoped to the variant. `collection_ready` compares against that
variant's chunk count, not a global one, so building a second variant does not
make the first look stale.

## Providers

`config.yaml` is enough to add one: any `{name}_embedding_model` key makes
`{name}` a buildable provider, with optional `{name}_batch_size`,
`{name}_max_seq_length`, `{name}_local_files_only`, `{name}_dtype`,
`{name}_attn_implementation`, and `{name}_query_prompt` / `{name}_document_prompt`
for models whose saved sentence-transformers config declares no prompts.

**Every provider reads its model's full context.** `max_seq_length` is a
truncation ceiling, not an allocation: sentence-transformers pads each batch to
its own longest item, so raising the ceiling costs nothing on short inputs and
simply stops cutting the long ones. With nothing able to overflow it, chunk size
becomes the only variable a chunking ablation moves.

The configured lengths are each model's own declared capacity:

| provider | model | dims | length |
|---|---|---:|---:|
| `qwen` | Qwen3-Embedding-0.6B | 1,024 | 32,768 |
| `qwen4b` | Qwen3-Embedding-4B | 2,560 | 40,960 |
| `qwen8b` | Qwen3-Embedding-8B | 4,096 | 32,768 |
| `nemotron` | Nemotron-3-Embed-1B | 2,048 | 32,768 |
| `nemotron8b` | Nemotron-3-Embed-8B | 4,096 | 32,768 |
| `english` | all-MiniLM-L6-v2 | 384 | 256 (a real hard limit) |
| `qwen_s512` | Qwen3-Embedding-0.6B | 1,024 | 512 (historical baseline) |

`qwen`/`qwen4b`/`qwen8b` and `nemotron`/`nemotron8b` are two scale ladders, which
is what makes "is the embedder too small?" answerable rather than assumed. They
are not interchangeable ladders, though: **Nemotron-3-Embed declares 42 languages
at both sizes and Slovenian is not one of them**, while Qwen3-Embedding claims
100+. On this corpus a nemotron-to-qwen gap is therefore ambiguous between scale
and language coverage, and the 1B-to-8B step is what separates the two readings —
if coverage is the binding constraint, four times the parameters will not fix it.

All five large checkpoints load in bfloat16 (transformers reads the dtype from the
checkpoint), so residency is roughly the on-disk size: 8.1 GB for `qwen4b`,
15.1 GB for `qwen8b`, 15.9 GB for `nemotron8b`. Two of them do not fit beside a
reranker on one 48 GB card, which is why `experiments/indexing/index_variants.sh`
gives each provider its own process instead of looping in one.

Until 2026-08-11 every provider was pinned to `embedding_max_seq_length: 512`,
which is not a limit any of these models impose — it is the conventional value
for BERT-era encoders. At that length 66% of `base` chunks were truncated and
**22% of the corpus never reached the embedder**, so every retrieval number
published before then describes a corpus the model only partly read.

`qwen_s512` keeps that length available as its own provider so the cost stays
measurable: run it against `qwen` on the same chunks and the same labels and the
only difference is how much of each chunk is read.

> **`page_chunks_qwen_chunk` changed meaning.** It now holds 32,768-length
> vectors, not 512. Rebuild it before comparing against any earlier report.

Both nemotron providers pin `{name}_revision`: the cache is shared between users
and `refs/main` for the 1B points at a metadata-only snapshot, so the model fails
to load with "does not appear to have a file named model.safetensors" despite the
weights being present. The 8B is pinned pre-emptively for the same reason.

Model and collection settings live in `config.yaml`.

Rebuild one collection:

```bash
uv run python -m src.indexing.chunks --method qwen
```

Build the metadata-context collection without replacing v1:

```bash
uv run python -m src.indexing.chunks --method qwen --chunk-version v2
```

The Nemotron collection uses `nvidia/Nemotron-3-Embed-1B-BF16`, its saved
query/document prompts, BF16 weights, SDPA attention, and its full 32,768-token
context. It requires a CUDA-capable NVIDIA GPU for practical inference. Rebuild
the independent 2048-dimensional collection with:

```bash
uv run python -m src.indexing.chunks --method nemotron
```

Rebuild the default regenerable vector cache:

```bash
just eval-index qwen v1
just eval-index qwen v2
```

Audit the effective token length of every current SQLite chunk without computing
embeddings or changing stored artifacts:

```bash
just audit-chunk-tokens
```

The audit loads only locally cached Qwen 0.6B, Qwen 4B, and Nemotron models. It
counts each model's document prompt, the selected v1/v2 representation, and
special tokens with truncation disabled, then compares the result with the
configured and effective sequence limit. Missing model caches fail clearly and
never trigger an implicit download.

The optional paired-language section requires exactly 30 reviewed records at
`experiments/indexing/data/sl_en_tourism_pairs.jsonl`. Each JSONL row must have
non-empty string fields `id`, `sl`, `en`, `source`, and `reviewed_by`, with a
unique `id`. The fixture is intentionally absent until trustworthy reviewed
translations are available; the audit reports that section as blocked rather
than inferring review status.

Durable reference databases belong under `data/db/`. Regenerable Chroma cache
artifacts belong under `data/cache/chroma/`. Use `.env` overrides when a run
should write to private scratch paths under `.local/`.

## Benchmarking against OKF generation

A Chroma build and an OKF bundle build have different products. Compare them
from the same frozen full-page corpus using clean-build wall time, pages and
source MiB per second, coverage, retries, model calls/tokens/cost, peak memory,
artifact size, and storage amplification. Keep chunking, embedding, persistence,
OKF discovery, canonicalization, enrichment, index generation, and validation
as separately timed stages. Also run incremental updates and attach retrieval
and answer quality to every efficiency result.

See [`experiments/indexing/README.md`](../../experiments/indexing/README.md) for the full
protocol derived from BEIR, ANN benchmark, RAG evaluation, and production search
benchmark practices.
