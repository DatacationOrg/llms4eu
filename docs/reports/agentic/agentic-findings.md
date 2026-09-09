# Agentic retrieval — findings

What we tried, what we observed, and enough detail to reproduce. Metric is hit@5
unless noted.

## Setup

- Corpus: 726 chunks of Slovenian encyclopedic text (history, biographies,
  geography).
- Eval: 3,471 approved questions across five categories — `crosslingual`,
  `direct_long`, `direct_short`, `vague_long`, `vague_short`. hit@k measures
  whether a gold chunk lands in the top-k over the full corpus.
- Components: BM25 sparse; dense embeddings (Qwen3-Embedding 0.6B and 4B); score
  or RRF fusion; cross-encoder rerank (Qwen3-Reranker-0.6B). The agentic
  sufficiency judge is a DeepSeek-V4 model on Azure Foundry; embedding and
  reranking run locally on GPU.

## Baseline

Hybrid (dense + sparse) followed by cross-encoder rerank is the strong baseline:
on the current index, qwen4b hybrid+rerank ≈ 0.91 hit@5 on the full set. The
reranker is the dominant factor in retrieval quality.

## Experiments

- **Fusion.** RRF-merging distinct rerankers lifts recall@10 (~0.79 → 0.83) at a
  small hit@1 cost. Multi-query expansion (an LLM generates query variants, then
  fuse) did not help — the variants were near-duplicate paraphrases — and cost
  3–5× more.
- **Agentic loop.** Retrieve, then an LLM judge reads the retrieved chunks and
  decides whether they suffice to answer, reformulating and retrying if not. On
  raw dense retrieval it scored ~0.33; rewired to sit on top of the reranker it
  reaches reranker level (~0.85–0.9). The loop adds little over reranking, and
  retries are driven by the judge's "can I answer" bar, which is stricter than —
  and only loosely aligned with — recall.
- **Answerability gating.** Let the system decline underspecified questions and
  ask back. An LLM gate asks back ~17% overall (42% on `vague_short`) with a real
  lift where vagueness lives (`vague_short` 0.76 → 0.83). A BERT-NER gate (abstain
  when no named entity is present) is cheap but weak: denial precision caps at
  ~24% at any confidence threshold — entity presence is a poor proxy for
  answerability, since many entity-free questions are answerable.

## Bugs found

- **Un-cached vector-store client** — a new Chroma client was created per query,
  leaking connections until readiness checks failed mid-run. Likely cause of long
  runs stalling partway. Fixed by caching the client.
- **Agentic result truncation** — an early "sufficient" verdict could return fewer
  than the requested number of results, capping hit@10 / recall@10. Fixed.
- **Agentic base retriever** — rewired onto the reranked pipeline.

## Reproduce (minimal)

- Build a chunk vector index for a provider, then run the retrieval-method
  comparison over the eval questions; it checkpoints and is resumable.
- Judge-dependent methods need the Azure judge model configured; everything else
  runs locally.
- Rate note: the Azure judge is token-limited (~125k tokens/min) and each call
  sends the retrieved chunk texts (~4–5k tokens), so throttle to ~25–30 calls/min.

## Future work

Answerability gating is the promising direction, but neither a general LLM nor an
off-the-shelf NER model separates answerable from unanswerable well. Options:
fine-tune a small classifier on the question set, or gate on retrieval confidence
(the reranker's top score, already computed).
