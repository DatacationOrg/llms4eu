**A pruned baseline (one screen)**

**Retrieval**
- **Hybrid first stage:** dense search and BM25 each return their top 30. Each list's scores are rescaled to 0–1 per query, then combined as 0.7 × dense + 0.3 × BM25. This keeps the size of score gaps, which RRF throws away. The July runs found it better than RRF.
- **Rerank:** a cross-encoder (Qwen3-Reranker-0.6B) cuts the top 30 to 10. It reads only the question and the chunk text, with no location. A 4B reranker was tested as well.
- **Agentic variants:** page tools and file-based exploration.

**Chunking**
- **How pages are cut:** split at headings, then filled up to a size limit in characters (`base`: about 1,800) or in embedder tokens.
- **Variants compared:** 256, 512 (with and without overlap) and 1,024 tokens.
- **What the sweep found:** the reranker matters most, model size next. Bigger chunks score higher, but the test favours them.

**Geo**
- **Location data:** coordinates and NUTS codes are stored as chunk metadata. Places are named by an LLM and located by a gazetteer.
- **Soft score, after reranking:** 0.7 × reranker score + 0.3 × geo score. The geo score decays with distance (50 km); an unknown location counts as neutral.
- **What isn't location-aware:** none of the text a model reads contains location. A strict-filter variant is kept only for comparison.

**Evaluation**
- **Questions:** LLM-generated from chunks, five types (direct/vague × short/long, plus cross-lingual). Each has one gold chunk, the one it was written from. No human review.
- **Anchoring:** each answer's supporting quote is stored as a character span in its page. Gold for any other cutting comes from overlap with that span.
- **Metrics:** hit@k and MRR; text coverage and budget metrics (needs anchors); share of the store read per query; judge-adjusted hits for agents.

**Known weak spots** (from the team's own reports)
- **Chunk sizes aren't fairly compared.** Each cutting got its own questions: more of them for small cuts, easier ones for big cuts. Big chunks also contain answers more easily. The shared-question rerun hasn't been done.
- **Most results come from the 176 Brestanica pages.** That's where the labels are, and it's what the chunk sweep ran on, from before the corpus grew to 1,338 pages.
- **The Nemotron comparison is skewed.** Its model card doesn't list Slovenian, so Nemotron-vs-Qwen gaps say little about model size.
