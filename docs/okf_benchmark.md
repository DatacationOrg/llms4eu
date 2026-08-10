Overall

method	hit@1	hit@5	hit@10	recall@10	mrr@10
sparse_rerank	0.742	0.841	0.851	0.851	0.786
qwen_hybrid_rerank	0.754	0.873	0.886	0.886	0.804
okf	0.473	0.501	0.501	0.501	0.485
okf_search	0.570	0.605	0.605	0.605	0.584

RAG knowledge bases diagram

```mermaid
flowchart LR
	classDef canonical fill:#fff8e6,stroke:#7a5c00,color:#2b2200,stroke-width:1.2px;
	classDef derived fill:#e8f5ff,stroke:#0b4f82,color:#06243a,stroke-width:1.2px;
	classDef query fill:#eaf8ee,stroke:#1f6a3d,color:#103320,stroke-width:1.2px;

	subgraph Sources[Source data and ingestion]
		SEED[data/places.jsonl\nseed places]
		SCRAPED[scraped markdown pages\n(raw text + metadata)]
	end

	subgraph CanonicalKB[Canonical knowledge bases (source of truth)]
		SQLITE[(.local/places.db\nplaces + place metadata)]
		PAGES[(data/db/pages.sqlite3\npage chunks + eval rows)]
	end

	subgraph DerivedKB[Derived retrieval knowledge bases]
		CHROMA[(data/cache/chroma/\nChroma collections\nqwen / nemotron / embed_v4 / ...)]
		BM25[[In-memory BM25 sparse index\nbuilt from page chunks]]
	end

	subgraph QueryFlow[Query-time retrieval path]
		Q[User query]
		DENSE[Dense retriever\n(vector search)]
		SPARSE[Sparse retriever\n(BM25)]
		HYBRID[Hybrid fusion / rerank / agentic loop]
		CONTEXT[Top-k chunks / places\nfor answer generation]
	end

	SEED --> SQLITE
	SCRAPED --> PAGES

	SQLITE -->|place embeddings| CHROMA
	PAGES -->|chunk embeddings| CHROMA
	PAGES -->|tokenized chunk text| BM25

	Q --> DENSE --> CHROMA
	Q --> SPARSE --> BM25
	CHROMA --> HYBRID
	BM25 --> HYBRID
	HYBRID --> CONTEXT
	CONTEXT -->|full text + metadata lookup| PAGES

	class SQLITE,PAGES canonical
	class CHROMA,BM25 derived
	class Q,DENSE,SPARSE,HYBRID,CONTEXT query
```

Speed

method	seconds	ms/query	queries/query	queries
sparse_rerank	531.29	1345.0	1.00	396
qwen_hybrid_rerank	325.16	823.2	1.00	396
okf	5385.05	13633.0	4.40	1737
okf_search	4884.96	12367.0	3.72	1468

hit@5 by category

method	crosslingual	direct_long	direct_short	vague_long	vague_short
sparse_rerank	0.710	0.945	0.831	0.905	0.802
qwen_hybrid_rerank	0.899	0.945	0.892	0.929	0.721
okf	0.435	0.548	0.566	0.583	0.372
okf_search	0.594	0.699	0.687	0.643	0.419



