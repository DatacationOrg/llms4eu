# Investigating Language Preference of Multilingual RAG Systems
### Park & Lee 2025, Language preference of multilingual RAG systems (Findings of ACL 2025)  (https://aclanthology.org/2025.findings-acl.295.pdf)
- Problem stated: Retrievers prefer high-resource and query languages; generators prefer the query language or Latin scripts. Relevant docs in other languages are either not retrieved or not used.
- New problem for us? Already known (Amiraz), not acted on
- Goal: evaluation (MLRS) + technique (DKM-RAG, generation only)
- Relates to: dense + BM25 + reranker; cross-lingual questions; Amiraz card
- Verdict: adds (metric); technique does not touch retrieval
- The idea in one sentence: MLRS (MultiLingualRankShift) measures how far documents rise when translated into one language and re-scored; DKM-RAG translates the retrieved passages into the query language and adds an LLM-rewritten version.
- So what for us: MLRS needs no labels, so we can measure our retriever's language preference per language and locality. English queries over low-resource docs lose most, likely the December case.
- Feasibility: local (NLLB-600M, bge-m3, code on GitHub); no training, no labels; one translation per non-query-language doc.
- Open question: Not answered. Could location choose which passages get translated, to cut cost?