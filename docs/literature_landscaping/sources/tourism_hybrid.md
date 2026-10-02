# Evaluating User Intent Classification and Hybrid Retrieval in a RAG-based Conversational Tourism Recommender System
### Tandon & Banerjee 2025, Intent + hybrid RAG in a conversational TRS (RecTour@RecSys)  (https://mediatum.ub.tum.de/doc/1834489/document.pdf)
- Problem stated: Recommender systems suffer from cold start, hindering
  personalization and relevant recommendation.
- New problem for us? Already known
- Goal: technique
- Relates to: baseline "Retrieval → hybrid + rerank"; WP3 relation field
- Verdict: confirms (hybrid + rerank) / adds (intent routing)
- The idea in one sentence: Classify user intent first (multi-label, few-shot
  LLM), then pick the retrieval and action per intent (answer vs recommend).
- So what for us: Modular design for different user intents (e.g. factual
  answering vs recommendation), close to WP3's relation field. Location is a
  hard city filter, which works only because every chunk has a city.
  Recommendations are not evaluated (no reference context).
- Feasibility: Local (Llama-3.1-8B, few-shot intent, no training); judge is
  GPT-4o-mini via RAGAS. Code released. Intent labels needed to evaluate.
- Open question: Is it beneficial to have different retrieval modules per
  intent, and to stratify evaluation by intent? Would this work at different
  levels of locality (country, region, city, within city)? They never ablate
  intent-driven retrieval, so the gain is untested.