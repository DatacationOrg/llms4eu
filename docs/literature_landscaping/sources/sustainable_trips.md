
# Enhancing Tourism Recommender Systems for Sustainable City Trips Using Retrieval-Augmented Generation

https://arxiv.org/pdf/2409.18003
### Banerjee et al. 2024, Sustainable city trips RAG (RecSoGood@RecSys)  (arXiv 2409.18003)
- Problem stated: Tourism recommender systems focus too much on user preferences
  and ignore broader sustainability goals. Tourism is a multi-stakeholder
  alignment problem (environment, local businesses, residents): balancing
  preference, popularity, seasonality.
- New problem for us? Already known, but not acted on (team note: no popularity
  signal exists).
- Goal: technique
- Relates to: baseline "Retrieval → Rerank"; the soft geo score
- Verdict: adds
- The idea in one sentence: Add a sustainability score (popularity × seasonal
  demand) to the retrieved context and let the LLM rerank cities with it.
- So what for us: A second soft score next to location, scoring how
  over-visited a place is. The authors suggest moving it into the retrieval
  phase, which is exactly the slot the geo score's weighted sum already has.
- Feasibility: Runs locally (Llama-3.1-8B, Mistral-7B). Popularity and
  seasonality come from the Tripadvisor API, so we need our own signal
  (e.g. Wikipedia pageviews ⚑) or the December data.
- Open question: Paper assumes city-level recommendation (which European city to
  visit), so does it work within one locality?