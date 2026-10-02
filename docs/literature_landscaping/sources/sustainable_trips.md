
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
- The idea in one sentence: A computed sustainability score (popularity ×
  seasonal demand) is put into the prompt, and the LLM reranks the retrieved
  cities with it.
- So what for us: A second soft score next to location. The authors list moving
  it into retrieval as future work, which is the slot the geo score's weighted
  sum already has. Two candidate signals:
  visitor popularity (external data, targets crowding) or documentation volume
  (pages/chunks per place, free from the corpus, targets the retrieval bias).
- Feasibility: Runs locally (Llama-3.1-8B, Mistral-7B). Popularity comes from
  the Tripadvisor API; documentation volume needs no external data.
- Evaluation gap: shift toward sustainable cities reported for SAR only, with
  no baseline comparison and no split by popularity tier.
- Open question: Paper assumes city-level recommendation, so does it work
  within one locality? And for the team: fix retrieval bias or steer away from
  crowds?