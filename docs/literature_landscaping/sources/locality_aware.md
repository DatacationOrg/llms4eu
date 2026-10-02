
# Sustainable Tourism through Locality-Aware Recommendations
### Çelik et al. 2026, Locality-aware recommendations (UMAP4GOOD@UMAP)  (https://ceur-ws.org/Vol-4206/UMAP4GOOD-5.pdf)
- Problem stated: Recommenders that rank by popularity and proximity concentrate
  tourists in a small set of hotspots (overtourism), while locally oriented
  venues stay under-represented.
- New problem for us? Already known, but not acted on yet
- Goal: technique + evaluation
- Relates to: baseline "Geo → soft score"; Banerjee et al. 2024
- Verdict: adds
- The idea in one sentence: Define a per-venue locality score (do its reviewers
  mostly live in the city, or travel widely?) and add it to the ranker as one
  tunable weight in a convex combination.
- So what for us: A third candidate signal next to popularity and documentation
  volume: who visits a place, not how many. Same weighted-sum shape as our geo
  score. Also gives the evaluation Banerjee lacked: accuracy plus novelty,
  coverage, geographic spread and local share, reported against a baseline.
- Feasibility: No training; but needs reviews with per-user histories (Yelp),
  which our corpus doesn't have. Only targets restaurants in Philadelphia.
- Open question: Can this be applied to European localities and broader
  categories, and without review data (December data, or a corpus proxy)?