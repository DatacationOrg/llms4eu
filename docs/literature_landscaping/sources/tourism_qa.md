### Contractor et al. 2020, Large Scale QA using Tourism Data  (arXiv 1909.03527)
Note: old dataset (2020)
- Problem stated: Entity-seeking recommendation questions (hotels, attractions,
  restaurants). Each question can have thousands of candidate answers, each described
  by unstructured reviews.
- New problem for us? Yes: recommendation-style questions with a set of valid
  answers, not one gold chunk.
- Goal: benchmark
- Relates to: benchmark positioning (datasets.md); Spatial-RAG's TourismQA-NYC
- Verdict: adds
- The idea in one sentence: Match a real user question and its implied
  preferences against a large candidate set; based on real forum data.
- So what for us: If the December data contains recommendation-style questions,
  TourismQA is a close existing benchmark: real forum questions, a set of
  valid answers, and mostly locative or budget constraints (~64%).
  (Secondary: our proxy set has 37 of 3,476 near-wording questions.)
- Feasibility: local yes; no training data needed; needs labels for our own
  questions.
- Open question: / 