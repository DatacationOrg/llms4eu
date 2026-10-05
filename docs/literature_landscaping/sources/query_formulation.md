### Wen et al. 2024, Elaborative Subtopic Query Reformulation (EQR) + TravelDest (ROEGEN@RecSys 2024)  (https://arxiv.org/abs/2410.01598)
- Problem stated: Travel queries are often broad ("youth-friendly activities")
  or indirect ("a high school graduation trip"); the intent isn't in the
  wording, so retrieval over destination descriptions fails.
- New problem for us? Yes: the team's "vague" questions are vague factual
  questions from chunks; broad/indirect recommendation queries are a different
  case, and the likely December one.
- Goal: technique + benchmark
- Relates to: hybrid retrieval (BM25 + dense, 70/30); documentation skew
  (place-level scoring, H4); Tandon card (intent routing); LLM-judge card
  (human calibration)
- Verdict: adds (query reformulation; place-level scoring; human-labelled
  benchmark design)
- The idea in one sentence: An LLM rewrites the query into k subtopics with a
  short elaboration each (breadth + depth), appended to the query for dense
  or BM25 retrieval; destinations are scored by the average of their top-n
  paragraph scores.
- So what for us:
  - Technique for December recommendation questions: prompt-only, training-free
    query reformulation before retrieval.
  - Place-level scoring (Algorithm 1) is a working example of H4: a place wins
    by its best n passages, not by how many it has, so it counters
    documentation skew. The authors don't frame it this way [claude].
  - TravelDest design (50 queries × 774 cities, every pair labelled 1–5 by 3
    people, relevant if avg ≥ 3, checked by 2 experts) is a template for a
    small, exhaustive human-labelled recommendation set, which also serves as
    the calibration sample an LLM judge needs.
  - Data-dependent hint: dense beat BM25 here and reformulation barely helped
    BM25, so recommendation questions might need more dense weight than 70/30
    (fusion weights per question type; cf. Tandon).
  - Recall is framed as geographical fairness: missing relevant places is the harm.
- Feasibility: local possible (prompt released) but tested only with GPT-4o;
  one LLM call per query. Place-level scoring needs a place ID per chunk.
  Labelling at TravelDest scale is costly (50 × 774 judgments × 3 people).
- Open question: Does a local model (gpt-oss:20b) reformulate as well as
  GPT-4o? Does top-n averaging hold up at POI level, where many places have
  only one or two passages? (n is tuned in the appendix; not read.)
- Limits: city level; English only; 50 queries; no hybrid retrieval tested.