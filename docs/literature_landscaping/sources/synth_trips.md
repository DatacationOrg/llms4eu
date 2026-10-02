### Banerjee et al. 2025, SynthTRIPs (SIGIR)  (doi 10.1145/3726302.3730321)
SynthTRIPs: A Knowledge-Grounded Framework for BenchmarkQuery Generation for Personalized Tourism Recommenders

- Problem stated: Tourism recommender systems need high-quality data in breadth
  (diverse places, not just popular ones) and depth (nuanced preferences like
  budget, sustainability, avoiding overtourism).
- New problem for us? Already known (data quality), but not acted on directly.
  New: LLMs favour popular destinations even when users ask for less popular ones.
- Goal: benchmark + evaluation
- Relates to: benchmark positioning (datasets.md); Banerjee et al. 2024 (same group)
- Verdict: adds
- The idea in one sentence: Synthetic queries grounded in a knowledge base, with
  diverse user personas, structured filters and natural-sounding requests;
  reusable generation and evaluation components.
- So what for us: If the December data contains preference-based queries,
  SynthTRIPs is a positioning point (4,604 queries, equal share of popular and
  less popular destinations). Its popularity-bias finding is evidence for the
  bias toward well-documented places. Evaluation idea: stratify questions by popularity (as here and in PopQA) and report results per group; this measures the documentation-skew bias directly.
- Feasibility: Code and data released (CC BY-NC); generation used
  Llama-3.2-90B and Gemini, so a local run needs a smaller substitute.
- Open question: City level only (European cities, "City A vs City B"), so how
  far does it transfer to attractions within one locality?