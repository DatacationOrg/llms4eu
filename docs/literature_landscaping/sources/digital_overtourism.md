# Digital overtourism in AI travel recommendations: evidence froma comparative analysis
### Najafi & Costa 2026, Digital overtourism in AI travel recommendations (Current Issues in Tourism)  (https://doi.org/10.1080/13683500.2026.2654066)
- Problem stated: AI travel recommenders may concentrate attention on already
  famous destinations before anyone travels ("digital overtourism"), driven by
  uneven online representation and popularity-weighted ranking.
- New problem for us? Partly: the documentation-skew concern is known (Pepe),
  but this names it, shows it across 10 AI systems, and links it to
  overtourism theory.
- Goal: evaluation + problem framing
- Relates to: documentation skew; evaluation toolkit; Banerjee et al. 2024/2025
- Verdict: new ground (problem named) + adds (concentration metrics)
- The idea in one sentence: AI recommendations look diverse (many different
  places mentioned) but attention concentrates on a few iconic places, even
  when users explicitly ask for hidden gems or sustainable options.
- So what for us:
  - Evaluation instruments, data-independent: count how often each place
    appears across many queries; report concentration (HHI, Top-k share) and
    effective diversity (Shannon entropy, effective number), not just the
    number of distinct places. Applicable to retrieval: which places keep
    appearing in the top-5?
  - Failure mode to test for: "horizontal substitution". Alternatives to
    Santorini become nearby similar islands (Milos, Naxos), staying within the
    familiar circuit instead of reaching the periphery.
  - Positioning: LLMs and assistants without retrieval, English prompts only,
    city/region/country level. Whether RAG amplifies or dampens this, in
    several languages and at POI level, is open.
- Feasibility: metrics are simple counts over outputs; no training. Needs a
  query set and a way to identify places in results (place IDs or names).
- Open question: Does retrieval over a skewed corpus reproduce this
  concentration, and does a documentation-volume signal reduce it?
- Note: Exploratory study (420 responses, one point in time). The recursive
  feedback loop is the authors' conceptual model, not something they measured.