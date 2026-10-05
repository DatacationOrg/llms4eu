### Banerjee et al. 2026, Collab-REC: multi-agent balancing of tourism recommendations (ACM TORS, just accepted)  (https://doi.org/10.1145/3837862)
- Problem stated: One LLM asked to balance personalization, popularity and sustainability in one prompt quietly favours one objective, usually famous places ("objective collapse"), and may name items outside the catalog.
- New problem for us? Partly known (popularity bias); separating stakeholder objectives is new
- Goal: technique + evaluation (label-free concentration metrics)
- Relates to: soft geo score (weighted sum); second soft score thread; Andreev card (label-free metrics); SynthTRIPs and S-Fairness cards (same group)
- Verdict: adds (framing, metrics); the agent machinery does not transfer
- The idea in one sentence: Three role-prompted LLM agents (personalization, popularity, sustainability) each propose 10 cities from a 200-city catalog, and a deterministic non-LLM moderator validates, scores, merges and bans cities over ~4 rounds.
- So what for us: Keep objectives as separate, visible scores combined with explicit weights. That is our soft score extended (e.g. + documentation volume, + popularity), with per-query weights from the WP3 parser, or a diversity rerank (MMR/xQuAD, quotas). If December questions are recommendation-style, their protocol (constraint satisfaction against attributes + Gini/coverage) is a template that needs no gold chunk.
- Feasibility: local possible (gpt-oss-20b tested) but 4–8 min and 44–76k tokens per query; needs structured item attributes; no training, no labels.
- Open question: Does the diversity gain survive against a simple diversity reranker, the comparison they did not run? And do documentation volume and visitor popularity measure the same thing on the December data?