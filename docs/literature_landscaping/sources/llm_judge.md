### Banerjee et al. 2026, Multi-dimensional evaluation of sustainable city trips with LLM-as-a-judge (UMAP 2026, short paper)  (https://doi.org/10.1145/3774935.3812717)
- Problem stated: Travel recommendation lists have no ground truth, expert evaluation is costly, and standard metrics ignore sustainability and popularity balance. LLM judges scale but carry model-specific biases on subjective criteria.
- New problem for us? Partly known (recommendation questions can't be evaluated with one gold chunk); LLM-judge calibration is new
- Goal: evaluation
- Relates to: TourismQA / Tandon thread (set-valued answers); Collab-Rec card (subjective constraints, same group); Andreev card (label-free audits)
- Verdict: adds (pairwise multi-dimensional protocol); contradicts the hope that an LLM judge can replace experts
- The idea in one sentence: Judge two recommendation lists side by side on four separate dimensions (relevance, diversity, sustainability, popularity mix), compare LLM judges with human experts, and turn their large disagreements into explicit rules and few-shot examples in the judge prompt.
- So what for us: If December questions are recommendation-style, this is the template for evaluating lists without ground truth: pairwise A vs B (e.g. with vs without the soft geo score), per dimension, never one aggregate. Judges match the expert majority only 17–54% of the time, so a human-judged sample is needed to calibrate any judge.
- Feasibility: local possible (prompts and code released), but judges were large API models (gpt-5, gemini-2.5-pro, deepseek-v3). Calibration needs domain experts: ~15 h for 100 queries × ≥3 experts.
- Open question: Do rules calibrated on one query set hold on new queries, and how do local judges (gpt-oss:20b) behave?