### Mallen et al. 2023, When not to trust language models (ACL 2023)  (https://arxiv.org/abs/2212.10511)
- Problem stated: LLMs have difficulty with less popular factual knowledge, and scaling mostly helps popular facts. Retrieval helps on the long tail but can hurt on popular entities, as retrieved context can be misleading.
- New problem for us? Already known (the reason we retrieve); the harm case is new
- Goal: benchmark (PopQA) + evaluation (results split by popularity) + technique (Adaptive Retrieval)
- Relates to: retrieval; popularity thread (second soft score); Park & Lee card (model's own knowledge)
- Verdict: confirms retrieval helps for less popular entities; contradicts that retrieval always helps
- The idea in one sentence: Adaptive Retrieval retrieves only when the question's entity is below a popularity threshold (Wikipedia page views, tuned per question type on labelled data), and uses the model's own knowledge otherwise.
- So what for us: Wikipedia page views are a open popularity signal that also covers small places, so we could stratify the eval by popularity. However, it can be language dependent how many views it has.
- Feasibility: local yes; page views via the Wikipedia API; threshold needs labelled questions
- Open question: Partly answered: scaling lowers the popularity threshold but not far into the tail. Transfer question: does a current local model (gpt-oss:20b) know anything about our localities, closed-book? If not, adaptive retrieval is moot for us and the risk is the model overriding retrieved context.