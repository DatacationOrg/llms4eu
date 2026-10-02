### Qu, Tu & Bao 2025, Is Semantic Chunking Worth the Computational Cost?
(https://aclanthology.org/2025.findings-naacl.114/)
- Goal: evaluation (+ technique)
- Relates to: chunking (team: split at headings, then size limit); evaluation design
- Verdict: confirms (no case for semantic chunking); adds (evaluation checklist)
- The idea in one sentence: Semantic chunking is rarely worth its cost over
  fixed-size chunking on real documents; embedding quality matters more.
- So what for us: Don't invest in semantic chunking. Use the authors' list of
  what a chunking comparison needs (real documents, diverse queries, human
  answers, relevance scores, human-labelled evidence) as a checklist for
  December. The team has LLM-made evidence quotes, but no human labels and
  one gold chunk per question.
- Feasibility: no action needed
- Open question: Heading-based chunking is untested here. The authors name
  contextual embeddings as the next thing to explore (see Merola & Singh).