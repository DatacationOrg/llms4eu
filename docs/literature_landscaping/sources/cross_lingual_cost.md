# A guiding question: which techniques counter language preference in retrieval, and could any be combined with location as a language-independent signal?

### Amiraz et al. 2025, The Cross-Lingual Cost (ArabicNLP@EMNLP)  (https://aclanthology.org/2025.arabicnlp-main.6.pdf)
- Problem stated: Retrieval ranking suffers in cross-lingual domain-specific
  scenarios where the query language differs from the document language (drops
  of over 40%). Retrievers rank well within one language but favour documents
  in the query's language.
- New problem for us? Already known (cross-lingual question type exists), but
  not taken into account in retrieval
- Goal: evaluation and technique
- Relates to: cross-lingual evaluation questions; baseline "Retrieval → hybrid"
- Verdict: confirms (impact on the retriever), adds (methods to counter)
- The idea in one sentence: Counter language preference by retrieving an equal
  top-k per language, or by searching with both the original and the translated
  query and merging by score; both work equally well, and balanced costs
  nothing extra.
- So what for us: Balanced retrieval needs a set of languages; location could
  choose it (languages around the place). The quota would sit before the
  reranker, which then merges into one ranking.
- Feasibility: Balanced is simple to add locally. Translation used Google
  Translate, so it needs a local translation model. Equal quotas stayed stable
  at 25/50/75% English, but only two languages were tested.
- Open question: Does domain-specific matter here, as knowledge about places is
  mostly wiki-style? Can language selection be based on the languages spoken
  around the POI's location? How does it work with 8 languages?