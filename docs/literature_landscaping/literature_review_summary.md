**What I did**

- Mapped the literature on tourism, location-aware and multilingual RAG, plus adjacent work on popularity bias, and compared it with what our system already does.
- About 18 papers in depth, a broader scan of roughly 100, and a targeted check for novelty.

**What I found**

- **The current design holds up.** The soft geo score instead of a hard filter, and the weighted score fusion, match what the literature supports. Keeping goals as separate scores also avoids a documented failure: one model asked to balance several goals quietly drops all but one.
- **Chunking:** no case for semantic chunking; fixed or structure-based chunking does as well at lower cost. Chunk sizes can't be compared fairly with the current question design, and our proposed fix (one shared question set across all chunkings) is the right one.
- **Pepe's two concerns are real and documented elsewhere,** but not for tourism RAG:
  - AI travel recommendations concentrate on famous places, even when users ask for hidden gems.
  - Multilingual retrieval favours the query's language and English, so good places described in other languages, like across a border, get missed.
- **Two things to be careful with:**
  - "Well documented" isn't the same as "much visited"; they need different signals.
  - The geo score itself could favour well-known places, because those are the ones whose location we know.
- **Existing benchmarks:** the closest are TourismQA (real forum questions with several correct answers), SynthTRIPs (generated questions balanced between popular and less popular places) and TravelDest (vague queries, fully human-labelled). None is multilingual or at the level of individual places in Europe. The December dataset could fill that gap.
- **Possible paper angle, for later:** nobody seems to have combined location with language bias in retrieval.

**What it could contribute**

*Retrieval*
- **Per-language balancing:** take the top results from each language separately, then merge. Training-free, with published gains.
- **A geo-aware variant:** balance only the languages spoken near the place in the question.
- **Cap or group results per place,** so one well-documented place can't fill the list. For example, score each place by the average of its best few passages.
- **Query rewriting for vague questions** like "a trip for teenagers": the LLM expands them into concrete subtopics. Prompt-only.
- **Routing by question type:** factual and recommendation questions may need different retrieval settings, such as more weight on dense retrieval for vague questions. Relevant because December brings both types.
- **Over-fetch size matters:** a hidden gem that never enters the candidate pool can't be rescued by any later re-ranking.

*Reranking*
- **A quick check:** does our reranker add language bias? Compare the language mix before and after reranking.
- **Test re-ranking that spreads results over more places** (MMR or xQuAD). Mixed evidence, so test it.
- **Possibly add a term next to the geo score,** with a small weight: documentation volume, popularity, or a locality signal (are visitors mostly locals?), perhaps approximated by the share of local-language sources.
- **Longer term:** a reranker trained against language bias exists, but needs labelled data.

*Generation*
- **A closed-book test:** what does our local model know about our places without retrieval?
- **Adaptive retrieval:** retrieve only when the model is unlikely to know the answer. Only useful if the closed-book test shows the model knows something, and if we allow it to use its own knowledge.
- **Language bias also appears here:** the model prefers citing sources in the query's language.

*Evaluation (works on any dataset, ready for December)*
- **Concentration:** how much of the results go to a few places, across many questions.
- **Breakdowns by language and by how well-documented a place is.**
- **Three measurement points:** model alone, retrieval alone, model plus retrieval, to see where a bias enters.
- **For recommendation questions:** side-by-side LLM judging per aspect (relevance, diversity, popularity mix), checked against a small human-labelled sample. LLM judges agreed with experts only 17–54% of the time.
- **This also applies to our own equivalence judge,** which has never been checked against humans.

*Dataset*
- **Cross-border test questions** (Slovenian to Croatian, German and Hungarian pages).
- **Spatial questions generated from our location metadata,** like "castles within 30 km of X", with answers derived automatically.
- **One shared question set across chunkings,** for fair chunk-size comparisons.
- **The Wikidata place ID in chunk metadata,** needed for all per-place measurements.
- **A check on why pages have no location:** not a place, or too obscure to locate?

**Next step and ask**

- I'll work these out as concrete experiment proposals, then ask you to pick two or three.
- Two questions that shape this:
  - May the chatbot use the model's own knowledge, or must every answer come from our corpus?
  - Should we target the bias toward well-documented places, or actual crowding?