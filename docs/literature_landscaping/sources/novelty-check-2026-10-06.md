# Novelty check: location × language bias and documentation skew in retrieval (final)

## Bottom line

No published work combines a document's location with language bias in retrieval or RAG, so that part of the idea (H1–H5) is a defensible novelty claim. "A geo score reduces popularity bias" (H6) is not new in recommenders and has mixed results; it is open only for text RAG with documentation skew.

- **Location × language (H1–H5):** none found for H1, H2, H5; partial for H3, H4. Frame the claim as bias mitigation in multilingual dense RAG, not as cross-lingual geographic IR, which GeoCLEF covered in 2005–2009.
- **Geography × popularity (H6):** recommender papers show geographic context can cut popularity bias, but one geographic model raised it by 40%. Nobody has tested it on chunk-count skew in RAG.
- **Diversity and caps (Task 2):** MMR and coverage re-ranking have measured effects in RAG; a per-place cap has none.

**Is the research enough?** Enough to decide the novelty claim and start experiments. Not enough to publish it: three key papers were seen only through citing papers, and the GIS and RecSys-workshop venues were not searched systematically. The open questions in the last sections are better answered by your own experiments than by more reading.

This document replaces the earlier research report. It merges that report, a cross-check against the Klimashevskaia et al. popularity-bias survey (UMUAI 2024), and a targeted 2024–2026 search.

## Hypotheses H1–H6

Every row lists the closest papers found by either search; "none found" means nothing combines the two parts of the hypothesis.

| Hyp. | Verdict | Closest papers | How close |
| --- | --- | --- | --- |
| **H1** distance score counters language bias for nearby other-language docs | None found | [Goworek et al.](https://arxiv.org/abs/2511.19324) (arXiv 2025) · [Ki et al., Linguistic Nepotism](https://arxiv.org/abs/2509.13930) (ICML 2026) · [Żatuchin](https://arxiv.org/abs/2608.30052) (arXiv 2026) · [Localization Boosting](https://arxiv.org/html/2605.11272v1) (arXiv 2026) | Goworek: recall correlates with how close *languages* are geographically, and multilingual-E5 keeps a strong query-language bias. Ki: generators cite query-language or English documents even when less relevant. Żatuchin: query language and user location act separately; mismatched language drops local suppliers (n=6 per cell). Localization Boosting: a locale boost, not distance. None uses document location. |
| **H2** language quotas chosen by geography | None found | [BORDIRLINES](https://arxiv.org/abs/2410.01171) (Findings ACL 2025) · [DELTA query fusion](https://arxiv.org/abs/2601.02956) (arXiv 2026) · [MLAIRE](https://arxiv.org/abs/2605.07249) (arXiv 2026) | BORDIRLINES adds the languages of the countries in a territorial dispute: a geography-chosen language set, by topic not distance. It is the strongest counter-evidence. DELTA fuses English and local-language queries, locale taken from language. MLAIRE treats query-language preference as partly wanted. |
| **H3** location-first, language-blind candidate pool | Partial | [Tool4POI](https://arxiv.org/abs/2511.06405) (AAAI 2026) · [Tatarstan Toponyms](https://arxiv.org/pdf/2605.05962) (arXiv 2026) · [Multilingual search for geo-textual data](https://link.springer.com/article/10.1007/s41651-025-00232-5) (JGSA 2025) · [GeoCLEF 2008](https://ceur-ws.org/Vol-1174/CLEF2008wn-GeoCLEF-MandlEt2008.pdf) | The mechanism exists (region/distance tools, multilingual dense retrieval plus spatial filter), but no paper measures language bias with it. GeoCLEF runs often found geo filters no better than text alone. |
| **H4** place (Wikidata ID) as retrieval unit across languages | Partial | [ELERAG](https://arxiv.org/abs/2512.05967) (arXiv 2025) · [Entity Retrieval](https://aclanthology.org/2025.knowledgenlp-1.1.pdf) (KnowledgeNLP 2025) · [HIKE](https://cdn.aaai.org/ojs/20355/20355-13-24368-1-2-20220628.pdf) (AAAI 2022) · [GikiCLEF](https://dl.acm.org/doi/10.5555/1887364.1887393) (CLEF 2009) | ELERAG boosts chunks linked to Wikidata IDs, Italian only. Entity Retrieval makes the entity the unit, monolingually. HIKE uses multilingual Wikidata entities to enrich queries. GikiCLEF motivates H4: only 9 of 50 topics had answers in all ten languages. None merges one place's passages across languages. |
| **H5** location decides which passages get translated | None found | [QTT-RAG](https://aclanthology.org/2025.mrl-main.12.pdf) (MRL 2025) · [CrossRAG](https://aclanthology.org/2026.findings-eacl.35.pdf) (Findings EACL 2026) · Goworek et al. 2025 · Ki et al. 2026 | QTT-RAG translates the top-5 non-query-language documents (rank decides, not location). CrossRAG translates everything. Goworek: translation adds little for dense *retrieval*; Ki: the bias left is in *generation*, where H5 would act. |
| **H6** location score reduces concentration on popular or well-documented places | Prior work in recommenders; none in RAG | [Forster et al.](https://arxiv.org/abs/2507.03503) (RecSys 2025) · Rahmani et al. 2022a, *ESWA* 205:117700 · Sandholm & Ung 2011 (CaRR) · [Banerjee et al.](https://arxiv.org/abs/2011.07359) (IEEE BigData 2020) | Forster (popularity = normalised check-ins): LORE cut popularity \~93% on Foursquare; USG (geographic influence) raised it 40%. Rahmani (check-ins): context fusion less biased than collaborative filtering. Sandholm & Ung: location relevance over popularity. Banerjee (exposure in "near me" search): distance ranking favours nearby, lower-rated venues. |
| **H6** context: how concentration is measured | Audits only | [Digital overtourism](https://www.tandfonline.com/doi/full/10.1080/13683500.2026.2654066) (Current Issues in Tourism 2026) · [Destination (Un)Known](https://www.mdpi.com/2673-2688/6/9/236) (AI 2025) · [SPACE](https://arxiv.org/html/2608.07998) (arXiv 2026) · [The Concentrated City](https://www.mdpi.com/2413-8851/9/7/268) (Urban Science 2025) · [Mozafari et al.](https://arxiv.org/abs/2605.12382) (arXiv 2026) | Popularity measures differ per paper: digital footprint; Euromonitor/WEF rankings; visit frequency; ChatGPT mentions vs Instagram presence; pretraining exposure vs Wikipedia pageviews. No paper uses documentation volume (chunk count), which supports keeping it separate from visitor popularity. |

## Task 2: diversity re-ranking

Diversity and coverage re-ranking in RAG has evidence of effect; a per-entity or per-place cap against popularity concentration still has none.

| Paper | Setting | Evidence |
| --- | --- | --- |
| [DF-RAG](https://arxiv.org/abs/2601.17212) (Findings EACL 2026) | MMR in RAG, λ tuned per query | +4–10% F1 over plain cosine retrieval on reasoning-heavy QA. |
| [Ross et al., redundancy and diversity](https://arxiv.org/abs/2608.13956) (arXiv 2026) | Duplicated vs diverse RAG context | Duplicates add nothing; genre-diverse context raises correctness 17–47%. |
| [SetR, Shifting from Ranking to Set Selection](https://aclanthology.org/2025.acl-long.861.pdf) (ACL 2025) | Reranker picks a set for coverage, not a ranked list | Beats LLM rerankers on multi-hop RAG and raises information coverage. |
| Abdollahpouri et al. 2019, xQuAD for popularity bias (via the survey) | xQuAD with popular/long-tail as the two "aspects" | Increases long-tail exposure. A reproduction (Klimashevskaia et al. 2022) found the related popularity re-ranker fits each user but barely moves platform-wide bias. |
| [Forster et al.](https://arxiv.org/abs/2507.03503) (RecSys 2025) | Calibrated-popularity re-ranking on POI lists | Lowers popularity on Foursquare and Yelp, at an accuracy cost; tail items stay under-represented when the candidate pool lacks them. |

Negative evidence: [Bal & Puhan](https://arxiv.org/pdf/2605.02520) (arXiv 2026, biomedical RAG) found MMR gave the lowest answer relevancy (0.658) of five strategies, so diversity can push out needed passages.

Nearest thing to a per-place cap: MacRAG (2025) builds context from top-k *distinct documents*, and Intercom's Fin [retrieves per source type and merges](https://fin.ai/research/using-llms-as-a-reranker-for-rag-a-practical-guide/) to keep one source from dominating. Forster's last finding matters for you: re-ranking cannot surface a hidden gem that never entered the over-fetched pool.

## Is the novelty claim defensible?

Yes for location × language, if scoped to a method evaluated as language-bias reduction in multilingual dense RAG. No for "a geo score reduces popularity bias" on its own.

Language-bias work from 2024–2026 (Park & Lee, LAURA, Linguistic Nepotism, MLAIRE, Goworek et al.) has no document geography. Geo-RAG work has no language-bias measurement. Three things a reviewer will raise:

- **GeoCLEF/GikiCLEF (2005–2009)** already combined spatial filters with cross-language retrieval. Frame the claim as bias mitigation, not as cross-lingual geographic IR.
- **Goworek et al. 2025**: neighbouring languages are already retrieved better (correlation with language geography). Slovenian–Croatian may show little gap; Slovenian–Hungarian or Slovenian–German is the harder test.
- **BORDIRLINES** picks languages by the countries involved, which is close to H2.

For H6, the honest claim is narrower: geography as a *corrective* for documentation skew (chunk count) in text RAG with half the pages unlocated. Forster et al. and Banerjee et al. show a geo signal can also create its own concentration, on whatever is nearest, so the claim needs a test that measures both.

## Worth a further deep-research pass

Six threads came up that this check did not settle, ordered by how much they would change the design.

1. **Does the geo score create its own bias?** Forster's USG result and Banerjee's "near me" exposure finding say distance ranking can concentrate on the nearest places. Needed: how decay width and fusion weight shift exposure, and any fix (exposure-aware distance decay).
2. **Measuring selection for open questions.** Recommender metrics (ARP, PopLift, Gini, long-tail share) exist; no standard exposure metric for RAG answers was found. Needed: a metric that counts which *places* an answer names, against chunk count and against a visitor signal.
3. **A visitor or crowding signal for Europe.** The papers above use check-ins, Wikipedia pageviews, Euromonitor rankings or pretraining exposure. Needed: open, local-hardware-friendly sources at LAU/NUTS-3 level (Eurostat nights spent, pageviews, OSM tags) and how well they track crowding.
4. **Language-pair specificity.** Per-pair language-bias numbers for sl, hr, hu, de, cs, sk, it in LAURA, SHIFT and MLAIRE, to pick test pairs where the gap is real.
5. **Generation-stage language bias.** Linguistic Nepotism says the generator, not only the retriever, drops other-language evidence. Needed: mitigations that run locally (citation-aware prompting, selective translation into the query language), which is where H5 lives.
6. **Unsearched venues.** SIGSPATIAL, GIScience, ECIR/BIAS workshop and RecSys tourism workshops were not searched systematically; geo × language work, if it exists, would most likely be there.

## Caveats

Both searches were targeted, not systematic, so "none found" is weaker evidence than a full review.

- **Read in full:** Forster et al. 2025, Goworek et al. 2025, the Klimashevskaia survey, Żatuchin 2026.
- **Abstract or summary only:** MLAIRE, Linguistic Nepotism, Mozafari et al., ELERAG, Tool4POI, SetR, BORDIRLINES, DELTA (its \~20% local-only figure is unverified), CrossRAG.
- **Seen only through citing papers:** Rahmani et al. 2022a, Sandholm & Ung 2011, Banerjee et al. 2020, Abdollahpouri et al. 2019. Read these before citing a number from them.
- **Preprints, not peer-reviewed:** Goworek, MLAIRE, Mozafari, Ross, ELERAG, Żatuchin, SPACE, Localization Boosting, DELTA, Bal & Puhan.
- Żatuchin uses six runs per cell and the author works at an AI-visibility firm.
