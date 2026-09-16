# Geo-Aware Retrieval: Literature 2023–2026

One entry per paper or vendor write-up cited in
[geo-improvement-plan.md](geo-improvement-plan.md). Each entry gives the
citation, what was done, the numbers that matter, how much of it was read
(full text or abstract, checked 2026-09-09), and what it means for this
repository. Anything published before 2023 is deliberately absent.

Corpus facts the relevance notes refer to: 176 pages, 726 chunks, 89 pages
with a primary location, 80 of them within 2 km of Rajhenburg Castle, 375
`mentioned` rows over 173 distinct places, 141 model-produced names Nominatim
could not resolve, hybrid BM25 + dense (Chroma) + cross-encoder rerank, agentic
retrieval, Slovenian and English questions.

## A. Spatial and geographic RAG

### Spatial-RAG: Spatial Retrieval Augmented Generation for Real-World Geospatial Reasoning Questions
Yu, Bao, Ning, Peng, Mai, Zhao. arXiv 2502.18470, 2025. https://arxiv.org/abs/2502.18470 (full text, v2)

An LLM parses the question into a reference geometry (point, polyline,
polygon), a spatial predicate (distance ≤ ε or intersection) and the distance
ε itself, estimated from wording ("walking distance"). Sparse spatial
retrieval runs the predicate against a spatial database; the spatial score is
`1/(1+d)` (1 on overlap) blended with a dense cosine between the spatial part
of the query and the candidate text. A semantic score is cosine on the
non-spatial part. Candidates on the Pareto front over (spatial, semantic) go
to the LLM, which chooses weights summing to 1 and returns the argmax.
Datasets: TourismQA-NYC (9,470 POIs, 17,448 QA), TourismQA-Miami, MapQA-ADJ/AME.

| result (NYC, GPT-4-Turbo) | delivery | spatial pass | semantic pass |
|---|---|---|---|
| full system | 86.1% | 71.6% | 50.1% |
| without sparse spatial stage | 98.9% | 53.3% | 78.4% |
| without dense semantic stage | – | 75.9% | 34.8% |

Headline: +19.9% P@1 on TourismQA-NYC. The 13% delivery loss of the full
system is attributed to geometry-recognition failures in the hard spatial
stage.

Relevance: the design template for `GeoScopedRetriever`, and the only paper
with a measured hard-versus-soft ablation on tourism questions. Its gain came
from 9,470 POIs spread across a city with complete footprints; the spatial
signal is a constant on our corpus (80 of 89 located pages at one castle), so
the ablation transfers, the headline number does not.

### GeoRAG: A Question-Answering Approach from a Geographical Perspective
Wang, Zhao, Wang, Cheng, Nie, Luo, Yu, Yuan. arXiv 2504.01458, 2025. https://arxiv.org/abs/2504.01458 (abstract)

A geographic knowledge base (145k entries, 875k QA pairs) organised along
seven dimensions (semantic, spatial location, geometric morphology,
attributes, feature relationships, evolution, mechanisms). A BERT-base
multi-label classifier routes each query to the dimensions it needs; a
retrieval evaluator scores relevance; dimension-specific prompt templates
assemble context. Reports gains over vanilla RAG across base models; no
numbers in the abstract.

Relevance: the geo-intent classifier in front of retrieval is the pattern
behind the `relation` field in WP3. Encyclopaedic geography QA, not a document
corpus, so the KB design does not transfer.

### GeoRAG: A Geographic RAG Framework Based on Urban Spatio-Temporal Knowledge Graph
Chen et al. CLNLP 2025, Springer LNCS. https://link.springer.com/chapter/10.1007/978-981-95-4788-3_12 (abstract, paywalled)

Builds a spatio-temporal knowledge graph from OpenStreetMap with both
hierarchical containment and proximity features, and retrieves through two
paths: place-name matching and coordinate query. Motivation: vanilla RAG lacks
spatial semantics.

Relevance: the closest published analogue to "every page carries a hierarchy
path plus a point" (WP2). Evidence level too thin to quote numbers.

### Tatarstan Toponyms: A Bilingual Dataset and Hybrid RAG System for Geospatial QA
Arabov, Khaybullina, Naumetova. CLIB 2026, arXiv 2605.05962. https://arxiv.org/abs/2605.05962 (abstract)

9,688 Russian–Tatar toponym records, 93.1% georeferenced. Hybrid retriever:
multilingual-e5-large dense index plus KD-tree haversine filtering and
ranking. Recall@1 0.988, MRR 0.994 on 500 queries, beating BM25 and
spatial-only baselines; no baseline delta stated.

Relevance: the most similar setting found (small, bilingual, low-resource
language, heritage toponyms, partial georeferencing, dense plus haversine).
Records without coordinates fall back to the dense score, the neutral-unknown
convention. Abstract only, and a near-perfect recall@1 suggests lookup-style
queries, so it is not the reference study until the full text is read.

### Intelligent Multimodal Retrieval and Reasoning for Geospatial Knowledge Discovery on I-GUIDE
Kang et al. arXiv 2606.15838, 2026. https://arxiv.org/abs/2606.15838 (abstract)

Production RAG (17 months) over OpenSearch keyword, vector and spatial indexes
plus Neo4j, with retrieval-method routing and a 170-query benchmark. Names
spatial metadata quality as the deployment bottleneck.

Relevance: independent confirmation that footprint coverage, not ranking, is
what limits a geo-aware system in practice.

### CubeGraph: Efficient RAG for Spatial and Temporal Data
Yang, Li, Wang. arXiv 2604.06616, 2026. https://arxiv.org/abs/2604.06616 (abstract)

Hierarchical spatial grid cells each holding an HNSW sub-graph, stitched at
query time. Motivation: spatial pre-partitioning destroys HNSW routing
connectivity.

Relevance: the systems-level statement of why filter-first vector search loses
recall; supports over-fetch-then-score in WP1.

### Real-time Spatial RAG for Urban Environments
Campo, Conde, Alonso, Huecas, Salvachúa, Reviriego. ACM TIST 2026, arXiv 2505.02271. https://arxiv.org/abs/2505.02271 (abstract)

FIWARE linked-data context brokers filter retrieval spatially and temporally
before generation; validated with a Madrid tourism assistant. Architecture
paper, no retrieval metrics.

Relevance: tourism domain; confirms filter-first is the common default. No
numbers to reuse.

### Spatially-Enhanced RAG for Walkability and Urban Discovery (WalkRAG)
Amendola, Pugliese, Perego, Renso. arXiv 2512.04790, 2025. https://arxiv.org/abs/2512.04790 (abstract)

Conversational spatial RAG for walkable itineraries; spatial constraints and
preferences stated by the user, POIs along the route retrieved. Preliminary
results only.

Relevance: route-shaped queries ("between X and Y") as a relation type for
the probe set in WP6.

### RALLM-POI: Retrieval-Augmented LLM for Zero-shot Next POI Recommendation with Geographical Reranking
Li, Lim. PRICAI 2025, arXiv 2509.17066. https://arxiv.org/abs/2509.17066 (abstract)

Retrieves historical trajectories, then a geographical distance reranker
re-orders by proximity, then an agentic rectifier. Gains on three Foursquare
datasets without training; decay function not given.

Relevance: distance as a soft reranking stage after retrieval, the WP1 shape.
Trajectory data, not text.

### TP-RAG: Benchmarking RAG LLM Agents for Spatiotemporal-Aware Travel Planning
Ni et al. EMNLP 2025, arXiv 2504.08694. https://arxiv.org/abs/2504.08694 (abstract)

2,348 queries, 85,575 POIs, 18,784 reference trajectories; measures route
efficiency and spatiotemporal compliance; retrieved trajectories improve
spatial efficiency.

Relevance: agentic travel planning is out of scope; useful only as evidence
that agents need retrieved spatial structure rather than model recall.

### Enhancing Tourism Recommender Systems for Sustainable City Trips Using RAG
Banerjee, Satish, Wörndl. RecSoGood@RecSys 2024, arXiv 2409.18003. https://arxiv.org/abs/2409.18003 (abstract)

"Sustainability Augmented Reranking": a metadata-derived score (popularity ×
seasonality) re-ranks retrieved context before prompting; matches or beats the
baseline with Llama-3.1-8B and Mistral-7B.

Relevance: a metadata-score rerank stage in tourism RAG with small open
models, the same shape as a distance multiplier.

### Geospatial reasoning and awareness in LLMs: a systematic review
Dorobantu, Badea. Artificial Intelligence Review 59:111, 2026. https://link.springer.com/article/10.1007/s10462-026-11512-x (full text)

PRISMA review of 54 papers (2022–2025). LLMs hold static near/far
associations, struggle with distance and area computation; RAG plus fine-tuned
open models is the practical path; cites Hu et al.'s 0.91 Acc@161 km as the
toponym-resolution state of the art.

Relevance: the justification for computing distance and containment in code
and never asking the model.

### Agents (abstract level)
GeoAgent (Chen et al., arXiv 2410.18792, 2024): code interpreter plus RAG
inside MCTS for geospatial data processing. GeoLLM-Squad (arXiv 2501.16254,
2025): multi-agent remote-sensing copilot, +17% agentic correctness.
Barrier-free GeoQA Portal (arXiv 2503.14251, 2025): multi-agent decomposition
over geoportal layers. Relevance: none directly; listed for completeness of
the survey.

## B. LLM toponym recognition, resolution and document geolocation

### Toponym resolution leveraging lightweight and open-source LLMs and geo-knowledge
Hu, Kersten, Klan, Farzana. IJGIS 2024. https://doi.org/10.1080/13658816.2024.2405182 (full text via DLR PDF)

LoRA-fine-tuned 7–13B models (Mistral, Baichuan2, Llama2, Falcon) on LGL
generate an unambiguous reference string ("Glasgow, Montana, US"), which is
then geocoded through GeoNames, Nominatim and ArcGIS in turn.

| system | Accuracy@161 km |
|---|---|
| bare geocoders | 0.21–0.77 |
| prior voting ensemble | 0.84 |
| fine-tuned Mistral-7B | 0.91 (0.81–0.98 per dataset) |
| Llama2-70B | 0.93 |

Historical corpora (WOTR, CLDW) are weakest at 0.75–0.81. Abstract claims
+13% over state of the art and 83% mean-error reduction.

Relevance: this is tier 3 of `src/preprocess/locations.py`. Our 141 misses
against 377 accepted names (about 72% acceptance) is their bare-geocoder
baseline. **Reference study for WP4** together with the 2025 follow-up. Their
data is English; our misses are Slovenian surface forms, so lemmatisation is
the adaptation.

### Scalable Toponym Resolution with LLMs: Accuracy and Speed Optimizations
Hu, Kersten, Klan. GeoExT@ECIR 2025, CEUR 3969. https://ceur-ws.org/Vol-3969/paper6.pdf (full text)

Resolves all toponyms of a text in one pass and injects the top-17 GeoNames
candidates (exact-name group ranked by population, then non-exact) into the
prompt; vLLM inference. Over 7 datasets and 83,365 toponyms Acc@161 km rises
0.90 → 0.93 with Mistral-7B, 7× faster (1.4 h vs 10.8 h), matching
Llama2-70B.

Relevance: candidate-list resolution (WP4) and per-page joint resolution of
`mentioned_places`; open 7B models fit the local-only inference decision.

### GeoLM: Empowering Language Models for Geospatially Grounded Language Understanding
Li, Zhou, Chiang, Chen. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.317/ (abstract)

Contrastive and masked-LM pretraining aligning Wikipedia/Wikidata text with
OSM entities, with a coordinate embedding for distance and direction;
supports toponym recognition, linking, relation extraction and typing. Used as
a baseline in Spatial-RAG (69.5% spatial pass, 42.8% semantic pass).

Relevance: a specialised encoder we do not need at 176 pages; shows Wikidata
plus OSM as the grounding pair, which is our tier 1 and tier 3.

### Mordecai 3: A Neural Geoparser and Event Geocoder
Halterman. arXiv 2303.13675, 2023. https://arxiv.org/abs/2303.13675 (abstract)

Neural ranker over GeoNames candidates plus a QA model for event geocoding;
open Python library. Beaten by direct LLM coordinate prediction in Mioduski
(2025).

Relevance: the GeoNames candidate ranking is the fallback gazetteer proposed
in WP4; English-centric.

### Towards efficient and accessible geoparsing of U.K. local media
Bisiani, Gulyas, Heravi. Computational Humanities Research 2025. https://www.cambridge.org/core/journals/computational-humanities-research/article/ED2946D9A21D1A1E86D4A8E1F4EEF193 (abstract and summary)

Local Gemma2, Llama3.1, Qwen2 and Mistral via Ollama with majority voting:
78% accuracy overall; unanimous-vote filtering gives 91% Acc@20 km and 100%
Acc@161 km at 68% retention. Adding metadata to the prompt reduced accuracy.

Relevance: local Ollama models with voting is a direct fit for the gemma and
gpt-oss judges already in the repo; keep enrichment prompts lean.

### Benchmarking LLMs for Geolocating Colonial Virginia Land Grants
Mioduski. JOSIS 31, 2025, arXiv 2508.08266. https://arxiv.org/abs/2508.08266 (abstract)

Six OpenAI models on 17th–18th century metes-and-bounds text; o3 mean error
23 km, 5-call ensemble 19.2 km; external geocoding tools gave no measurable
benefit; name redaction raised error only about 7%.

Relevance: direct coordinate prediction from descriptive historical text is
viable, but the project decision stands: the model names, the gazetteer
places. Listed under "deliberately not doing" in the plan.

### Georeferencing complex relative locality descriptions with LLMs
Fernando, Ranathunga, Stock, Prasanna, Jones. IJGIS 2026, arXiv 2512.14228. https://arxiv.org/abs/2512.14228 (abstract)

QLoRA-tuned LLMs georeference "5 km NW of X" descriptions across regions and
languages: 65% within 10 km on average; 85% / 67% within 10 km / 1 km for New
York.

Relevance: typical error around 10 km means radii should decay, not cut off
(WP3 radius defaults).

### Large Language Models for Geolocation Extraction in Humanitarian Crisis Response
Cafferata, Demarco, Kalimeri, Mejova, Beiró. arXiv 2602.08872, 2026 (companion WWW'26 paper). https://arxiv.org/abs/2602.08872 (abstract)

Few-shot LLM NER plus an agent-based geocoder that uses context to
disambiguate; improves precision and reduces geographic disparity for
under-represented regions.

Relevance: context-aware disambiguation of candidates is the WP4
candidate-list step.

### Comparative Performance of NLP Models and LLMs in Multilingual Geo-Entity Detection
Kopanov. AICCONF 2024, arXiv 2412.20414. https://arxiv.org/abs/2412.20414 (abstract)

SpaCy, XLM-R, mLUKE, GeoLM versus GPT-3.5/4 on English, Russian and Arabic
Telegram text; each has distinct strengths, no universal winner.

Relevance: do not assume one extractor covers Slovenian and English equally;
measure per language.

### Where on Earth Do Users Say They Are? Geo-Entity Linking for Noisy Multilingual User Input
Masis, O'Connor. NLP+CSS@NAACL 2024, arXiv 2404.18784. https://arxiv.org/abs/2404.18784 (abstract)

Averaged-embedding location representations with confidence-based
abstention; a cheap multilingual alternative to LLM geocoding.

Relevance: abstention when unsure is the right behaviour for tier 3; an
embedding-based alias match could back the alias table in WP4.

### CLASSLA-Stanza: the next step for linguistic processing of South Slavic languages
Ljubešić et al. arXiv 2308.04255, 2023. https://arxiv.org/abs/2308.04255

Slovenian NER (LOC) and Sloleks-backed lemmatisation (about 99% lemma F1).

Relevance: the lemmatiser for WP4, if the dependency is accepted; otherwise
the Slovenian suffix fold in `src/preprocess/rejections.py`.

## C. Spatial relations, geo-intent and spatial SQL

### GeoSQL-Eval / GeoSQL-Bench
arXiv 2509.25264, 2025 (journal version Expert Systems with Applications 2026). https://arxiv.org/abs/2509.25264 (abstract)

14,178 natural-language-to-GeoSQL instances over 340 PostGIS functions and 82
databases; 24 models evaluated; public leaderboard.

Relevance: spatial SQL generation is heavier than we need; confirms LLMs can
emit structured spatial predicates when given a schema, which the `relation`
field relies on in miniature.

### Text-to-OverpassQL
Staniek, Schumann, Züfle, Riezler. TACL 12, 2024. https://aclanthology.org/2024.tacl-1.31/ (abstract)

8,352 natural-language to OverpassQL pairs with execution-grounded metrics;
the standard OSM query interface for tool-using agents.

Relevance: a possible agent tool if the corpus ever needs live OSM lookups;
not planned.

### Foundation Models for Geospatial Reasoning: Geometries and Topological Spatial Relations
Ji, Gao, Nie, Majić, Janowicz. IJGIS 2025, arXiv 2505.17136. https://arxiv.org/abs/2505.17136 (abstract)

GPT-4 few-shot reaches about 0.66 accuracy on topological predicates from
WKT; LLM-generated geometries nevertheless improved geo-entity retrieval.

Relevance: containment is computed by point-in-polygon in `src/shared/nuts.py`,
never asked of the model.

### Can LLMs Integrate Spatial Data?
Han, Wolfe, Caspi, Howe. arXiv 2508.05009, 2025. https://arxiv.org/abs/2508.05009 (abstract)

LLMs reason well about human-scale relations but produce incoherent geometry
computations; review-and-refine helps.

Relevance: same conclusion as Ji et al.; geometry stays in code.

### GPSBench
Truong, Lau, Qi. arXiv 2602.16105, 2026. https://arxiv.org/abs/2602.16105 (abstract)

17 tasks, 57,800 samples; strong country-level and weak city-level
localisation; fine-tuning on geometry hurts world knowledge.

Relevance: region codes are the robust fallback layer when a point is
unavailable (WP2 scope-for-every-page).

## D. Filtering, fusion and metadata in retrieval

### An Analysis of Fusion Functions for Hybrid Retrieval
Bruch, Gai, Ingber. ACM TOIS 2023. https://dl.acm.org/doi/10.1145/3596512 (abstract and snippets)

A convex combination of normalised scores beats reciprocal rank fusion in and
out of domain, is robust to the normalisation choice, and its weight can be
tuned from a handful of labelled queries; RRF discards score magnitudes.

Relevance: WP1 fuses the normalised reranker score and the distance score by
convex combination; distance is never folded in through RRF.

### Filtered ANN Search: A Unified Benchmark and Systematic Experimental Study
Shi, Cai, Zheng. arXiv 2509.07789, 2025. https://arxiv.org/abs/2509.07789 (abstract and snippets)

Taxonomy of filter-then-search, search-then-filter and hybrid; recall and QPS
are heavily influenced by filter selectivity, with some methods degrading
sharply at low selectivity.

Relevance: selectivity is what the WP1 gate measures before applying a scope
level.

### Attribute Filtering in ANN Search: An In-depth Experimental Study
Li, Yan, Lu, Zhang, Cheng, Ma. SIGMOD 2026, arXiv 2508.16263. https://arxiv.org/abs/2508.16263 (abstract)

12 methods, 4 datasets up to 10M vectors, selectivity 0.1–100%;
component-level analysis and selection guidelines.

Relevance: background for the Qdrant move; at 726 chunks none of the index
engineering matters yet.

### Accurate and Efficient Metadata Filtering in Pinecone's Serverless Vector Database
Ingber, Liberty. ICML 2025. https://www.pinecone.io/research/ICML_2025.pdf (full text, introduction)

Formalises exact filter recall and argues filtering must be integrated into
the retrieval path to avoid recall loss from ad-hoc post-filtering.

Relevance: over-fetch factor in WP1 must be large enough that post-scoring
does not starve; the paper is the argument for `geo_overfetch`.

### Compass: General Filtered Search across Vector and Structured Data
Ye, Yan, Lo. arXiv 2510.27141, 2025. https://arxiv.org/abs/2510.27141 (abstract)

Cooperative execution over HNSW/IVF and B+-trees supporting arbitrary
predicates without new indexes.

Relevance: none at our scale; noted for the Qdrant decision.

### STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases
Wu et al. NeurIPS 2024 Datasets and Benchmarks. https://openreview.net/forum?id=QSS5cGmKb1 (abstract)

Retrievers struggle when queries mix relational constraints with textual
semantics.

Relevance: "castle near Brežice with a wine cellar" is this query class; the
probe set should contain them.

### Multi-Meta-RAG
Poliakov, Shvai. ICTERI 2024, arXiv 2406.13213. https://arxiv.org/abs/2406.13213 (abstract)

LLM-extracted metadata used as a hard database filter improves MultiHop-RAG,
with the authors conceding the filter set is specific to one question format.

Relevance: the cautionary case for hard LLM-inferred filters, which is what
the 2026-09-08 run measured.

## E. Evaluation

### MapEval
Dihan et al. ICML 2025, arXiv 2501.00316. https://arxiv.org/abs/2501.00316 (abstract)

700 multiple-choice questions over 180 cities in 54 countries; textual, API
and visual tracks; best of 30 models under 67%, more than 20 points below
humans; failures on distance, direction and routing.

Relevance: model-side spatial reasoning is weak, so retrieval must supply the
spatial facts.

### MapQA: Open-domain Geospatial QA on Map Data
Li, Grossman, Qasemi, Kulkarni, Chen, Chiang. SIGSPATIAL 2025, arXiv 2503.07871. https://arxiv.org/abs/2503.07871 (abstract)

3,154 QA pairs with OSM geometries; nine question types covering closeness,
direction and distance; retrieval baselines capture closeness and direction
but fail explicit distance computation.

Relevance: the relation taxonomy for the probe set annotations in WP6.

### GS-QA
Saeedan, Rashid, Eldawy, Hristidis. arXiv 2605.22811, 2026. https://arxiv.org/abs/2605.22811 (abstract)

2,800 QA from 28 templates over OSM plus Wikipedia, including directional
predicates and multi-source questions; nine baselines; large drops on complex
predicates and numeric outputs.

Relevance: template-generated spatial questions with typed predicates is a
cheap way to extend the probe set.

### GeoBenchX
Krechetova, Kochedykov. SIGSPATIAL GeoGenAgent 2025, arXiv 2503.18129. https://arxiv.org/abs/2503.18129 (abstract)

23 geospatial tools, four difficulty tiers, deliberately unsolvable tasks,
LLM-as-judge.

Relevance: include unanswerable geo questions in the probe set so the intent
gate is tested.

### MultiGlobeQA
Böckling, Nosova, Paulheim, Iana. arXiv 2608.03882, 2026. https://arxiv.org/abs/2608.03882 (abstract)

46,060 QA in 17 languages across 201 countries; topological relations and
directions fare best, grid and shape computation collapse; retrieval plus
tools plateaus under 67%.

Relevance: multilingual evidence that country and region level are the
reliable layers.

### Evaluating LLMs on Spatial Tasks: A Multi-Task Benchmarking Study
Xu et al. arXiv 2408.14438, 2024. https://arxiv.org/abs/2408.14438 (abstract)

Twelve task types including place-name recognition and route planning;
GPT-4o 71.3% zero-shot; chain-of-thought lifts route planning from 12.4% to
87.5%.

Relevance: background only.

## F. Vendor and industry write-ups

- **Qdrant, A Complete Guide to Filtering in Vector Search (2024).**
  https://qdrant.tech/articles/vector-search-filtering/ and
  https://qdrant.tech/documentation/search/filtering/. Pre-, post- and
  filterable-HNSW; restrictive filters disrupt graph connectivity; geo radius,
  box and polygon as payload filters; `is_null` for unlocated points.
  Relevance: the target store; `is_null` makes include-null native.
- **Elastic Search Labs, Geo-semantic search to refine recommendations (2024).**
  https://www.elastic.co/search-labs/blog/geo-semantic-search-elasticsearch.
  kNN with a hard `geo_distance` filter. Relevance: the filter-first default
  our run measured.
- **Elastic Search Labs, Multimodal RAG with Elasticsearch geospatial capabilities (2025).**
  https://www.elastic.co/search-labs/blog/multimodal-rag-elasticsearch-geospatial.
  RRF over lexical, text-kNN and image-kNN sharing one 100 km geo filter.
  Relevance: fusion shape to avoid for distance (Bruch et al.).
- **Elastic Search Labs, LangChain self-querying retriever with Elasticsearch (2025).**
  https://www.elastic.co/search-labs/blog/self-querying-retrievers. Warns that
  results are only as good as the LLM's filter extraction. Relevance: why the
  resolver output must be verified by a gazetteer and cached.
- **Alonso, Geospatial RAG on Postgres (2026).**
  https://www.pedroalonso.net/blog/geospatial-rag-postgres/. Eligibility
  filters, then `0.45·semantic + 0.35·exp(−d/5000) + 0.20·exp(−d_line/3000)`
  over a few thousand candidates; a blended score cannot ride the HNSW index.
  Relevance: the production shape of WP1; assumes non-null geometry.
- **Milvus, Hybrid Spatial and Vector Search with Milvus 2.6.4 (2026).**
  https://milvus.io/blog/hybrid-spatial-and-vector-search-with-milvus-264.md.
  GEOMETRY type with R-tree, `ST_DWITHIN` / `ST_WITHIN` as pre-filter,
  vector-only ranking. Relevance: another filter-first store.
- **Weaviate `withinGeoRange`** https://docs.weaviate.io/weaviate/search/filters ;
  **LlamaIndex VectorIndexAutoRetriever**
  https://developers.llamaindex.ai/python/framework/integrations/vector_stores/chroma_auto_retriever/ ;
  **LangChain self-query** https://js.langchain.com/docs/how_to/self_query/.
  All emit LLM-inferred hard metadata filters. Relevance: the pattern to
  avoid as a default on a half-located corpus.

## Not found in 2023–2026 sources

- A paper quantifying recall loss from *incomplete* metadata coverage in RAG;
  filtered-ANN benchmarks vary selectivity, not missingness. The 2026-09-08 run
  is that measurement for this corpus.
- A dedicated geospatial-RAG survey; Dorobantu & Badea (2026) and the VLDB
  2025 tutorial by Hussein, Hemdan and Mokbel are the closest.
- Any Slovenian-specific geoparsing evaluation.
- Any vendor post applying a geo decay inside a RAG pipeline rather than a hard
  filter.
- Location encoders (SatCLIP, GeoCLIP, RANGE) used to condition text
  retrieval; found only for image and coordinate tasks.
