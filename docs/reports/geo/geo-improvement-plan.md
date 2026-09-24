# Geo Retrieval Improvement Plan

Written 2026-09-08 after the first measured geo run
([../reports/retrieval/geo-run-2026-09-08.md](../reports/retrieval/geo-run-2026-09-08.md)).
It says what that run showed, what the geographic-IR literature and the search
vendors do about exactly that failure, and the work packages that follow, in
order, each with an acceptance test. The current design is
[geo-retrieval.md](geo-retrieval.md); decisions taken here go into the
"Geographic Scope" section of [decisions.md](decisions.md) when implemented.

## 1. What the run showed

Six methods, 500 questions, `base` variant, qwen embedder, Azure judge. The
resolver scoped 72 questions. On those 72, hit@5 fell from 0.917 (`qwen_hybrid_rerank`)
to 0.722 (`qwen_hybrid_rerank_geo`): one win, fifteen losses. Recomputed from
the checkpoint and `data/db/pages.db`:

| fact | number |
|---|---|
| scoped questions, by final filter level | radius 53, country 10, nuts3 6, nuts2 3 |
| losses, by level | radius 8, country 5, nuts3 2 |
| lost questions whose gold page has **no** location | 15 of 15 |
| scoped questions whose gold pages are all unlocated | 16 (15 lost, 1 rescued by widening) |
| of those 16, gold page has a `mentioned` row naming the scope place | 7 |
| located pages in NUTS-3 SI036 (Posavska) | 85 of 89 |
| enrichment rejections | Nominatim no hit 141, Wikidata not-a-place 36, name mismatch 3 |
| eval questions with near / blizu / okolica wording | 37 of 3,476 |

Four distinct causes sit behind the fifteen losses:

1. **Hard filter under partial footprint coverage.** 89 of 176 pages have a
   primary location. Any hard filter removes the other 87 outright, and the
   gold page was among them every time. Widening never fired because five or
   more located pages always supplied chunks. Spatial-RAG (Yu et al.
   2025) measured the same shape on TourismQA-NYC: its hard spatial stage cut
   answer delivery from 98.9% to 86.1%.
2. **Uninformative scope levels.** Every located page is in Slovenia and 85 of
   89 are in one NUTS-3 region. A country or NUTS-3 filter therefore keeps all
   located pages and removes only the unlocated ones: pure loss, no
   discrimination. This is why "Slovenija" was the worst scope.
3. **Footprint too narrow in kind.** The gold pages were not unlocatable, they
   were unlocated by design: the volleyball club page (a Brestanica club), the
   Tönnies biography (a Ljubljana builder), Celeja (Roman Celje, rejected by the
   Wikidata class rule), the statistical-regions and heritage-register pages
   (national scope). A single `primary` point per page cannot express "a club in
   Brestanica" or "about all of Slovenia".
4. **Relation not modelled.** The resolver returns a place and a width but not
   the relation. "Kdo je bil varuh trgovcev v Celeji?" is *about* Celje, not
   *near* it; the right behaviour is a boost for Celje pages, not a fence
   around them. Most of the 72 scoped questions are of this kind, because the
   eval set was generated from chunks and has almost no true spatial questions.

Two smaller findings: 141 Nominatim misses is the largest single loss of
footprint in enrichment (Slovenian declined forms and qualifiers in the
LLM-produced names), and the `v3` chunk text with a `Location:` line changed
nothing, so location belongs in metadata and scoring, not in the embedded text.

## 2. What the recent literature and the vendors say

Only work from 2023 onwards, and only the findings that change a decision.
The full list is in §7.

**A hard spatial stage costs recall; a soft spatial score keeps most of both
objectives.** Spatial-RAG (Yu et al. 2025) is the closest published design to
ours: an LLM parses the question into a reference geometry, a spatial predicate
and a distance, a sparse spatial retrieval runs against a spatial database, a
dense semantic score runs on the rest of the question, and candidates on the
Pareto front over (spatial, semantic) go to the LLM, which chooses the weights
per query. Their ablation on TourismQA-NYC is the number to remember: with the
hard sparse spatial stage, 86.1% of questions get an answer at all; without it,
98.9%. The spatial pass rate falls from 71.6% to 53.3% and the semantic pass
rate rises from 50.1% to 78.4%. Their own explanation of the 13% delivery loss
is geometry-recognition failures in the hard stage. Our fifteen losses are the
same mechanism with a different cause (missing footprint instead of wrong
geometry). Their spatial score is `1 / (1 + d)`, soft by construction.

**Weight the spatial score per query.** Spatial-RAG lets the LLM pick the
spatial and semantic weights per question. GeoRAG (Wang et al. 2025) puts a
BERT multi-label classifier in front of retrieval to route each query to the
geographic dimensions it needs, and bypasses the others. Our relation classes
(in, near, about, none) are the cheap version of both: `about` questions, which
are most of our 72 scoped ones, should not touch the spatial score at all.

**Fuse by convex combination of normalised scores, not by rank.** Bruch, Gai &
Ingber (ACM TOIS 2023) show a convex combination of normalised scores beats
reciprocal rank fusion in and out of domain, is robust to the normalisation
choice, and can be tuned from a handful of labelled queries. RRF throws away
the magnitude of the distance score, which is the one thing a decay function
carries. The geospatial-RAG-on-Postgres write-up (Alonso 2026) uses exactly
this shape in production: eligibility filters first, then
`0.45 * semantic + 0.35 * exp(-d / 5000) + 0.20 * exp(-d_line / 3000)` over a
few thousand candidates, with the note that a blended score cannot ride the
HNSW index, so over-fetch then fuse.

**Missing footprint is the binding constraint, and nobody scores it as far
away.** Tatarstan Toponyms (Arabov et al. 2026) is the closest setting to ours:
a small bilingual heritage gazetteer, 93.1% georeferenced, multilingual-e5
dense index plus a KD-tree haversine stage, recall@1 0.988. The 7% without
coordinates simply fall back to the dense score. I-GUIDE (Kang et al. 2026),
seventeen months in production over OpenSearch keyword, vector and spatial
indexes, names spatial metadata quality as its deployment bottleneck. CubeGraph
(Yang et al. 2026) shows the systems-level version of the same problem: spatial
pre-partitioning breaks HNSW routing connectivity, so filter-first search loses
recall. The filtered-ANN benchmarks (Shi et al. 2025; Li et al. SIGMOD 2026;
Pinecone at ICML 2025) all show recall collapsing at low filter selectivity.
No paper quantifies recall loss from *incomplete* metadata specifically; the
2026-09-08 run is that measurement for this corpus.

**Give every page a hierarchy path and a point.** The urban spatio-temporal KG
GeoRAG (Chen et al., CLNLP 2025) builds both hierarchical containment and
proximity features from OpenStreetMap and retrieves through two paths, place
name matching and coordinate query, because vanilla RAG lacks spatial
semantics. MultiGlobeQA (Böckling et al. 2026) and GPSBench (Truong et al.
2026) find LLMs reliable at country and region level and weak at city and
point level, which makes region codes the robust fallback layer when a point is
unavailable.

**Resolve toponyms with a small model naming and a gazetteer placing.** Hu,
Kersten, Klan & Farzana (IJGIS 2024) fine-tune 7B models to emit an
unambiguous reference string that GeoNames, Nominatim and ArcGIS then geocode,
reaching Accuracy@161 km of 0.91 against 0.21 to 0.77 for bare geocoders; the
follow-up (GeoExT@ECIR 2025) resolves all toponyms of a text in one pass with
the top seventeen GeoNames candidates in the prompt, reaching 0.93 seven times
faster. Historical corpora are their weakest case (0.75 to 0.81), which is
where Celeja sits. Bisiani et al. (2025) run local Gemma, Llama, Qwen and
Mistral through Ollama with majority voting and reach 91% within 20 km on the
unanimous subset, and report that adding metadata to the prompt *reduced*
accuracy. Dorobantu & Badea's systematic review (AI Review 2026, 54 papers)
concludes that LLMs hold only static near/far associations and cannot compute
distance or containment; Ji et al. (IJGIS 2025) measure GPT-4 at about 0.66 on
topological predicates from WKT. Distance and containment are computed in
code, never asked of the model.

**Evaluate with a protocol that separates delivery from spatial and semantic
correctness.** Spatial-RAG reports delivery, spatial pass and semantic pass as
three numbers; MapQA (Li et al. 2025) and GS-QA (Saeedan et al. 2026) type
every question by spatial relation (closeness, direction, distance,
containment) and show retrieval baselines handling closeness and direction but
failing explicit distance. TourismQA-NYC and Miami are the only retrieval
benchmarks found with explicit in / near / along-route constraints. A probe set
for this corpus needs the same relation typing, and the comparison report needs
a coverage column so a footprint failure is never read as a ranking failure.

**Vendor practice is hard filters, which is why the literature above matters.**
Qdrant's filtering guide (2024) documents that restrictive filters disrupt the
HNSW graph and offers geo radius, box and polygon as payload filters, with
`is_null` to let unlocated points through. Elastic's geo-semantic (2024) and
multimodal-RAG (2025) posts share one `geo_distance` filter across lexical and
kNN retrievers fused by RRF. Milvus 2.6.4 (2026) adds a GEOMETRY type with an
R-tree, used as a pre-filter with vector-only ranking. LangChain and LlamaIndex
self-query retrievers, and Elastic's own post on them (2025), all emit
LLM-inferred hard metadata filters and warn that results are only as good as
the extraction. None of these applies a geo decay inside a RAG pipeline, and
all assume complete geometry; for a half-located corpus that assumption is the
bug.

## 2a. Reference study: which result transfers to this corpus

Scored 2026-09-09 on eight criteria (0 to 2 each): corpus size and shape,
language, domain, footprint coverage, query type, match to our retrieval
stack, quality of the evidence, and whether the paper's mechanism targets the
bottleneck the run exposed.

| study | corpus | lang | domain | coverage | queries | stack | evidence | bottleneck | total |
|---|---|---|---|---|---|---|---|---|---|
| Hu et al., IJGIS 2024 + GeoExT 2025 (LLM names, gazetteer places, candidate RAG) | 1 | 0 | 1 | 2 | 1 | 2 | 2 | 2 | **11** |
| Tatarstan Toponyms, 2026 (bilingual heritage, e5 + haversine, 93% georeferenced) | 2 | 2 | 1 | 2 | 1 | 2 | 0 | 1 | 11 |
| Spatial-RAG, 2025 (LLM spatial parse, spatial + semantic Pareto) | 0 | 0 | 2 | 0 | 1 | 2 | 2 | 1 | 8 |
| GeoRAG urban STKG, CLNLP 2025 (hierarchy + coordinate dual retrieval) | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 5 |
| RALLM-POI, PRICAI 2025 (distance reranker over trajectories) | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 4 |

Two facts about our data decide the pick.

First, **the spatial signal is a constant on the current corpus.** Of the 89
located pages, 80 lie within 2 km of Rajhenburg Castle and 86 within 25 km;
there are 15 distinct coordinate pairs in total. All six scoped questions the
text baseline misses have their gold page at 0 km from the scope and outside
the top 10, so no distance score, however shaped, can reorder them in. The only
retrieval gain available on this corpus is recovering the fifteen losses, which
any neutral-for-unknown soft method does. Spatial-RAG's +19.9% P@1 came from
9,470 POIs spread across New York with complete footprints; none of that
spread exists here, so its headline number will not reproduce. It stays the
design template for the ranking stage (§2, WP1), not the study whose result we
expect to see.

Second, **the spread that does exist is in the footprints we do not yet use.**
The 375 `mentioned` rows name 173 distinct places and 204 of them lie more than
25 km from Rajhenburg (Celje, Ljubljana, Trst, Gradec, the biographies' towns).
That layer is what a future spatial question would discriminate on, and its
quality is set by tier 3: the model names a place, Nominatim resolves it. The
run rejected 141 names as "no hit", against 377 accepted, an acceptance rate of
about 72%. That is precisely the "bare geocoder" baseline of Hu et al. (0.21 to
0.77 Accuracy@161 km across seven datasets), which their unambiguous reference
string plus GeoNames candidate list lifts to 0.91 to 0.93 with 7B open models
that fit our local-only inference decision. Their weakest case, historical
corpora at 0.75 to 0.81, is our Celeja case.

**Pick: Hu, Kersten & Klan (GeoExT@ECIR 2025), built on Hu et al. (IJGIS
2024).** It is the best-evidenced study whose mechanism is our tier 3 and whose
setting shares our defining property, a text corpus with partial footprints.
Expected effect on our data: tier-3 acceptance from about 72% toward 90%, so
roughly 100 more mentioned rows and footprints for the handful of concept and
historical pages Wikidata rejected; no change to the 500-question hit@5, which
the spatial signal cannot move here. The gap to close is language: their
datasets are English, and our misses are Slovenian surface forms, so the
lemmatisation step in WP4 is the adaptation, not an extra.

Tatarstan Toponyms ties on the rubric and is the closest setting, but only the
abstract is available and its recall@1 of 0.988 has no stated baseline delta,
so it cannot be the reference until the full text is read.

**First action, done 2026-09-09** (`just locate-pages --rejections`,
[../reports/geo/locate-rejections-2026-09-09.md](../reports/geo/locate-rejections-2026-09-09.md)):
151 distinct refused names over 160 mentions. 124 resolve with a fix that
needs no new model: 14 are names the corpus already holds, 17 lose a type
word or qualifier, 42 resolve once the `si` country hint is lifted, 51 are
found by a Slovenian-language Wikidata search (exonyms and historical
places). Zero are inflected forms. 86 of the 124 are places abroad. The same
check found that the hint had silently placed Gradec, Dunaj, Rim, Trst and
Gorica on Slovenian hamlets in the stored `mentioned` rows. So the reference
study's mechanism (name, then resolve against a candidate list with a country
*prior* rather than a country *filter*) is confirmed as the right one, and the
lemmatisation half of WP4 is demoted: the model already writes nominatives.

## 3. Design principles adopted

1. Geography re-ranks; it does not pre-filter. A hard filter survives only as an
   explicit `_strict` method for measurement and for the agent's `pages_in_region`
   tool, where the user asked for containment.
2. Unknown footprint is neutral: multiplier 1.0, never excluded.
3. Every page gets a scope at the deepest level the evidence supports:
   point, municipality, NUTS-3, NUTS-2, country, or none, with a role (`literal`,
   `associative`, `national`) and a confidence.
4. A scope level is applied only when it discriminates: if it keeps more than
   90% of located pages it carries no information and is skipped.
5. The question's spatial relation decides the weight: `near` strongest,
   `in` next, `about` weakest, `none` off. The resolver returns the relation.
6. The model names, the gazetteer places. Names are lemmatised and matched
   against multilingual aliases before any network call. Every external lookup is
   cached in SQLite.
7. Coordinates stay canonical and codes stay derived, as today. Chunk metadata
   stays payload-only so the Qdrant move keeps working.

## 4. Work packages

Ordered by expected lift per hour. Each ends with a measurable gate; the
rerun of the three reranked geo cells on 500 questions is about 40 minutes
with a warm resolver cache.

### WP1. Soft geography (ranking only, no re-enrichment)

**Status: implemented and gate passed, 2026-09-09**
([../reports/geo/geo-soft-vs-strict-2026-09-09.md](../reports/geo/geo-soft-vs-strict-2026-09-09.md)).
On the 72 scoped questions the soft `qwen_hybrid_rerank_geo` matches the text
baseline question for question (0.917, no wins, no losses; 0.884 on all 500),
the 16 unlocated-gold questions are all recovered, and the near-wording
questions are unchanged. The strict shape with `geo_include_null: true` lands on the same numbers, so on this corpus the null policy is the whole effect. The relation weights below are not yet in use: `w` is
a single 0.3 until WP3 returns the relation. The strict shape survives as
`*_hybrid_rerank_geo_strict`.


Change `GeoScopedRetriever` (`src/retrieval/retrievers/geo.py`) from
filter-then-boost to over-fetch-then-score:

- Retrieve `limit * geo_overfetch` (default 4) chunks unfiltered from the
  existing stage, rerank as today, then multiply every score by
  `(1 - w) + w * s_geo`, where `s_geo = exp(-km / decay)` for a point scope,
  `1.0` for a chunk whose codes lie in a region scope, `0.0` for a located
  chunk outside it, and `1.0` (neutral) for an unlocated chunk. Truncate to
  `limit`.
- `w` comes from the relation (WP3 supplies it; until then `near` 0.4, `in`
  0.3, `about` 0.15) instead of the single `geo_boost_weight`. Normalise the
  reranker score to [0, 1] first so this is a convex combination of two
  normalised scores, the fusion shape Bruch et al. (TOIS 2023) found beats rank
  fusion and tunes from a handful of labelled queries; do not fold distance in
  through RRF, which discards its magnitude.
- Selectivity gate: `GeoScope.effective_level(counts)` drops any level whose
  in-scope share of located pages exceeds `geo_max_scope_share` (0.9). With
  today's corpus this turns "Slovenija" and "Posavska" scopes into no-ops,
  which is the correct answer for a single-region corpus.
- Keep the current hard path as `*_hybrid_rerank_geo_strict` (with
  `geo_include_null: true` and widening on "candidate set unchanged" rather
  than on count) so the two shapes can be compared in one report.
- Config: `geo_include_null: true`, `geo_overfetch: 4`, `geo_relation_weights`,
  `geo_max_scope_share: 0.9`; retire `geo_min_candidates` for the soft path.

Gate: on the 72 currently scoped questions, `qwen_hybrid_rerank_geo` hit@5 ≥
0.917 (no losses against the baseline) and the 37 near/okolica questions show
no regression. Effort: one day plus the 40-minute rerun.

### WP2. Footprint coverage: a scope for every page

Extend enrichment (`src/preprocess/locations.py`, `page_locations`) so that
"unlocated" means "we looked and there is nothing", not "the page is not a
place":

- **Associative footprints.** Promote `mentioned` rows to scoreable
  footprints with role `associative` and a lower in-scope multiplier
  (`geo_associative_factor`, 0.85). The seven rescuable losses (volleyball club,
  Tönnies) come back without any new lookup. Denormalise the top two
  associative places per page into chunk metadata (`assoc_nuts3_1`,
  `assoc_lat_1`, …) or, simpler, into a per-page lookup the boost reads from
  SQLite, which it already does for coordinates.
- **Document scope by aggregation** (the dual hierarchy-plus-coordinate
  retrieval of the urban STKG GeoRAG, Chen et al. 2025). For a page with
  mentions but no primary, take the lowest hierarchy node that covers the
  majority of its resolved mentions: a biography mentioning Brestanica, Krško
  and Ljubljana gets scope `SI` (country) with role `national` if the mentions
  scatter, or `SI036` if they cluster. Store as a row with `granularity`
  region/country and `method: aggregated`.
- **National-scope pages.** Statistical regions of Slovenia, the heritage
  register, summer time in Slovenia: a page whose LLM answer names the country
  or several regions gets a `country` row, role `national`. With the
  selectivity gate this row is neutral today and becomes useful the day a
  second country enters the corpus.
- **Historical places.** Relax the Wikidata class rule: an item that is an
  ancient city, archaeological site or former settlement with P625 and a P17 or
  P131 (Celeja has both) is a place. Keep the exclusion for languages, empires
  and states.
- **Multi-place list pages.** Allow several `primary` rows for pages the LLM
  classifies as a listing ("The castles of Posavje"), and take the metadata
  point from their centroid with `granularity: region`.
- Report coverage in `just locate-pages` output: pages by deepest scope level
  and by role.

Gate: primary-or-scope coverage ≥ 160 of 176 pages; on the 16 scoped questions
with unlocated gold pages, at least 12 gold pages now carry a footprint at the
queried level. Effort: two days.

### WP3. Query understanding: relation and intent

Extend `ExtractedQueryLocation` (`src/shared/geo_resolver.py`) with
`relation: in | near | between | about | none` and, for `between`, a second
place. The prompt gets one example per relation, in Slovenian and English.

- `about` (the place is the topic) maps to a boost with a small weight and
  never to a filter; `near` maps to a point scope with radius from the wording
  or granularity; `in` maps to the containment level the place actually has
  (a municipality name gives LAU, a region name gives NUTS-3); `between`
  resolves both places and boosts by distance to the segment midpoint with
  radius half the separation.
- Radius defaults by granularity: point 10 km, municipality 15 km, town with
  "okolica" 25 km, region none. Today everything gets 25 km.
- Cache key includes the prompt version so a prompt change invalidates
  `geo_scope_cache` without a manual delete.
- Report per-relation counts in the `Geo scope` table of the comparison.

Gate: on a 60-question labelled sample from the eval set (20 about, 20 in,
20 near/none), relation agreement with a human label ≥ 0.85. Effort: one day.

### WP4. Gazetteer quality and the 141 misses

Reference study: Hu, Kersten & Klan (GeoExT 2025), see §2a. The first step,
logging and classifying the rejected names, is done: `just locate-pages
--rejections PATH` (`src/preprocess/rejections.py`, 2026-09-09) writes one row
per refused name with the cheapest fix that resolves it. The result is in
§2a.

- **Persistent geocode cache.** A `geocode_cache(query_norm, provider, hit_json,
  created_at)` table, consulted before Wikidata and Nominatim, satisfies the
  Nominatim policy and makes `just locate-pages` idempotent.
- **Country hint as a prior, not a filter** (the 2026-09-09 result: 86 of 124
  recoverable misses were abroad, and five exonyms were mis-placed on Slovenian
  hamlets). Query Nominatim without `countrycodes`, take the top three, and
  prefer the in-country candidate only when its name matches as well as the
  others do. Re-resolve every stored `mentioned` row this way before WP2 makes
  those rows scoreable.
- **Exonyms through the Slovenian Wikidata search.** Dunaj, Budimpešta,
  Celovec, Trst, Rim, Gradec, Pariz, Ženeva all resolve by `wbsearchentities`
  with `language=sl`; verify with the returned item's `sl` label rather than the
  English one. This is the alias table below, seeded from the misses.
- **Strip type words and saint names before lookup**, as
  `src/preprocess/rejections.py::retry_candidates` already does for the
  classifier: 17 of the misses were churches, basilicas and castles whose
  place segment resolves once the type segment is dropped.
- **Lemmatisation, demoted.** The run found no oblique forms among 151 misses;
  the model writes nominatives. Keep the Slovenian suffix fold in
  `rejections.py` as a fallback and revisit CLASSLA only if a later run shows
  inflected misses.
- **Alias table from Wikidata.** For every QID the corpus knows, store labels
  and aliases in sl, en, de, hr, it in `place_aliases`; the corpus-names tier
  of the gazetteer matches against all of them. Štajerska / Styria /
  Steiermark stop being three lookups.
- **Candidate-list resolution** (Hu et al. 2025). When Nominatim or Wikidata
  return several candidates, hand the top five, with type and admin path, back
  to the LLM to pick, instead of taking the first hit and checking a shared
  token. Apply a Slovenia-bounded prior: candidates in the country hint rank
  first, ties by Nominatim importance or P1082 population.
- **Second gazetteer.** Add GeoNames (feature class filter to A, P, S, T) as a
  fallback for names Nominatim misses, or self-host Photon once volume grows.
- Domain sources to verify: the Slovenian Register of Immovable Cultural
  Heritage (EŠD units with coordinates, open data) would locate every castle
  and church in the corpus authoritatively; the national geographic names
  register (REZI) is a nominative gazetteer with high coverage. Both need a
  licence and endpoint check before they enter the tiers.

Gate: `just locate-pages --rejections` reports under 30 unresolved names and
zero `outside_hint` or `wikidata` rows (those fixes are in the tiers, so the
classifier has nothing left to find); every mentioned row whose name is an
exonym has moved out of Slovenia. Effort: one and a half days.

### WP5. Storage schema `geo2` and the municipality level

- Chunk metadata gains `geo_status` (`literal`, `associative`, `national`,
  `unknown`), `lau` (LAU / ISO 3166-2 municipality code from GISCO LAU
  polygons, point-in-polygon in `src/shared/nuts.py`), and `granularity`.
  Collections record `metadata_schema: geo2`; geo methods accept `geo1` for
  the soft path and require `geo2` for the strict path.
- Radius filtering stays a bounding box on the stored floats followed by
  haversine, which Chroma supports. Grid cells (H3 resolution 6 to 8 as
  scalar keys `h3_r6`, `h3_r7`, `h3_r8`, filtered with `$in` over a k-ring)
  are the fallback if a polygon scope (vernacular Posavje across the border)
  ever needs emulating in Chroma; Chroma has no array metadata, so one key per
  resolution. Not built until a query needs it.
- `pages_in_region` accepts LAU codes; `find_pages_near` reports the footprint
  role so the agent can tell a castle from a biography.

Gate: `just eval-index qwen v1 base` rebuilds with `geo2`; existing tests pass;
`pages_in_region("SI-054")` returns the Krško pages. Effort: one day plus the
index rebuild.

### WP6. Evaluation that can see the G

- **Footprint coverage column.** For every scoped question the comparison
  driver (`experiments/indexing/compare_qwen_modes.py`) records whether each
  gold page has a footprint at the level used, and reports hit@5 separately
  for "gold covered" and "gold uncovered". The fifteen losses would have been
  labelled a coverage failure, not a ranking failure, on day one.
- **Spatial probe set.** 40 to 60 hand-written questions in Slovenian and
  English, each annotated with relation (in, near, between, about, none),
  place, gold footprint type (point, municipality, NUTS-3, vernacular) and gold
  pages: "castles near Sevnica", "kaj si lahko ogledam v Posavju", "cerkve med
  Brestanico in Krškim", plus negatives whose gold page is outside the scope
  and non-geo controls that must not trigger the intent gate. Judged with the
  equivalence judge, stored beside `retrieval-results-handmade.md`.
- **Per-relation and per-level tables** in the report, and an explicit
  "recall lost to null footprint" line.
- **Selectivity report.** Print the in-scope share per level at resolve time
  so an uninformative scope is visible in the log.

Gate: the probe set exists, is approved, and the soft geo method beats the
text baseline on the `near` and `between` buckets while matching it on
`about` and `none`. Effort: two days, mostly writing and approving questions.

### WP7. Scrape-time structured data (fourth tier)

`page_fetch.py` holds the HTML in memory and persists only the trafilatura
markdown, so schema.org JSON-LD (`Place`, `TouristAttraction`,
`GeoCoordinates`, `containedInPlace`), `geo.position` / ICBM meta tags and
`og:latitude` are lost. Add a `page_structured_data(page_id, kind, json)` table
written at fetch time and a `tier 0: schema_org` in enrichment, confidence 1.0.
The four sources re-fetch in minutes. Gate: the castle site's pages carry a
JSON-LD footprint or the run log says none was published. Effort: half a day.

## 5. Sequence, cost and gates

| step | packages | measured by | wall time |
|---|---|---|---|
| 1 | WP1 (done 2026-09-09) | rerun 3 reranked geo cells, 500 q | 1 day + 40 min |
| 2 | WP2 + WP3 | `locate-pages --apply` on a DB snapshot, rerun geo slice | 3 days + 40 min |
| 3 | WP6 | probe set judged, coverage column in report | 2 days |
| 4 | WP4 + WP5 | dry-run rejection counts, index rebuild, geo slice | 3 days + rebuild |
| 5 | WP7 | re-fetch, tier-0 rows | 0.5 day |

Do WP1 before anything else: it is the only package that can be measured on
the existing data and it decides whether the filter shape or the footprint
coverage was the larger problem. Do WP6 before WP4, because the gazetteer
work has no honest measurement without a probe set that contains real spatial
questions.

Environment constraints from the handoff still apply: work on a
`PAGES_DB_PATH` snapshot (re-chunking cascade-deletes labels; enrichment does
not, but the rule stands), vacuum `data/db/pages.db` before pushing, launch
reruns through `experiments/indexing/launch_cron.sh`, and do not quote
nemotron 1B cells until its weights are restored.

## 6. Deliberately not doing

- Embedding location text into chunks (`v3` measured no effect).
- Trusting model-produced coordinates, in any tier.
- A polygon library or a spatial database now: 176 pages and 726 chunks are
  served by ray casting and a bounding box. Revisit at the Qdrant move.
- Hard filtering as the default shape, for any scope level.
- Vernacular-region modelling: NUTS-3 SI036 is an adequate proxy for Posavje
  until a question proves otherwise.
- Direct coordinate prediction by the model. Mioduski (2025) shows it is
  viable for descriptive historical text (19 to 23 km mean error), but the
  project decision stands: the model names, the gazetteer places and verifies.

## 7. References (2023–2026)

Spatial and geographic RAG:
- Yu, Bao, Ning, Peng, Mai, Zhao. Spatial-RAG: Spatial Retrieval Augmented Generation for Real-World Geospatial Reasoning Questions. arXiv 2502.18470, 2025. https://arxiv.org/abs/2502.18470
- Wang, Zhao, Wang, Cheng, Nie, Luo, Yu, Yuan. GeoRAG: A Question-Answering Approach from a Geographical Perspective. arXiv 2504.01458, 2025. https://arxiv.org/abs/2504.01458
- Chen et al. GeoRAG: A Geographic RAG Framework Based on Urban Spatio-Temporal Knowledge Graph. CLNLP 2025, Springer LNCS. https://link.springer.com/chapter/10.1007/978-981-95-4788-3_12
- Arabov, Khaybullina, Naumetova. Tatarstan Toponyms: A Bilingual Dataset and Hybrid RAG System for Geospatial QA. CLIB 2026, arXiv 2605.05962. https://arxiv.org/abs/2605.05962
- Kang et al. Intelligent Multimodal Retrieval and Reasoning for Geospatial Knowledge Discovery on I-GUIDE. arXiv 2606.15838, 2026. https://arxiv.org/abs/2606.15838
- Yang, Li, Wang. CubeGraph: Efficient RAG for Spatial and Temporal Data. arXiv 2604.06616, 2026. https://arxiv.org/abs/2604.06616
- Campo, Conde, Alonso, Huecas, Salvachúa, Reviriego. Real-time Spatial RAG for Urban Environments. ACM TIST 2026, arXiv 2505.02271. https://arxiv.org/abs/2505.02271
- Amendola, Pugliese, Perego, Renso. Spatially-Enhanced RAG for Walkability and Urban Discovery (WalkRAG). arXiv 2512.04790, 2025. https://arxiv.org/abs/2512.04790
- Li, Lim. RALLM-POI: Retrieval-Augmented LLM for Zero-shot Next POI Recommendation with Geographical Reranking. PRICAI 2025, arXiv 2509.17066. https://arxiv.org/abs/2509.17066
- Ni et al. TP-RAG: Benchmarking RAG LLM Agents for Spatiotemporal-Aware Travel Planning. EMNLP 2025, arXiv 2504.08694. https://arxiv.org/abs/2504.08694
- Banerjee, Satish, Wörndl. Enhancing Tourism Recommender Systems for Sustainable City Trips Using RAG. RecSoGood@RecSys 2024, arXiv 2409.18003. https://arxiv.org/abs/2409.18003
- Dorobantu, Badea. Geospatial reasoning and awareness in LLMs: a systematic review. Artificial Intelligence Review 59:111, 2026. https://link.springer.com/article/10.1007/s10462-026-11512-x

LLM toponym resolution and document geolocation:
- Hu, Kersten, Klan, Farzana. Toponym resolution leveraging lightweight and open-source LLMs and geo-knowledge. IJGIS 2024. https://doi.org/10.1080/13658816.2024.2405182
- Hu, Kersten, Klan. Scalable Toponym Resolution with LLMs: Accuracy and Speed Optimizations. GeoExT@ECIR 2025, CEUR 3969. https://ceur-ws.org/Vol-3969/paper6.pdf
- Li, Zhou, Chiang, Chen. GeoLM: Empowering Language Models for Geospatially Grounded Language Understanding. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.317/
- Halterman. Mordecai 3: A Neural Geoparser and Event Geocoder. arXiv 2303.13675, 2023. https://arxiv.org/abs/2303.13675
- Bisiani, Gulyas, Heravi. Towards efficient and accessible geoparsing of U.K. local media. Computational Humanities Research 2025. https://www.cambridge.org/core/journals/computational-humanities-research/article/ED2946D9A21D1A1E86D4A8E1F4EEF193
- Mioduski. Benchmarking LLMs for Geolocating Colonial Virginia Land Grants. JOSIS 31, 2025, arXiv 2508.08266. https://arxiv.org/abs/2508.08266
- Fernando, Ranathunga, Stock, Prasanna, Jones. Georeferencing complex relative locality descriptions with LLMs. IJGIS 2026, arXiv 2512.14228. https://arxiv.org/abs/2512.14228
- Cafferata, Demarco, Kalimeri, Mejova, Beiró. Large Language Models for Geolocation Extraction in Humanitarian Crisis Response. arXiv 2602.08872, 2026. https://arxiv.org/abs/2602.08872
- Kopanov. Comparative Performance of NLP Models and LLMs in Multilingual Geo-Entity Detection. AICCONF 2024, arXiv 2412.20414. https://arxiv.org/abs/2412.20414
- Masis, O'Connor. Where on Earth Do Users Say They Are? Geo-Entity Linking for Noisy Multilingual User Input. NLP+CSS@NAACL 2024, arXiv 2404.18784. https://arxiv.org/abs/2404.18784
- Ljubešić et al. CLASSLA-Stanza: the next step for linguistic processing of South Slavic languages. arXiv 2308.04255, 2023. https://arxiv.org/abs/2308.04255

Spatial relations, geo-intent and spatial SQL:
- GeoSQL-Eval / GeoSQL-Bench. arXiv 2509.25264, 2025. https://arxiv.org/abs/2509.25264
- Staniek, Schumann, Züfle, Riezler. Text-to-OverpassQL. TACL 12, 2024. https://aclanthology.org/2024.tacl-1.31/
- Ji, Gao, Nie, Majić, Janowicz. Foundation Models for Geospatial Reasoning: Geometries and Topological Spatial Relations. IJGIS 2025, arXiv 2505.17136. https://arxiv.org/abs/2505.17136
- Han, Wolfe, Caspi, Howe. Can LLMs Integrate Spatial Data? arXiv 2508.05009, 2025. https://arxiv.org/abs/2508.05009
- Truong, Lau, Qi. GPSBench. arXiv 2602.16105, 2026. https://arxiv.org/abs/2602.16105

Filtering, fusion and metadata in retrieval:
- Bruch, Gai, Ingber. An Analysis of Fusion Functions for Hybrid Retrieval. ACM TOIS 2023. https://dl.acm.org/doi/10.1145/3596512
- Shi, Cai, Zheng. Filtered ANN Search: A Unified Benchmark and Systematic Experimental Study. arXiv 2509.07789, 2025. https://arxiv.org/abs/2509.07789
- Li, Yan, Lu, Zhang, Cheng, Ma. Attribute Filtering in ANN Search: An In-depth Experimental Study. SIGMOD 2026, arXiv 2508.16263. https://arxiv.org/abs/2508.16263
- Ingber, Liberty. Accurate and Efficient Metadata Filtering in Pinecone's Serverless Vector Database. ICML 2025. https://www.pinecone.io/research/ICML_2025.pdf
- Ye, Yan, Lo. Compass: General Filtered Search across Vector and Structured Data. arXiv 2510.27141, 2025. https://arxiv.org/abs/2510.27141
- Wu et al. STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases. NeurIPS 2024 D&B. https://openreview.net/forum?id=QSS5cGmKb1
- Poliakov, Shvai. Multi-Meta-RAG. ICTERI 2024, arXiv 2406.13213. https://arxiv.org/abs/2406.13213

Evaluation:
- Dihan et al. MapEval. ICML 2025, arXiv 2501.00316. https://arxiv.org/abs/2501.00316
- Li, Grossman, Qasemi, Kulkarni, Chen, Chiang. MapQA: Open-domain Geospatial QA on Map Data. SIGSPATIAL 2025, arXiv 2503.07871. https://arxiv.org/abs/2503.07871
- Saeedan, Rashid, Eldawy, Hristidis. GS-QA. arXiv 2605.22811, 2026. https://arxiv.org/abs/2605.22811
- Krechetova, Kochedykov. GeoBenchX. SIGSPATIAL GeoGenAgent 2025, arXiv 2503.18129. https://arxiv.org/abs/2503.18129
- Böckling, Nosova, Paulheim, Iana. MultiGlobeQA. arXiv 2608.03882, 2026. https://arxiv.org/abs/2608.03882
- Xu et al. Evaluating LLMs on Spatial Tasks: A Multi-Task Benchmarking Study. arXiv 2408.14438, 2024. https://arxiv.org/abs/2408.14438

Vendor and industry write-ups:
- Qdrant. A Complete Guide to Filtering in Vector Search, 2024. https://qdrant.tech/articles/vector-search-filtering/ and filtering docs https://qdrant.tech/documentation/search/filtering/
- Elastic Search Labs. Geo-semantic search to refine recommendations, 2024. https://www.elastic.co/search-labs/blog/geo-semantic-search-elasticsearch
- Elastic Search Labs. Multimodal RAG with Elasticsearch geospatial capabilities, 2025. https://www.elastic.co/search-labs/blog/multimodal-rag-elasticsearch-geospatial
- Elastic Search Labs. LangChain self-querying retriever with Elasticsearch, 2025. https://www.elastic.co/search-labs/blog/self-querying-retrievers
- Alonso. Geospatial RAG on Postgres, 2026. https://www.pedroalonso.net/blog/geospatial-rag-postgres/
- Milvus. Hybrid Spatial and Vector Search with Milvus 2.6.4, 2026. https://milvus.io/blog/hybrid-spatial-and-vector-search-with-milvus-264.md
- Weaviate filters (withinGeoRange) https://docs.weaviate.io/weaviate/search/filters ; LlamaIndex VectorIndexAutoRetriever https://developers.llamaindex.ai/python/framework/integrations/vector_stores/chroma_auto_retriever/ ; LangChain self-query https://js.langchain.com/docs/how_to/self_query/

Standards (undated by nature): Eurostat NUTS 2024 and LAU (GISCO); ISO 3166-1/-2; Wikidata P625, P131, P17, P276, P706, P1082; GeoNames feature codes; schema.org Place / GeoCoordinates / containedInPlace; OSMF Nominatim usage policy; H3 and S2 cell hierarchies.

Not found in the 2023–2026 literature: a paper quantifying recall loss from *incomplete* metadata coverage in RAG (the filtered-ANN benchmarks vary selectivity, not missingness); a dedicated geospatial-RAG survey; any Slovenian-specific geoparsing evaluation; any vendor post applying a geo decay inside a RAG pipeline rather than a hard filter.
