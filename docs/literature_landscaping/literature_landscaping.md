# LLMs4EU Tourism: Landscape and Task Scoping

Single source for the literature review task. The hub holds current beliefs and
links; reasoning and evidence live in the spokes.

## Links

- [consolidation-2026-10-06.md](consolidation-2026-10-06.md): coverage table, concept map, hypotheses with novelty verdicts, what the proxy data can support, low-hanging improvements, next steps (replaces the 2026-10-03 version)
- [sources/novelty-check-2026-10-06.md](sources/novelty-check-2026-10-06.md): targeted search on location × language bias and diversity re-ranking
- [decisions-assumptions.md](decisions-assumptions.md): full decision table with assumptions and open checks
- [question-types.md](question-types.md): question dimensions and candidate question types for the official dataset
- [multi-stakeholder.md](multi-stakeholder.md): tourism as a multi-stakeholder problem; the tiktokrijen loop
- [datasets.md](datasets.md): dataset catalog, one row per dataset
- [sources/README.md](sources/README.md): the source card format
- [sources/team-list.md](sources/team-list.md): the team's existing literature, as reference

## 1. Task definition

Scan the literature for new ideas, techniques, knowledge and evaluation methods
that go beyond the team's baseline, and that either come from spatial,
location-aware or tourism RAG, or transfer to a problem this domain has:
location, multilinguality and cross-border retrieval, bias toward
well-documented places, and scale. Tourism-specific work comes first. Generic
RAG engineering is in scope only when it addresses one of these domain
problems. I also map existing tourism and geo benchmarks, so the December
dataset can be positioned quickly. I judge papers by their ideas, not their
reported performance, because neither their data nor the team's proxy data
predicts the December setting. Each paper gets a 15-minute scan and a verdict
relative to the baseline; only papers that add something or contradict the
team's choices get a full card. Out of scope for now: agentic retrieval,
storage, and the hidden-gems recommender itself.

**Status (2026-10-06):** reading phase closed; all high-priority coverage cells
addressed. Next: experiment proposals for the team (applied focus: system,
evaluation, proxy dataset).

---

## 2. Project context

- **End goal:** recommend "hidden gems" across Europe: places interesting for tourists that get fewer visitors `[team: Pepe]`.
- **Current step:** compare retrieval strategies, measured on retrieval accuracy and speed, not yet targeting hidden gems `[team: Pepe, Gergely]`.
- **Data:** today's corpus is a proxy (1,338 pages, 8 languages, 8 countries); the official ~300K-document multilingual dataset arrives around December with unknown questions and gold `[team]`. Details: consolidation §4.
- **Team concerns** `[team: Pepe]`: documentation skew favours well-known places; language and format may favour same-country places over closer cross-border ones; coordinates are being added to chunk metadata.

---

## 3. Current beliefs

### Framing

- **Documented ≠ visited ≠ crowded.** Documentation volume (corpus), visitor
  popularity (reviews, page views, arrivals) and crowding (time-varying) are
  different signals; how you measure decides whether they correlate, and often
  they don't [src: Banerjee 2025]. No paper found uses documentation volume
  (chunk count) as its popularity measure [src: novelty check]. For small
  places, the corpus may be the only available signal.
- **Availability vs selection.** Availability: can the system answer correctly
  when a place is named (factual questions)? Selection: does the place get
  chosen when it isn't named (recommendation questions)? Knowing a place is
  necessary but not sufficient for recommending it [src: Najafi & Costa].
  Evaluation should be split the same way.
- **Where the bias enters.** Measure at three points for the same questions:
  LLM alone, retriever top-k, LLM with retrieval [precedent: PopQA compares
  LLM alone with LLM + retrieval; the retriever-only point is ours].
- **Corpus skew ≠ model skew.** A place thin in the corpus may be known to the
  LLM; retrieval can then override correct knowledge. Adaptive retrieval
  (retrieve only below a popularity threshold) is the known fix [src: PopQA];
  it is moot if the local model knows nothing about our places (closed-book
  test). See T9.
- **Language bias enters at retriever, reranker and generator**
  [src: Park & Lee; All Languages Matter].
- **Tourism RAG papers evaluate answers, rarely recommendations** (quality,
  diversity, popularity mix) [me, across ~6 papers].
- **LLM judges need human calibration.** Judges matched the expert majority
  only 17–54% of the time on recommendation lists [src: Banerjee 2026,
  LLM-judge]. This also applies to the team's own uncalibrated LLM judgments.
- **Keep objectives as separate, explicit scores.** One LLM asked to balance
  several goals quietly favours one, usually famous places ("objective
  collapse") [src: Collab-Rec]. The weighted-sum design guards against this.
- **Tourism is multi-stakeholder** (tourists, residents, businesses,
  environment); the tiktokrijen loop shows how visibility becomes crowding [me].
  Details: [multi-stakeholder.md](multi-stakeholder.md).

### Location

- **Constraint vs score.** A constraint must say yes or no to a page with
  unknown location; a score can stay neutral. With half the pages unlocated,
  this is why the hard filter failed `[src: Spatial-RAG]` `[docs]` `[me]`.
  Hard location filters in the literature assume complete metadata
  [src: Tandon; LAMB].
- **The geo score can create its own concentration.** Across many queries, a
  distance score can favour places near common anchors, and places whose
  location is *known*. Location metadata comes mostly from Wikidata, so it
  correlates with fame; "unknown = neutral" may then favour well-documented
  places. It depends on *why* a page is unlocated: not a place (fine) or too
  obscure to locate (biased) [src: novelty check; Forster et al.] `[me]`.
- **Place as disambiguator (hypothesis).** The team's questions mostly use the
  place to say *which* entity is meant, not for spatial reasoning; the geo-RAG
  literature mostly targets spatial reasoning. Possible positioning angle;
  check against real questions `[me]` `[claude]`.
- **Location is language-independent.** A distance score can counter language
  bias across borders, which gives the geo layer a purpose beyond
  disambiguation `[claude]`. See H1–H5.

### Techniques (current view)

- **Only post-processing fits a RAG system** (re-scale, re-rank, aggregate);
  pre- and in-processing need interaction data [src: popularity-bias survey].
- **In RAG, act per place, not per chunk.** Skew works through chunk count, so
  down-weighting chunks is not enough. Options: a per-place cap in the top-k,
  place-level scoring (average of a place's top-n passages, as in EQR's
  Algorithm 1), or xQuAD with places as aspects `[claude]` [src: Wen et al.].
- **Quotas are a general training-free tool:** per language (Amiraz), per
  place, per popularity tier. A per-place cap has no published evidence yet
  [src: novelty check].
- **Diversity re-ranking (MMR, set selection) helps in some RAG studies and
  hurt answer relevance in one** [src: novelty check]. Test, don't assume.
- **Re-ranking cannot surface what never entered the candidate pool**
  [src: Forster et al.]; the over-fetch size matters for hidden gems.
- **Language debiasing per stage:** retriever, balanced quota (Amiraz,
  training-free); reranker, utility-based LAURA (needs answer-labelled
  training data); generator, translate into the query language (DKM-RAG).
  Translation adds little for dense retrieval [src: novelty check].
- **Broad and indirect queries:** LLM query reformulation into subtopics
  with elaborations (EQR), prompt-only [src: Wen et al.].
- **Keep any anti-popularity weight small;** users accept only a light shift
  toward less popular items [src: survey, Steck 2011].

### Hypotheses, with novelty verdicts

Verdicts from the novelty check (targeted, not systematic).

- **H1** Geo score counteracts language bias for nearby other-language pages. *None found.*
- **H2** Language quota only for languages spoken near the anchor place. *None found;* BORDIRLINES picks languages by country, the closest counter-evidence.
- **H3** Location-first candidates, then text ranking (language-blind pool). *Partial:* mechanism exists, never measured for language bias.
- **H4** Place as retrieval unit (one score per Wikidata ID, all languages). *Partial:* entity-as-unit exists monolingually; no cross-language merging.
- **H5** Location decides which passages get translated. *None found;* low priority (translation adds little for dense retrieval).
- **H6** A location/context score reduces concentration on well-documented or popular places. *Prior work in recommenders, mixed results* (one geo model raised popularity bias by 40%); *none in RAG.* Must be measured in both directions.

**Positioning (parked):** location × language bias is a defensible novelty
claim if framed as bias mitigation in multilingual dense RAG, not as
cross-lingual geographic IR (GeoCLEF 2005–2009). Neighbouring languages are
already retrieved better, so Slovenian–Croatian may show little gap; test
Slovenian–German and Slovenian–Hungarian too [src: novelty check].

### Evaluation toolkit (data-independent)

- Popularity / documentation stratification [SynthTRIPs, PopQA]
- Concentration and effective diversity: HHI, Top-k share, entropy [Najafi & Costa]; measured across many queries, in both directions
- Novelty, coverage, geographic spread, local share [Çelik]
- Language preference: MLRS [Park & Lee]; language share of top-k before vs after reranking [All Languages Matter]; citation language [BORDIRLINES]
- Intent / question-type stratification [Tandon]
- Three-point measurement: LLM alone / retriever / LLM + retrieval; closed-book test for the LLM alone [PopQA]
- Place exposure in answers: which places an answer names, against chunk count and a visitor signal (no standard metric exists; ours to define) [novelty check]
- Pairwise, per-dimension LLM judging with a human-calibrated sample [Banerjee 2026, LLM-judge]
- Small exhaustive human-labelled set for recommendation questions [TravelDest]
- Candidate pool size, local vs global [LAMB]

### Other concerns

- **Gold answers:** how they are produced decides what is measurable.
- **Scale:** at ~300K documents, speed and filtered search become real concerns; at 726 chunks they did not `[docs: geo-literature]`.
- **Experiment vs real world:** generated questions reuse the gold chunk's wording, which flatters text similarity; real users ask vaguely, with preferences, in other languages `[claude]`.
- **Deployment constraints:** data licensing, GDPR, EU rules on recommenders `[claude]`.
- **Signals not yet available:** visitor popularity (proxies: Wikipedia page views, which exist only for places with an article and differ per language edition) and quality ("worth a visit", source unknown).

### Dependencies

- Location data before any geo task.
- Place ID (Wikidata) in chunk metadata before per-place metrics, caps or aggregation.
- Set-based gold questions before evaluating set or ranked answers.
- Documentation, popularity and quality signals before evaluating recommendations.
- Crowding data and personal profiles need external data; out of reach now.

---

## 4. Decision baseline

One line per decision in the current implementation; assumptions, reasons and
open checks in [decisions-assumptions.md](decisions-assumptions.md).

| # | decision | still valid? |
| - | -------- | ------------ |
| 1 | Coordinates are the true fact; region codes computed from them | Data-dependent (broke for Miramare; routes and areas aren't points) |
| 2 | Place names for display only, never matching | Stable |
| 3 | Regions use EU NUTS codes | Data-dependent (fails outside EU, natural regions) |
| 4 | Three location tiers: Wikidata, source default, LLM + gazetteer | Data-dependent (new source types may break tier 2) |
| 5 | Model names places, never outputs coordinates | Data-dependent (tested on Slovenian only) |
| 6 | Location is a soft score; unknown = neutral | Likely holds; risk: may favour places whose location is known (well-documented) |
| 7 | Region used only if it excludes >10% of located pages | Data-dependent (gate now fires, unmeasured) |
| 8 | Strict filter version kept alongside | Stable |
| 9 | Location in simple metadata, not store-specific features | Stable for now (4× over-fetch costs speed at scale; over-fetch size also limits which hidden gems can surface) |

---

## 5. Open questions

### For the team

| #  | question | status |
| -- | -------- | ------ |
| T1 | Goal? | **Answered:** hidden gems is the end goal; retrieval accuracy and speed now `[team]` |
| T2 | New question types or more of the same? | **Answered, provisional:** both, in December `[team]` |
| T3 | First task focus? | **Answered:** immediate questions (chunk sizes etc.) first, then the broader strands `[team: Pepe]` |
| T4 | Do partners have popularity, visitor or crowding data? | Deferred |
| T5 | In which languages will users ask? | Partly answered: multilingual, several countries |
| T6 | Has geo retrieval been rerun on the expanded proxy corpus? | Open |
| T7 | Is local-only inference still a principle, given a hosted model is the geo resolver? | Open |
| T8 | What will the official dataset contain (sources, languages, question types, format)? | Open, likely unknown until December |
| T9 | May the chatbot use the LLM's own knowledge, or must every answer be grounded in the corpus? | Open; decides whether adaptive retrieval is an option |
| T10 | Which harm should the system target: the retrieval bias (documentation) or crowding (visits)? | Open; prerequisite for choosing any signal [src: popularity-bias survey] |

### For the literature

Reading phase closed. Coverage table, hypotheses and remaining gaps:
[consolidation-2026-10-06.md](consolidation-2026-10-06.md). Further threads
(geo-score exposure effects, exposure metrics for RAG answers, open visitor
signals for Europe, per-language-pair bias numbers, generation-stage
mitigation, unsearched venues) are listed in the novelty check; most are
better answered by experiments.

Landscape, revisit: substitutability ("valid alternatives"), crowding-aware
recommendation, EU regulation, quality signals.

---

## 6. Method notes

- **Read with a question.** Before reading, write what I want to get out of
  the text; afterwards, produce an artifact that answers it, plus what the text did not answer.
- **Assumption test.** "This decision would be wrong if …". Use when inheriting decisions, before conditions change, when judging whether a result transfers, and when moving from experiment to the real world.
- **Direct vs adjacent literature.** No direct work on our use case doesn't
  mean nothing applies: look at adjacent fields (popularity bias in
  recommenders, long-tail QA, entity disambiguation, multilingual retrieval) and state the transfer assumption.
- **Saturation is a stopping signal.** When new papers mostly confirm, switch
  strand or switch from reading to applying.
- **Understanding vs remembering.** Keep the structure in my head and the
  details in this document. Maintain the document, or it decays.

---

## 7. Glossary

- **Gazetteer:** database of place names with coordinates (Wikidata, Nominatim, GeoNames).
- **Toponym resolution:** turning a place name in text into a specific location.
- **Footprint:** the location a page is about. "Unlocated" = page has none.
- **NUTS codes:** EU region hierarchy (country, large region, small region). SI036 = Posavska.
- **Hard filter vs soft score:** excluding pages outside an area vs ranking them lower.
- **Constraint vs score:** what is allowed vs what ranks higher.
- **Disambiguator:** a place in a question that says which entity is meant ("the castle *in Brestanica*").
- **hit@5:** share of questions with a correct page in the top 5 results.
- **Documentation skew:** well-known places having more text in the corpus.
- **Documentation volume:** how much corpus text exists about a place; a signal distinct from visitor popularity and crowding.
- **Availability vs selection:** answering correctly about a named place vs choosing the place when it isn't named.
- **Three-point measurement:** measuring the same questions on the LLM alone, the retriever's top-k, and the LLM with retrieval, to locate where a bias enters.
- **Adaptive retrieval:** retrieving only when the model is unlikely to know the answer itself.
- **Long tail:** the many items (places, entities) that are each rarely mentioned.
- **Popularity bias:** systems over-suggesting what is already popular.
- **Exposure:** how often an item appears in results across many queries.
- **Concentration (HHI):** sum of squared result shares per place; high means a few places dominate.
- **Effective diversity:** the number of equally-shown places that would give the same concentration; nominal diversity is just the count of distinct places.
- **Horizontal substitution:** "alternatives" that stay within the familiar circuit (Santorini → Milos) instead of reaching the periphery.
- **Objective collapse:** one model asked to balance several goals quietly optimizing only one.
- **Quota / cap:** a limit on how many top-k results may come from one language, place or popularity tier.
- **MMR / xQuAD:** re-ranking methods that trade relevance against redundancy or aspect coverage.
- **MLRS:** label-free measure of how far documents rise when translated into one shared language, i.e. the rank effect of language alone.
- **Digital overtourism / tiktokrijen:** crowding amplified by what platforms show; the viral-visit-post-visibility loop.

---

## 8. Source log

Coverage by domain problem and goal: consolidation §1.
Cards: [sources/](sources/) (format in [sources/README.md](sources/README.md)).
Team sources: [sources/team-list.md](sources/team-list.md).
Deferred with reasons: consolidation §6.