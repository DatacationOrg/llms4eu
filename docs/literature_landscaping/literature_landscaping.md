# LLMs4EU Tourism: Landscape and Task Scoping

Single source for the literature review task. The hub holds current beliefs and
links; reasoning and evidence live in the spokes.

## Links

- [consolidation-2026-10-03.md](consolidation-2026-10-03.md): coverage table, concept map, what the proxy data can support, low-hanging improvements, next steps
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

---

## 2. Project context

- **End goal:** recommend "hidden gems" across Europe: places interesting for tourists that get fewer visitors `[team: Pepe]`.
- **Current step:** compare retrieval strategies, measured on retrieval accuracy and speed, not yet targeting hidden gems `[team: Pepe, Gergely]`.
- **Data:** today's corpus is a proxy (1,338 pages, 8 languages, 8 countries); the official ~300K-document multilingual dataset arrives around December with unknown questions and gold `[team]`. Details: consolidation §3.
- **Team concerns** `[team: Pepe]`: documentation skew favours well-known places; language and format may favour same-country places over closer cross-border ones; coordinates are being added to chunk metadata.

---

## 3. Current beliefs

### Framing

- **Documented ≠ visited ≠ crowded.** Documentation volume (corpus), visitor
  popularity (reviews, page views, arrivals) and crowding (time-varying) are
  different signals; how you measure decides whether they correlate, and often
  they don't [src: Banerjee 2025]. For small places, documentation volume from
  the corpus may be the only available signal.
- **Availability vs selection.** Availability: can the system answer correctly
  when a place is named (factual questions)? Selection: does the place get
  chosen when it isn't named (recommendation questions)? Knowing a place is
  necessary but not sufficient for recommending it [src: Najafi & Costa].
  Evaluation should be split the same way.
- **Where the bias enters.** Measure at three points for the same questions:
  LLM alone, retriever top-k, LLM with retrieval [precedent: PopQA].
- **Corpus skew ≠ model skew.** A place thin in the corpus may be known to the
  LLM; retrieval can then override correct knowledge. Candidate fix: adaptive
  retrieval [src: PopQA, to read]. See T9.
- **Language bias enters at retrieval and at generation** [src: Park & Lee].
- **Tourism RAG papers evaluate answers, rarely recommendations** (quality,
  diversity, popularity mix) [me, from 4 papers].
- **Tourism is multi-stakeholder** (tourists, residents, businesses,
  environment); the tiktokrijen loop shows how visibility becomes crowding [me].
  Details: [multi-stakeholder.md](multi-stakeholder.md).

### Location

- **Constraint vs score.** A constraint must say yes or no to a page with
  unknown location; a score can stay neutral. With half the pages unlocated,
  this is why the hard filter failed `[src: Spatial-RAG]` `[docs]` `[me]`.
- **Place as disambiguator (hypothesis).** The team's questions mostly use the
  place to say *which* entity is meant, not for spatial reasoning; the geo-RAG
  literature mostly targets spatial reasoning. Possible positioning angle; check
  against real questions `[me]` `[claude]`.
- **Location is language-independent.** A distance score can counter language
  bias across borders, which gives the geo layer a purpose beyond
  disambiguation `[claude]`. See H1–H5.

### Open hypotheses (geo × language) `[claude]`

- H1 Geo score counteracts language bias for nearby other-language pages.
- H2 Language quota only for languages spoken near the anchor place.
- H3 Location-first candidates, then text ranking (language-blind pool).
- H4 Place as retrieval unit (one score per Wikidata ID, all languages).
- H5 Location decides which passages get translated.
- H6 the soft geo score may already reduce popularity concentration (after Rahmani et al.). It's testable directly: concentration with and without the geo score.

### Evaluation toolkit (data-independent)

- Popularity / documentation stratification [SynthTRIPs, PopQA]
- Concentration and effective diversity: HHI, Top-k share, entropy [Najafi & Costa]
- Novelty, coverage, geographic spread, local share [Çelik]
- Language preference: MLRS [Park & Lee]; citation language [BORDIRLINES]
- Intent / question-type stratification [Tandon]
- Three-point measurement: LLM alone / retriever / LLM + retrieval
- Candidate pool size, local vs global [LAMB]

### Other concerns

- **Gold answers:** how they are produced decides what is measurable.
- **Scale:** at ~300K documents, speed and filtered search become real concerns; at 726 chunks they did not `[docs: geo-literature]`.
- **Experiment vs real world:** generated questions reuse the gold chunk's wording, which flatters text similarity; real users ask vaguely, with preferences, in other languages `[claude]`.
- **Deployment constraints:** data licensing, GDPR, EU rules on recommenders `[claude]`.
- **Signals not yet available:** popularity (proxies: Wikipedia page views, Wikidata sitelinks) and quality ("worth a visit", source unknown).

### Dependencies

- Location data before any geo task.
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
| 6 | Location is a soft score; unknown = neutral | Likely holds |
| 7 | Region used only if it excludes >10% of located pages | Data-dependent (gate now fires, unmeasured) |
| 8 | Strict filter version kept alongside | Stable |
| 9 | Location in simple metadata, not store-specific features | Stable for now (4× over-fetch costs speed at scale) |

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
| T10 | which harm should the system target, the retrieval bias (documentation) or crowding (visits)? The recommender system survey's critique makes this the prerequisite for choosing any signal. |

### For the literature

Search task v3, the coverage table and the main gaps:
[consolidation-2026-10-03.md](consolidation-2026-10-03.md) §1.

Landscape, revisit: popularity bias in recommenders, exposure measurement,
substitutability ("valid alternatives"), crowding-aware recommendation, EU
regulation, quality signals.

---

## 6. Method notes

- **Read with a question.** Before reading, write what I want to get out of
  the text; afterwards, produce an artifact that answers it, plus what the text did not answer.
- **Assumption test.** "This decision would be wrong if …". Use when inheriting decisions, before conditions change, when judging whether a result transfers, and when moving from experiment to the real world.
- **Direct vs adjacent literature.** No direct work on our use case doesn't
  mean nothing applies: look at adjacent fields (popularity bias in
  recommenders, long-tail QA, entity disambiguation, multilingual retrieval) and state the transfer assumption.
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
- **Effective diversity:** the number of equally-shown places that would give the same concentration.
- **Digital overtourism / tiktokrijen:** crowding amplified by what platforms show; the viral-visit-post-visibility loop.

---

## 8. Source log

Coverage by domain problem and goal: consolidation §1.
Cards: [sources/](sources/) (format in [sources/README.md](sources/README.md)).
Team sources: [sources/team-list.md](sources/team-list.md).

---

## 9. Changelog

- **2026-09-30, v0.** Document created.
- **2026-09-30, superseded:** a single five-rung ladder. *Reason:* it mixed
  task, objective and data; offline exposure measurement is possible earlier.
  Replaced by the three-axis map.
- **2026-09-30, superseded:** "the spatial signal is a constant, so geography
  cannot help" `[docs: geo-rationale.md]`. *Reason:* measured on 176 pages; the
  expansion to 10 localities and 8 countries removes the premise.
- **2026-10-01, v1.** Team answers added; decision table, method notes,
  question dimensions and candidate question types added; Spatial-RAG card
  started.
- **2026-10-01, superseded:** "the goal is unclear: scalability, precision or
  exposure". *Reason:* Pepe confirmed the hidden-gems end goal and retrieval
  accuracy as the current step `[team]`.
- **2026-10-01, superseded:** literature questions on popularity bias,
  exposure, substitutability, crowding and regulation as the main strands.
  *Reason:* the team's priorities. Demoted to "landscape, revisit"; replaced by
  L1–L6.
- **2026-10-01, superseded:** "reason from the current data". *Reason:* the
  current corpus is a proxy; the official ~300K-document dataset arrives in
  December with unknown questions and gold passages `[team]`.
- **2026-10-01:** Restructured into hub and spokes; no content changes.
- **2026-10-05, v2.** Hub trimmed to current beliefs and links. Added Framing,
  open hypotheses H1–H5, evaluation toolkit, T9. Moved out: decision table →
  `decisions-assumptions.md`; question dimensions → `question-types.md`;
  multi-stakeholder and tiktokrijen notes → `multi-stakeholder.md`; corpus
  details → consolidation §3. T1–T3 collapsed; glossary extended.
- **2026-10-05, superseded:** the three-axis map (task, objective, signals).
  *Reason:* the problem-centred concept map and the Framing block say the same
  more usefully. Dependencies rewritten without its labels (A2, A3, B2).
- **2026-10-05, superseded:** literature questions L1–L6 and "coverage at a
  glance". *Reason:* replaced by search task v3 and the coverage table in the
  consolidation document.