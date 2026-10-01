# LLMs4EU Tourism: Landscape and Task Scoping (living document)

Version 1, 2026-10-01. Single source for literature review task.

## Links

- [question-types.md](question-types.md): candidate question types for the official dataset
- [datasets.md](datasets.md): dataset catalog, one row per dataset
- [sources/README.md](sources/README.md): the source card format
- [sources/yu-2025-spatial-rag.md](sources/yu-2025-spatial-rag.md): source card, Yu et al. 2025, Spatial-RAG
- [sources/team-list.md](sources/team-list.md): the team's existing literature, as reference

## How to use this document

Provenance tags:

- `[docs]` from the team's docs
- `[team]` from a team member (provisional)
- `[src: X]` from a source card
- `[claude]` Claude's claim, unverified
- `[me]` my own judgment

New knowledge:

1. New source → card in sources/ (or a row in datasets.md).
2. Answers a literature question → update that question's file in questions/
   (create it only when it gets its first real answer).
3. Changes a belief → update the hub and add a changelog entry.

Demote branches, don't delete them.


---

## 1. Task definition

> **Task:** Review the literature on tourism, spatial and location-aware RAG, for two uses: techniques directly usable in
> the current engineering (retrieval accuracy and speed), and an overview of what the field does, to inform future work and position a possible paper. In scope first are chunking and retrieval strategies, location as retrieval metadata, tourism RAG, tourism datasets, bias toward well-documented places, and language and cross-border bias. The hidden-gems recommender stays on the map but is not pursued yet. The output is this document, with an answer paragraph per literature question backed by sources I assessed, a short list of techniques worth trying, and candidate question types for the new dataset. I am working without knowing the December dataset's questions, answers or gold passages; what is known is that it will be about 300K
> documents, multilingual and from several countries.

To finalize: a timebox.

---

## 2. Project context

- **End goal:** democratize tourism across Europe by recommending "hidden
  gems": places interesting for tourists that get fewer visitors `[team: Pepe]`.
- **Current step:** compare retrieval strategies and techniques, measured on retrieval accuracy and speed, without yet targeting hidden gems
  `[team: Pepe, Gergely]`.
- **Current data is a proxy.** The official dataset arrives around December:
  about 300K documents, multilingual, several countries, with both factual and recommendation-style questions expected. Exact questions, answers and gold passages are unknown `[team: Pepe]`.
- **Proxy corpus today:** 1,338 pages, 38 sources, 8 languages, 10 localities; 690 pages located in 8 countries `[docs: decisions.md]`. Evaluated by hit@5 on questions generated from chunks, labels covering only the 176 Brestanica pages`[docs]`.
- **Known concerns for the official data** `[team: Pepe]`:
  - Well-known places will have more documentation, so retrieval will favour them. Techniques that favour under-documented information are desired.
  - Language and format differences may make retrieval prefer places in the country or language of the query over closer ones across a border.
  - Coordinates are being added to chunk metadata (from the source, e.g. Wikipedia, or via an API).

---

## 3. Conceptual map

### Foundations (shared by everything)

- **Corpus:** size (proxy 1,338 pages; official ~300K), languages, countries.
- **Location data:** how many pages are located, and how well.
- **Documentation coverage:** how much text exists per place. Expected to be
  skewed toward well-known places `[team]`. A foundation signal, because it biases retrieval before any recommender exists `[claude]`.
- **Popularity and quality signals:** neither exists yet. Popularity proxies: Wikipedia page views, Wikidata sitelink counts `[claude]`. Source of a quality ("worth a visit") signal: unknown.

### Axis A: the task

1. Answer a question about a place (one gold page)
2. Retrieve a spatially constrained set ("what is near X")
3. Recommend substitutes or hidden gems for stated preferences

### Axis B: the objective

1. Relevance per query
2. Exposure across many queries: are the same few places always returned?
   Measurable offline `[claude]`
3. Real crowding outcomes: needs deployment or simulation

### Axis C: the signals used

text → location → documentation coverage, popularity, quality → time-varying crowding → personal profiles. Further right means harder data to get and to deploy legally `[claude]`.

### Question dimensions (for designing and classifying questions)

- **Role of the place:** constraint (what is allowed), score (what ranks
  higher), disambiguator (which entity is meant), or none.
- **Anchor type:** named place ("in Amsterdam"), the user's location ("near my work"), or a natural or informal region ("in the Alps") `[me]`.
- **Answer form:** one right answer, a set, or a ranked list.
- **Room for hidden gems:** can a lesser-known place be a good answer?
- **Language and borders:** can a good answer lie in another language or
  across a border?

### Cross-cutting concerns

- **Gold answers:** how they are produced decides what is measurable.
- **Scale:** at ~300K documents, speed and filtered search become real
  concerns; at today's 726 chunks they did not `[docs: geo-literature]`.
- **Multilinguality and borders:** 8 languages now, more expected; language
  bias may hide cross-border answers `[team]`.
- **Generation-step bias:** the answering LLM may favour famous places it
  knows from pretraining `[claude]`.
- **Experiment vs real world:** generated eval questions reuse the gold
  chunk's wording, which flatters text similarity; real users ask vaguely, with preferences, in other languages `[claude]`.
- **Deployment constraints:** data licensing, GDPR, EU rules on recommenders
  `[claude]`.

### Dependencies

- Location data before any geo task.
- Set-based gold questions before evaluating A2 or A3.
- Coverage, popularity and quality signals before A3 and B2.
- Crowding data and personal profiles need external data; out of reach now.

### Key insights so far

- **Constraint vs score.** A constraint must say yes or no to a page with
  unknown location; a score can stay neutral. With half the pages unlocated, this is why the hard filter failed `[src: Spatial-RAG]` `[docs]` `[me]`.
- **Place as disambiguator (hypothesis).** The team's questions mostly use the place to say *which* entity is meant, not for spatial reasoning. The geo-RAG literature mostly targets spatial reasoning. Possible positioning angle;
  check against real questions and the literature `[me]` `[claude]`.
- **Location is language-independent.** A distance score can counter language bias across borders, which gives the geo layer a purpose beyond disambiguation `[claude]`.

---

## 4. Decisions in the current implementation

Evaluated for whether they will still hold for the official dataset.
Test: "this decision would be wrong if …". Decisions `[docs: decisions.md, geo-retrieval.md]`; assumptions and verdicts `[me]` refined with Claude.


| # | decision                                                        | reason given                                      | assumption                                                            | still valid?                                                                                                             |
| - | --------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 1 | Coordinates are the true fact; region codes computed from them  | Boundary revisions need only recomputation        | Every place is one point inside a region boundary                     | Data-dependent. Already broke once (Miramare's coastal point lies outside every region); routes and areas are not points |
| 2 | Place names for display only, never matching                    | Names are ambiguous and multilingual              | Codes exist wherever names would be used                              | Stable                                                                                                                   |
| 3 | Regions use EU NUTS codes                                       | Only system consistent across EU                  | All places lie inside the NUTS system                                 | Data-dependent. Fails outside the EU and for natural regions (the Alps)                                                  |
| 4 | Three location tiers: Wikidata, source default, LLM + gazetteer | Cheapest, most reliable first (implied)           | Cheap tiers cover most pages; a single-site source is about one place | Data-dependent. New source types at 300K may break tier 2                                                                |
| 5 | Model names places, never outputs coordinates                   | Models unreliable at coordinates (Hu et al. 2024) | Model extracts names in a form the gazetteer matches                  | Data-dependent. Tested on Slovenian only                                                                                 |
| 6 | Location is a soft score; unknown = neutral                     | Hard filter lost 15 of 72 questions               | Many relevant pages have no location                                  | Data-dependent, likely holds (690 of 1,338 located; biographies never will be)                                           |
| 7 | Region used only if it excludes >10% of located pages           | Otherwise it distinguishes nothing                | Threshold fits the corpus                                             | Data-dependent. With 8 regions the gate now fires; behaviour unmeasured                                                  |
| 8 | Strict filter version kept alongside                            | Comparison; explicit "in region X" requests       | Some users want strict containment                                    | Stable                                                                                                                   |
| 9 | Location in simple metadata, not store-specific features        | Survives Chroma → Qdrant move                    | Geo scoring stays outside the store                                   | Stable for now; at scale the 4× over-fetch costs speed                                                                  |

### Not answered by the texts

- How does the region gate behave now that it fires? (row 7)
- Does place-name extraction work in all corpus languages? (row 5)
- Do the eval questions resemble what real users will ask?
- How does the soft method's over-fetching scale to 300K documents? (row 9)
- Is "local-only inference" still a principle? A hosted model is the geo
  resolver and agent judge (see T7).

---

## 5. Open questions

### For the team


| #  | question                                                                             | status                                                                                                           |
| -- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| T1 | What is the goal?                                                                    | **Answered** `[team]`: hidden-gems recommender is the end goal; retrieval accuracy and speed is the current step |
| T2 | Will the expansion add new question types or more of the same?                       | **Answered, provisional** `[team]`: both, in the December dataset; details unknown                               |
| T3 | What should my first task focus on?                                                  | **Answered** `[team: Pepe]`: immediate questions first (chunk sizes etc.), then L3–L6                           |
| T4 | Do partners have popularity, visitor or crowding data?                               | Deferred                                                                                                         |
| T5 | In which languages will users ask?                                                   | Partly answered: multilingual, several countries                                                                 |
| T6 | Has geo retrieval been rerun on the expanded proxy corpus?                           | Open                                                                                                             |
| T7 | Is local-only inference still a principle, given a hosted model is the geo resolver? | Open                                                                                                             |
| T8 | What will the official dataset contain (sources, languages, question types, format)? | Open, likely unknown until December                                                                              |

### For the literature

Ordered by Pepe's priorities `[team]`.


| #  | question                                                                                                                                                          | when | status |
| -- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------ |
| L1 | Chunking and retrieval strategies: what is known about chunk size, hybrid search, reranking, and what is worth trying? Start from the team's own chunk-size sweep | now  | open   |
| L2 | Location as metadata in retrieval: who does it, how? Audit and fill the team's list                                                                               | now  | open   |
| L3 | Who works on RAG or retrieval for tourism data?                                                                                                                   | next | open   |
| L4 | Which tourism place datasets exist, and what do they look like?                                                                                                   | next | open   |
| L5 | How can retrieval favour under-documented places (long-tail bias)?                                                                                                | next | open   |
| L6 | How do language and border effects bias multilingual retrieval?                                                                                                   | next | open   |

Landscape, revisit: popularity bias in recommenders, exposure measurement,
substitutability ("valid alternatives"), crowding-aware recommendation, EU
regulation, quality signals.

---

## 6. Candidate question types for the official dataset

Moved to [question-types.md](question-types.md).

---

## 7. Method notes

- **Read with a question.** Before reading, write what I want to get out of
  the text; afterwards, produce an artifact that answers it, plus what the text did not answer.
- **Assumption test.** "This decision would be wrong if …". Use when inheriting decisions, before conditions change, when judging whether a result transfers, and when moving from experiment to the real world.
- **Direct vs adjacent literature.** No direct work on our use case doesn't
  mean nothing applies: look at adjacent fields (popularity bias in
  recommenders, long-tail QA, entity disambiguation, multilingual retrieval) and state the transfer assumption.
- **Understanding vs remembering.** Keep the structure in my head and the
  details in this document. Maintain the document, or it decays.

---

## 8. Glossary

- **Gazetteer:** database of place names with coordinates (Wikidata, Nominatim, GeoNames).
- **Toponym resolution:** turning a place name in text into a specific location.
- **Footprint:** the location a page is about. "Unlocated" = page has none.
- **NUTS codes:** EU region hierarchy (country, large region, small region). SI036 = Posavska.
- **Hard filter vs soft score:** excluding pages outside an area vs ranking them lower.
- **Constraint vs score:** what is allowed vs what ranks higher.
- **Disambiguator:** a place in a question that says which entity is meant ("the castle *in Brestanica*").
- **hit@5:** share of questions with a correct page in the top 5 results.
- **Documentation skew:** well-known places having more text in the corpus.
- **Long tail:** the many items (places, entities) that are each rarely mentioned.
- **Popularity bias:** systems over-suggesting what is already popular.
- **Exposure:** how often an item appears in results across many queries.

---

## 9. Source log

### Coverage at a glance

The team's list (`geo-literature-2023-2026.md`) is almost entirely about
retrieval mechanics (A1–A2) and location data (foundations). One entry
(Banerjee et al.) touches popularity. Nothing covers documentation skew,
cross-border language bias, tourism datasets or chunking `[claude]`. Most
entries were read at abstract level only `[docs]`.

Cards: [sources/](sources/) (format in [sources/README.md](sources/README.md)).
Team sources: [sources/team-list.md](sources/team-list.md).

---

## 10. Next steps

1. Finish Spatial-RAG: dataset and ablation sections, complete the card.
2. Extend the question-type list (section 6, "to add").
3. Start L1 from the team's own chunk-size sweep report.
4. Optional, 20 min: look at a sample of current eval questions to understand
   what the team's numbers measure.
5. Fill datasets.md as L4 progresses.

---

## 11. Changelog

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
