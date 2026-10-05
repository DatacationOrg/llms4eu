# Decisions and assumptions in the current implementation

Spoke of the hub ([landscape-and-task-scoping.md](landscape-and-task-scoping.md) §4).
Each decision is evaluated for whether it will still hold for the official dataset.
Test: "this decision would be wrong if …". Decisions `[docs: decisions.md, geo-retrieval.md]`;
assumptions and verdicts `[me]` refined with Claude.

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
| 9 | Location in simple metadata, not store-specific features        | Survives Chroma → Qdrant move                     | Geo scoring stays outside the store                                   | Stable for now; at scale the 4× over-fetch costs speed                                                                   |

## Not answered by the texts

- How does the region gate behave now that it fires? (row 7)
- Does place-name extraction work in all corpus languages? (row 5)
- Do the eval questions resemble what real users will ask?
- How does the soft method's over-fetching scale to 300K documents? (row 9)
- Is "local-only inference" still a principle? A hosted model is the geo
  resolver and agent judge (see T7 in the hub).