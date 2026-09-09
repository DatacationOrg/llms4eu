# Why Geo-Aware Retrieval, and Why This Shape

The argument behind [geo-retrieval.md](geo-retrieval.md) and the
[improvement plan](geo-improvement-plan.md), written 2026-09-09 for readers
who want the reasoning rather than the mechanism. Numbers are from the
2026-09-08 run and the 2026-09-09 rejection audit unless a source is named.

## 1. The claim

A tourism corpus is about places, and a share of the questions asked of it
are anchored to a place: what is near Brestanica, what to see in Posavje, who
built the castle above the town. Text similarity alone cannot tell "the castle
in Sevnica" from "the castle in Brestanica" when both pages use the same words,
and it cannot rank a page about the right village above a better-written page
about the wrong one. So retrieval should know where each page is about and
where each question points. The claim is narrower than "add a map": location
should enter as one more *score*, never as a filter that decides what may be
searched, and it should be anchored in open standards so it survives a change
of vector store, of corpus, and of country.

## 2. What the corpus looks like

The corpus is 176 pages from four Slovenian sources. Enrichment located 89 of
them with a primary point (74 from a per-source default, 13 from Wikidata, 2
from an LLM plus Nominatim) and attached 375 further places that pages mention.
The other 87 pages are biographies, concepts, national registers and clubs: not
unlocatable, but unlocated by design, because a single point cannot say "a
club in Brestanica" or "about all of Slovenia".

Two facts about the located half decide everything downstream. First, the
footprints are concentrated: 80 of the 89 points lie within 2 km of Rajhenburg
Castle, 86 within 25 km, and 85 sit in one NUTS-3 region (Posavska). There are
fifteen distinct coordinate pairs in total. Second, the spread that does exist
sits in the mentioned places: 204 of the 375 mentioned rows lie more than 25 km
away (Celje, Ljubljana, Trst, Gradec, the biographies' towns). The primary
layer discriminates almost nothing today; the mentioned layer is where a
spatial question would find its signal.

## 3. What the first run showed

The first design was the textbook one: resolve the question to a scope, filter
the search to pages inside it, widen the scope when too little comes back, then
nudge by distance. On 500 questions the resolver scoped 72. On those 72 the
filtered method fell from 0.917 hit@5 to 0.722: one win, fifteen losses.

All fifteen losses had the same cause. The gold page had no location, the
filter removed it before search, and widening never fired because five or more
located pages always supplied enough chunks. Country and NUTS scopes were the
worst, because every located page passes them: such a filter removes only the
unlocated pages and discriminates nothing. Spatial-RAG measured the same shape
on a New York tourism set, where its hard spatial stage cut answer delivery
from 98.9% to 86.1% [1]. With half the corpus unlocated, a hard filter is a
coverage lottery, and the house always wins.

A second finding came from auditing the enrichment. Of 151 distinct place names
the model produced that Nominatim refused, 124 resolve with no new model: 14 the
corpus already knew, 17 carried a type word ("Grad Sevnica"), 42 resolve once
the Slovenian country hint is lifted, and 51 are Slovenian exonyms or historical
names a Slovenian-language Wikidata search finds (Dunaj, Trst, Celeja). Zero
were inflected forms. Worse, the hint had silently placed Gradec, Dunaj, Rim and
Trst on Slovenian hamlets that carry the same name. The country *filter* on the
geocoder, not Slovenian morphology, was the cause of the misses.

## 4. What the literature says

Four results from 2023 to 2026 carry the design.

**Filter versus score.** Spatial-RAG [1] is the closest published design: an
LLM parses the spatial constraint, a sparse spatial stage and a dense semantic
stage each score candidates, and the final ranking trades the two off. Its gain
(+19.9% P@1) came from 9,470 fully located points spread across a city; its
loss came from letting the spatial stage filter. The lesson that transfers to a
half-located corpus is the second one.

**How to combine two scores.** Bruch, Gai and Ingber [2] show that a convex
combination of *normalised* scores beats reciprocal rank fusion in and out of
domain, is robust to the normalisation choice, and can be tuned from a handful
of labelled queries. Rank fusion discards the magnitude of a distance; a
weighted sum keeps it.

**How to locate text.** Hu, Kersten, Klan and Farzana [3] and Hu, Kersten and
Klan [4] resolve place names by having a small open model write an unambiguous
reference string and matching it against a gazetteer candidate list with a
country *prior*. That lifts accuracy from a bare-geocoder baseline of 0.21 to
0.77 (across seven datasets) to 0.91 to 0.93 with 7B models, with historical
corpora the weakest case at 0.75 to 0.81. Our tier-3 acceptance rate of about
72% is that baseline; our Celeja is their weak case; and the rejection audit
confirmed their mechanism (prior, not filter) as the fix. This is the approved
reference study, scored highest of five candidates on corpus shape, coverage,
stack match, evidence quality and fit to the bottleneck.

**Standards for the codes.** Coordinates are the canonical fact and region codes
are derived by point-in-polygon against Eurostat's GISCO NUTS 2024 boundaries,
because NUTS is the one region hierarchy that is consistent across EU countries
and Slovenia's statistical regions are not administrative units, so Wikidata's
administrative chain skips them. Names are labels, never keys: Štajerska,
Styria and Steiermark are one place.

## 5. The design that follows

1. **Geography re-ranks; it does not pre-filter.** The stage retrieves four
   times the needed candidates unfiltered, reranks as the baseline does, then
   recombines each score as `(1 - w) * text + w * s_geo`, both in [0, 1].
2. **Unknown footprint is neutral.** A page with no location gets `s_geo = 1`:
   never helped, never removed. This alone recovers the fifteen losses.
3. **A scope is applied only when it discriminates.** A region level that keeps
   more than 90% of the located pages is skipped. On this corpus "Slovenija" and
   "Posavska" are therefore no-ops, which is the correct answer.
4. **The model names, the gazetteer places.** The LLM never emits coordinates;
   it names a place and its width, and a deterministic chain (corpus names, NUTS
   names, Wikidata, Nominatim) resolves it, verified by name, cached in SQLite.
5. **Coordinates canonical, codes derived, metadata payload-only.** A NUTS
   revision is a recomputation, not a re-extraction; the Chroma-to-Qdrant move
   keeps working because nothing bespoke is scored inside the store.

The hard filter survives as an explicit `_strict` method for measurement, and
inside the agent's `pages_in_region` tool, where the user asked for containment.

## 6. What would confirm or refute it

The gate for the ranking change is on the 72 scoped questions: no losses against
the text baseline (hit@5 at or above 0.917), and no regression on the 37
near/okolica questions. Gains are *not* expected on this corpus, and the
literature says why: with fifteen coordinate pairs and every miss at 0 km from
its scope, no distance score can reorder anything. The gain the design is built
for appears when the footprint layer widens (the 375 mentioned places, the
gazetteer fixes of §3) and when the eval set contains real spatial questions,
which the chunk-generated set almost lacks (37 of 3,476 use near-wording). So
the honest sequence is: first stop geography from hurting (measurable now), then
give it something to discriminate on (enrichment), then write the probe set
that can see it. If the soft method loses scoped questions the baseline gets,
principle 1 is wrong for this stack. If it matches the baseline but the probe
set later shows no lift, the corpus is simply too small in space for geography
to matter, and the layer stays as EU-scale infrastructure rather than a
Brestanica feature.

## Sources

1. Yu, Bao, Ning, Peng, Mai, Zhao. Spatial-RAG: Spatial Retrieval Augmented
   Generation for Real-World Geospatial Reasoning Questions. arXiv 2502.18470,
   2025. https://arxiv.org/abs/2502.18470
2. Bruch, Gai, Ingber. An Analysis of Fusion Functions for Hybrid Retrieval.
   ACM TOIS, 2023. https://dl.acm.org/doi/10.1145/3596512
3. Hu, Kersten, Klan, Farzana. Toponym resolution leveraging lightweight and
   open-source LLMs and geo-knowledge. IJGIS, 2024.
   https://doi.org/10.1080/13658816.2024.2405182
4. Hu, Kersten, Klan. Scalable Toponym Resolution with LLMs: Accuracy and Speed
   Optimizations. GeoExT at ECIR 2025, CEUR vol. 3969.
   https://ceur-ws.org/Vol-3969/paper6.pdf

The full reading list with per-paper notes is
[geo-literature-2023-2026.md](geo-literature-2023-2026.md).
