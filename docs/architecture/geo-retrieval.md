# Geo-Aware Retrieval

How a question about a place finds pages about that place. This is the design
reference for the geolocation layer added in September 2026; the decision record
is the "Geographic Scope" section of [decisions.md](decisions.md), and the
run-time comparison of the resulting methods is described in
[../../experiments/indexing/README.md](../../experiments/indexing/README.md).

## What problem this solves

The corpus is tourism content: castles, churches, towns, regions, and the people
tied to them. A question like "which castles are near Brestanica?" or "what can
I see in Posavje?" names an area, and a plain text search has no notion of area:
it ranks a castle 200 km away above a chapel 2 km away when the castle's page
uses more of the question's words. Geo-aware retrieval gives the pipeline that
notion, without a model ever inventing a coordinate.

## The mechanism, step by step

Every geo method (`*_hybrid_geo`, `*_hybrid_rerank_geo`, the geo agents) runs the
same steps on each question. Geography re-ranks; it does not pre-filter (the
decision of 2026-09-09, see the "Geographic Scope" section of
[decisions.md](decisions.md)).

1. **Name the place.** The judge LLM reads the question and names the one place
   it is anchored to, how wide that place is (a site, a town, a region, a
   country) and, for "near X" wording, a radius. It returns nothing when the
   question has no anchor, including questions whose *answer* is a place ("Where
   was Milan Grlj born?"). It never returns coordinates.
2. **Place the name.** A gazetteer turns the name into a `GeoScope`: country code,
   NUTS-2 and NUTS-3 region codes and, for a point, coordinates with a radius. It
   tries the cheapest source first: places the corpus already holds, then the
   official NUTS region names, then Wikidata, then OpenStreetMap Nominatim. A hit
   is kept only when its name shares a word with what the model said. The scope
   is cached per normalised question, so an eval rerun or an agent retry costs
   no second call.
3. **Decide whether the scope says anything.** A point scope always does: it
   scores by distance. A region scope is applied at its most specific level that
   keeps at most `geo_max_scope_share` (90%) of the located pages; a level that
   keeps nearly all of them carries no information and is skipped. On the
   current corpus, where 85 of 89 located pages sit in one NUTS-3 region,
   "Posavska" and "Slovenija" are therefore no-ops, which is the right answer.
4. **Over-fetch, then fuse text and geography.** Hybrid retrieval runs a vector
   search and a BM25 search, fuses them and reranks (in the `_rerank_geo`
   methods) exactly as the baseline does, but asks for `limit * geo_overfetch`
   (4x) chunks and never filters. Each chunk's reranker score is min-max
   normalised over that list and recombined as `(1 - w) * text + w * s_geo`,
   with `w = 0.3`. `s_geo` is `exp(-km / decay)` (decay 50 km) to the scope's
   point, `1` for a page inside a region scope and `0` for a located page
   outside it, and `1` for a page with **no location**. Unknown footprint is
   neutral, never "outside": the 2026-09-08 run lost 15 of 72 scoped questions
   with a hard filter, every one to a gold page that had no footprint. The list
   is cut back to `limit`.
5. **Strict variant, for comparison.** `*_hybrid_rerank_geo_strict` keeps the
   first shape: the stage is restricted to chunks whose page lies inside the
   scope (a bounding box for a radius, the region code otherwise), with
   unlocated pages let through (`geo_include_null: true`), widened one level
   (radius, NUTS-3, NUTS-2, country, none) while fewer than
   `geo_min_candidates` *located in-scope* chunks come back, then nudged by
   `(1 - w) + w * exp(-km / decay)` with `w = 0.25`. It is the shape the agent's
   `pages_in_region` tool still uses, because there the user asked for containment.

The **agentic** variants wrap this stage. `*_hybrid_agentic_geo` asks the judge
whether the retrieved chunks literally contain the answer and, if not, rewrites
the query or widens the limit and runs the geo stage again, up to three attempts.
`*_hybrid_agentic_tools_geo` additionally lets the judge call two tools:
`find_pages_near(place, radius_km)` returns the located pages within a distance
of a named place, nearest first, and `pages_in_region(code)` lists the pages in a
NUTS or country code. Both return real chunk ids, so what the agent finds is
scored like everything else.

## Data model

Where a page is about lives in `page_locations` (`sql/eval.sql`), zero or more
rows per page.

| field | standard | role |
|---|---|---|
| `latitude`, `longitude` | WGS84 | the canonical fact; everything below derives from it |
| `country_code` | ISO 3166-1 alpha-2 | coarsest filter, last widening step |
| `nuts2`, `nuts3` | Eurostat NUTS 2024 | region filters; the only hierarchy consistent across EU countries |
| `iso_3166_2` | ISO 3166-2 | municipality, display only |
| `wikidata_qid` | Wikidata | stable entity key, dedup |
| `name`, `nuts3_name` | free text | labels for display and embedding text, never filter keys |
| `role` | `primary` / `mentioned` | one primary per page is denormalised into chunk metadata; mentioned rows serve the agent tools |
| `granularity` | point / municipality / region / country | how wide the place is; a region keeps no NUTS-3, a country no NUTS-2 |
| `method`, `confidence` | | which tier produced the row |

Names are never filter keys because they are ambiguous and multilingual
(Štajerska / Styria / Steiermark) and Slovenia's statistical regions are not
administrative units, so Wikidata's containment chain skips them. Codes are
derived from coordinates by point-in-polygon against the Eurostat GISCO
boundaries (`src/shared/nuts.py`, `data/geo/`, fetched by `just fetch-nuts`), so
a NUTS revision is `just locate-pages --recompute-codes`, not a re-extraction.

The primary row also reaches the vector store: chunk metadata carries
`country_code`, `nuts2`, `nuts3` (empty strings when unlocated, since Chroma has
no null) and `latitude`/`longitude` (omitted when unlocated). Collections
written with these keys record `metadata_schema: geo1`; a geo method refuses a
collection without it rather than filtering on keys that are not there. (A `v3`
chunk text with a `Location:` line was tried in the 2026-09-08 geo run and made
no measurable difference; it was removed.)

## Enrichment: how pages get located

`just locate-pages` (`src/preprocess/locations.py`) is read-only until `--apply`
and adds rows only; it never touches `page_chunks`, so the approved eval labels
are safe. Three tiers, cheapest and most reliable first:

1. **Wikidata** for Wikipedia pages: the page title resolves (through redirects)
   to an item whose class (P31) says whether it is a place at all, and whose
   coordinates (P625), NUTS code (P605) and ISO code (P300) say where. A person,
   a language or a vanished empire gets no location by construction, even when
   Wikidata gives it coordinates.
2. **Source default**: a single-site source (a castle's own website) is about that
   site, configured as one Wikidata QID per source in `src/preprocess/config.yaml`.
3. **LLM plus Nominatim** for the rest: the model names the place a page is about
   and up to five places it mentions; Nominatim resolves each; a hit is kept only
   when its name matches what the model said.

On the 176-page corpus this located 89 pages (74 by source default, 13 by
Wikidata, 2 by the model) and stored 375 mentioned places; the 51 biographies
and the concept pages correctly received no primary location.

`just locate-pages --rejections PATH` additionally keeps every name the
gazetteer refused and classifies each by the cheapest fix that resolves it
(`src/preprocess/rejections.py`): a type word or qualifier to strip, a
Slovenian oblique form to deinflect, a place outside the `geo_country_hint`,
a Wikidata-only (historical) place, or unresolved. The table is the acceptance
test for the gazetteer work in [geo-improvement-plan.md](geo-improvement-plan.md).

## Where the pieces live

| concern | module |
|---|---|
| NUTS boundaries, point to code, name to code | `src/shared/nuts.py` |
| Nominatim, haversine, distance boost | `src/shared/geocode.py` |
| Wikidata client and place classes | `src/shared/wikidata.py` |
| the scope type, widening, Chroma filter | `src/shared/geo_scope.py` |
| question to scope (LLM plus gazetteer, cache) | `src/shared/geo_resolver.py` |
| the geo retriever (soft: over-fetch and fuse; strict: filter, widen, boost) | `src/retrieval/retrievers/geo.py` |
| `where` filter and geo metadata | `src/vector_store/chunks.py`, `src/retrieval/retrievers/vector_chunks.py` |
| BM25 page restriction | `src/retrieval/retrievers/sparse.py` |
| geo tools for the agent | `src/retrieval/retrievers/page_tools.py`, `agentic_tools.py` |
| method names and the resolver factory | `src/retrieval/methods.py` |
| enrichment | `src/preprocess/locations.py` |
| storage helpers | `src/db/pages.py` (`page_locations`, `page_ids_in_scope`, `pages_near`) |

Settings are the `geo_*` keys in `src/retrieval/config.yaml`: resolver
provider, default radius, the soft path's weight, decay, over-fetch factor and
selectivity share, the strict path's minimum candidates, boost weight and
null policy, and the Nominatim country hint.

## Running and measuring it

```bash
just fetch-nuts                          # once: Eurostat boundaries into data/geo/
just locate-pages                        # report; --apply writes page_locations
just eval-index qwen v1 base             # rebuild collections with geo metadata
just full-comparison-dry-run geo,agents-geo base
just full-comparison geo,agents-geo base geo-run
```

The comparison report adds a `Geo scope` table per method: how many questions
resolved to a place, how many were widened (strict) or had a level skipped by
the selectivity gate (soft), and which level produced the final candidates.
The measured comparison of the two shapes is
[reports/geo/geo-soft-vs-strict-2026-09-09.md](../reports/geo/geo-soft-vs-strict-2026-09-09.md). Read it first. The eval questions were generated from
chunks, so most have no geographic anchor and the geo methods will match the
baseline on the full set; the lift, if any, is in the questions that resolved,
and the per-question sidecar lets you slice to exactly those.

## Standards and literature this follows

- Spatial-RAG (Yu et al., arXiv 2502.18470, 2025): sparse spatial filtering plus
  dense semantic matching, then joint ranking. The shape of steps 3 to 5.
- Toponym resolution with lightweight LLMs and geo-knowledge (Hu et al., IJGIS
  2024): the model names, the gazetteer places and verifies. The shape of steps
  1 and 2 and of enrichment tier 3.
- Eurostat NUTS 2024 and LAU (GISCO), ISO 3166, GeoJSON (RFC 7946), schema.org
  `Place` / `GeoCoordinates` for the data model; Wikidata as the entity anchor.
- Geo stays in payload metadata rather than a bespoke scoring pass so the planned
  Chroma to Qdrant move keeps it: Qdrant has native radius filters, Chroma only a
  bounding box on stored floats.

## Known limits

- Nominatim is an external call at 1 request per second; at query time only a
  cache miss reaches it. Enrichment of a new source is minutes, not seconds.
- The corpus is Slovenia only, so `geo_country_hint: si` biases ambiguous names.
  Clear it when a second country is added.
- The legacy `places` collection behind `src.rag.agent_search` carries no
  location payload; `WiderGeoFilter` widens a real scope there, but the places
  search itself does not filter. The page-chunk stack is where the filter applies.
- Scraping keeps only extracted Markdown, so any schema.org JSON-LD coordinates on
  the original HTML are lost. Capturing them at fetch time would be a free,
  authoritative fourth enrichment tier.
