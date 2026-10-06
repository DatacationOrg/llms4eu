# Implementation plan (draft v0, 2026-10-06)

Status: **draft, pending the team's selection** of 2–3 items and answers to
T9 (may the chatbot use the model's own knowledge?) and T10 (target
documentation bias or crowding?). Evidence behind each item:
[consolidation-2026-10-06.md](consolidation-2026-10-06.md) and the source cards.

## Principles

1. **Measure before fixing.** A technique is tested only once a measurement
   shows the problem it targets.
2. **Proxy results check instruments, not settings.** Nothing is tuned on the
   proxy corpus; the aim is working tools by December.
3. **Every item has a decision criterion,** set with the team before running.
   Thresholds below are placeholders (X, Y).
4. **Guardrail for every technique:** no drop in hit@5 / MRR on the existing
   factual questions beyond run-to-run noise.

Legend: **Proxy** = can run on the current corpus; **Dec** = needs the
December dataset; effort S (< half a day), M (1–2 days), L (more).

---

## Phase 0: Prerequisites

Unlock most later items; cheap; no decisions needed.

| id | what | why | effort | when | unlocks |
|---|---|---|---|---|---|
| P1 | Wikidata place ID in chunk metadata | per-place counting, caps, aggregation | S | Proxy | M4, T3, T4, T7 |
| P2 | One shared question set scored against every chunking (`just shared-overview`) | fair chunk-size comparison (team's own fix; Qu et al.) | S | Proxy | chunking decisions |
| P3 | Unlocated-page audit: sample unlocated pages, classify "not a place" vs "too obscure to locate" | the geo score may favour places whose location is known | S | Proxy | interpreting M4, H6 |
| P4 | Cross-border probe: 10–30 hand-written questions, Slovenian queries, gold on Croatian, German and Hungarian pages | test language bias across borders; neighbouring languages may show little gap | S | Proxy | M2, T1, T2 |
| P5 | Spatial set questions from location metadata ("castles within 30 km of X"; gold derived automatically) | set-valued gold without manual labels | M | Proxy | M4, T3–T5 |

---

## Phase 1: Measurements

Each answers "is the problem there, and where does it enter?"

**M1. Documentation skew in the corpus**
- What: distribution of pages and chunks per located place.
- Why: Pepe's concern; chunk count is how skew enters retrieval.
- How: one query over the pages DB. Effort S. Proxy.
- Decision: if the top 10% of places hold more than X% of chunks, the skew is
  material → T3/T4 worth testing.

**M2. Language mix before vs after reranking**
- What: language share in hybrid top-30 vs reranked top-10, on cross-lingual
  questions and the P4 probe.
- Why: rerankers favour the query language and English (All Languages Matter).
- How: log languages at both stages. Effort S. Proxy. Depends on P4.
- Decision: if non-query-language share drops by more than X points at
  reranking, the reranker adds bias → T1 and a reranker review.

**M3. Closed-book test**
- What: ask the local model (gpt-oss:20b) questions about the localities without
  retrieval.
- Why: decides whether the model's own knowledge matters (PopQA).
- How: 30–50 factual questions from the existing set. Effort S. Proxy.
- Decision: if accuracy is near zero, adaptive retrieval (D4) is dropped and the
  risk to watch is the model substituting famous places.

**M4. Concentration with and without the geo score**
- What: concentration and effective diversity of places in the top-k across many
  questions, for the baseline and the soft geo method.
- Why: the geo score may reduce concentration (H6) or create it (known locations,
  hubs); prior work shows both.
- How: count place IDs in top-k over P5 and general tourism questions. Effort S–M.
  Proxy. Depends on P1, P3, P5.
- Decision: if the geo method raises concentration on well-documented places,
  revisit "unknown = neutral" (decision 6).

**M5. Breakdown of existing results**
- What: hit@5 split by language match (question vs gold page) and by
  documentation level of the gold page.
- Why: shows where the system already fails, without new runs.
- How: re-slice stored per-question results. Effort S. Proxy (Brestanica labels only).
- Decision: a gap of more than X points between slices → the corresponding
  technique is worth testing.

**M6. Language preference (MLRS)**, optional
- What: rank shift of documents when translated into one shared language.
- Why: label-free measure of language bias (Park & Lee).
- Effort M (translation model). Proxy.

---

## Phase 2: Cheap techniques

Each tested against the measurement named; all training-free and local.

| id | technique | targets | measured by | effort | when | depends on | adopt if |
|---|---|---|---|---|---|---|---|
| T1 | Per-language balanced retrieval: top results per language, then merge (Amiraz) | language bias | M2, P4, M5 | S | Proxy | M2 or M5 shows a gap | cross-border recall up by X, guardrail holds |
| T2 | Geo-aware language quota: balance only languages spoken near the anchor (own H2) | language bias without noise | P4 | S | Proxy | T1 | beats T1 on P4, guardrail holds |
| T3 | Per-place cap: at most n chunks per place in the top-k | documentation skew | M1, M4 | S | Proxy | P1, M1 | concentration down by X, guardrail holds |
| T4 | Place-level scoring: score each place by the average of its best n passages (EQR) | documentation skew | M4 | M | Proxy | P1 | as T3; compare with T3 |
| T5 | Extra small-weight term next to the geo score: documentation volume, popularity or locality | skew or crowding (T10 decides) | M4, M5 | S | Proxy | T10, P1 | concentration down, guardrail holds |
| T6 | Query rewriting for vague questions (EQR prompt, local model) | recommendation questions | P5 or a small generated set | S–M | Proxy (small set) | a set of vague questions | recall on vague questions up by X |
| T7 | Diversity re-ranking (MMR / xQuAD over places) | skew | M4 | M | Proxy | P1; only if T3/T4 fall short | as T3; mixed evidence, watch answer relevance |

Note: re-ranking cannot surface places outside the over-fetched pool; check the
pool size (currently 4×) when T3–T7 show little effect.

---

## Phase 3: December-dependent

| id | what | why | depends on |
|---|---|---|---|
| D1 | Evaluation for recommendation questions: pairwise, per-aspect LLM judging (relevance, diversity, popularity mix), calibrated on a small human-labelled sample | no single gold answer; judges matched experts only 17–54% (Banerjee 2026) | December question types |
| D2 | Calibrate the team's equivalence judge against a human-checked sample | same finding applies to our own judge | a human sample (could start on proxy) |
| D3 | Routing by question type: different retrieval settings for factual vs recommendation questions | intent routing (Tandon); dense may matter more for vague queries (EQR) | December question types |
| D4 | Adaptive retrieval: retrieve only when the model is unlikely to know | PopQA | M3 and T9 |
| D5 | Position the December dataset against TourismQA, SynthTRIPs, TravelDest | goal 3 of the review | December dataset |
| D6 | Reranker trained against language bias (LAURA) | All Languages Matter | answer-labelled data; M2 shows reranker bias |

---

## Component × problem matrix

| component | language / cross-border | documentation skew / selection | recommendation questions | model knowledge |
|---|---|---|---|---|
| Data / metadata | P4 cross-border probe | P1 place ID; P3 audit; M1 | P5 spatial set; small generated set | |
| Retrieval | T1, T2 | T3, T4 | T6; D3 | |
| Reranking | M2; D6 | T5, T7 | | |
| Generation | language preference in citations | | | M3; D4 |
| Evaluation | M2, M5, M6 | M4, M5 | D1, D2; D5 | M3 (three-point) |

---

## For the team

1. Which 2–3 items are worth the time? Suggested start: P1, M1, M2 (with P4),
   M3: all small, and together they show whether the two main concerns are
   present at all.
2. T9: may the chatbot use the model's own knowledge? (decides D4)
3. T10: target the bias toward well-documented places, or crowding? (decides T5)
4. T6: has geo retrieval been rerun on the expanded proxy corpus? (baseline for M4)
5. Who sets the thresholds (X, Y) for the decision criteria?