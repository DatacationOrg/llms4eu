# Language anchors: what does retrieval rest on when the question changes language?

*This file: one experiment, the source of truth for its findings and decisions. Cross-experiment summary and ratings: [overview](../README.md).*

## In short (2026-10-07)

**Insights**

1. **BM25 locks onto the query's language.** A translated hard question is almost never found (0.7%), and 99% of its top-10 pages are in the query's language, which is 3% of the corpus. *Evidence: step 2, 300 questions.*
2. **The shared names and numbers would be enough, if they were alone, and a trivial heuristic finds them.** Keeping only the numbers and capitalised words of the translated question lifts hard questions from 0.7% to **42%** and easy ones from 38% to **89%**, as good as the oracle (the tokens shared with the original: 43% / 82%). The rest of the translated question drowns them. Weaker for German queries (every noun is capitalised: 35% against the oracle's 59%). *Evidence: steps 2 and 2b, 300 + 300 questions.*
3. **In hard questions, the specificity is in the combination.** Names alone (mostly regions: "Kiruna", "Västra Götaland") find 5%: right language, wrong page. Numbers alone find 27%. Together 46%: the name narrows the region, the numbers pick the page. In easy questions the name alone does it (95%). *Evidence: step 2, questions with that kind of anchor.*
4. **Dense knows the neighbourhood, not the house.** On hard questions it returns the right kind of place in the right region (same category 76%, same country 88%) but rarely the right one: it loses the distinguishing details. *Evidence: 30-question pilot; dense 33% vs BM25 83% hit@10.*
5. So **"BM25 works well" holds only within one language**, across languages only for named places, and on hard questions mainly through the numbers the generator added (insight 6). Hybrid inherits the language locking through its BM25 side.

6. **BM25's success on hard questions rests mostly on the numbers the generator added.** In the page's own language, removing the numbers drops hard questions from 54% to **14%**, removing names 55% to 42%, removing both 53% to **12%** (paired, same questions). Easy questions rest on the name: 96% to 21% without it. Real users rarely give exact numbers, so BM25's scores on this test set are inflated; but dense does not overtake it without them (insight 7). *Evidence: step 3, 300 + 300 questions; inflected names ("Ljusnans") are not removed, so the name effect is underestimated.*

7. **Removing the exact details does not make dense better than BM25** (confirmed on a larger, stratified sample, see *Robustness*). Outside Swedish, without numbers and names BM25 still finds 44% of hard questions, dense 6% (full: 65% vs 13%). Swedish hard questions (77% of this sample, mostly lake-register questions) are found only through their numbers: without them BM25 drops to 4%, dense stays near 0. So dense (Qwen3-0.6B, 512-token chunks) is weak on described places in general, not only because the generator added numbers. *Evidence: step 4, 300 hard questions (70 non-Swedish: a signal).*
8. **Dense has no language locking:** easy questions are found 97% of the time after translation (BM25 38%). Dense carries named places across languages, BM25 carries detailed descriptions within a language: the two sides of hybrid complement each other. *Evidence: step 4, 300 easy questions.*

9. **Fusing the full question with its anchors is a cheap, deployable fix.** BM25 on the full question and on its numbers + capitalised words, rankings combined with reciprocal rank fusion: translated hard questions 2% -> **25%**, translated easy 45% -> **77%**, and same-language questions do not suffer but gain a little (hard 71% -> 74%, easy 92% -> 96%). No model, no tuning (standard RRF k = 60). Fused is a little below anchors alone on translated questions (31%, 82%): the full question's language-locked list still pulls. Weak for German queries (hard 2%, easy 69%). *Evidence: step 2c, stratified 929 hard + 1,678 easy, 95% intervals.*

**Robustness (2026-10-08):** steps 2, 2b, 3 and 4 rerun on a stratified sample (up to 100 per language: 929 hard, 1,678 easy; 95% intervals). The direction of every insight holds. Sizes change: outside Swedish, numbers matter less than the first, Swedish-heavy sample suggested (BM25 70% -> 56% without numbers and names, not -> 12%), and the realistic anchor heuristic recovers 31% of translated hard questions (oracle 33%), but only 10% when the translation is German. Chunk size: at 256 tokens dense improves a little (other Latin script 22% -> 27% full, 14% -> 18% without numbers and names), BM25 hardly changes; dense stays far behind. Details in *Robustness* below.

**How far to trust each insight** (robust, general, surprise, value for the team): see the [overview](../README.md#how-much-each-finding-is-worth).

**What it suggests:** run BM25 twice, on the full question and on its numbers + capitalised words, and fuse the rankings (tested, insight 9). Open: weighting the anchor list higher for questions likely in another language than the pages, and a German exception. Also: a language-independent location filter (see [location](../location/README.md)), more BM25 weight than the default 30% for detail-rich questions, and a reranker for dense.

**Scope:** this corpus has one page per place, in the language of its country. In reality a tourist may find a page in their own language, mainly for famous places (49% of places here exist in only one Wikipedia language). So the language locking matters most for places that only exist in their local language: the obscure ones the project wants to surface. In a corpus with several language versions per place, it could favour famous places (inference). Hit rates count the gold page only; fine for steps 1-2 (questions generated to be unique), relevant from step 3 on.

**Next:** the oracle location filter is in [location](../location/README.md). **Parked:** see the table below.

## Decisions

- **Ambiguity is not an error.** Real tourists rarely ask questions with a unique answer (there are many castles). Report the gold rank; treat ambiguity as something the system must handle (asking back, filtering, a good set).
- **Location** (resolving the area a question is about, as a language-independent filter) has its own experiment: [location](../location/README.md).
- **Anchors = tokens shared by a question and its translation**, not LLM-extracted: cheap, reproducible, language-agnostic by construction.
- **BM25 before dense**, because dense fails even in the same language on hard questions; **easy and hard apart**, because names survive translation.

## Parked, and why

| task | why parked |
|---|---|
| dense/hybrid language study (high/low-resource query x page language) | dense fails even in the same language; understand it first. Design ready: paired original vs translated question per tier pair, top-10 language shares against corpus shares |
| reranker test on the 30 pilot questions | GPU, ask first; the logical next dense-side test |
| MLRS (Park & Lee 2025) | needs translating retrieved documents: heavy; plan only |
| cross-border geo questions | ~47 in dev: too few |
| English-preference measure from the literature | only 265 English pages (0.2%) |
| clue ablation | LLM throughput (free API rate-limited, GPU route unmeasured); see [its README](../clue_ablation/README.md) |

---

## Details

### Inputs: which question, which retriever, over what text

- **Questions:** `question` is in the page's language; `question_x` is Ling's translation into one other EU language, checked by two LLM judges (`x_ok`).
- **Stored BM25** (`rank_o[3]`, `rank_x[3]`, step 1): by the dataset creator, `bm25s` over **whole pages** (title + text), one index for all languages, lowercased `\w+`, no stemming. Gold-page rank only.
- **Pipeline BM25** (`src/retrieval/retrievers/sparse.py`, step 2): over **512-token chunks**, raw chunk text (casefolded, no stemming); keeps top-k lists. Its in-memory index build (~4 GB burst) kept getting killed by systemd-oomd under memory pressure on the shared server.
- **DuckDB BM25** (`fts.py`, `--engine duckdb`, from step 2b): the same chunks, full-text index on disk (out/bm25-512.duckdb), ~0.7 GB memory, ~0.1 s per query. Reproduces the pipeline's numbers within 1-2 points (original question 54% vs 53%, translated 0.7% vs 0.7%, anchors 43% vs 43%); compare within one engine.
- **Dense** (Qwen3-Embedding-0.6B): same chunks, embedded as title + headings + text, Qwen's own query instruction.
- **Anchors:** casefolded `\w+` tokens in both versions, 3+ characters or containing a digit (drops shared function words like "de"); kind `number` (has a digit), `name` (capitalised inside the original question), `other`.

**Caveat: BM25 scores words independently.** The pipeline's tokenizer (`[^\W_]+`) splits "0,014" into "0" and "014", "14,68 km²" into "14", "68", "km²", and "2,202" into "2", "202". A value is not bound to its unit, so "93" on a page about anything counts the same. Specificity comes only from several fairly rare tokens co-occurring in one chunk. Anchor counts in step 1 are inflated by such fragments ("0" counts as an anchor but carries almost no weight). Design input: normalise number formats and keep value + unit together.

**Caveat: paraphrasing within the same language.** The hard-question generator had to paraphrase and avoid the page's 25 most distinctive words, which sometimes bent facts. Part of the same-language difficulty of hard questions is a generation artefact. It does not stop BM25: 62% of hard questions are found without any anchor.

### Dense failure analysis (30 pilot questions)

In the 20 failures, dense's top-10 pages share the gold page's category 76%, country 88%, language 88%: other Spanish castles for a Spanish castle, other lakes in Kouvola for a lake in Kouvola. It misses "1451", "Port-Vendres", "450 m wide", which BM25 matches (gold at rank 1 in 5 of the 6 inspected). Not a setup bug (Qwen's own instruction; 30/30 easy questions found). Consequences: the default hybrid (0.7 dense) scored below BM25 alone; a cross-encoder reranker could use the lost details; some apparent "query-language preference" of dense may be region preference.

### Step 1: anchors and stored BM25 ranks

`shared_tokens.py`; dev, answer_ok, x_ok: 9,096 easy and 4,638 hard questions; hit@10.

| | anchors per question | has a number / name | original | translated |
|---|---|---|---|---|
| easy (named) | 2.0 | 3% / 86% | 97% | 43% |
| hard (described) | 3.3 | 85% / 56% | 62% | 0.6% |

- Easy: found after translation in proportion to anchors (12% with none, 72% with 4+, 88% without bot pages); across scripts (Greek, Bulgarian) 6%, within Latin script 46%.
- Hard: not found after translation even with anchors (1.2% with 4+).
- Same-language success of hard questions does not depend on anchors (62% with none, 70% with 4+).
- Number formatting breaks anchors: "2202 m" (de) vs "2,202 m" (en).

### Step 2: why translated hard questions fail

`translated_topk.py`; pipeline BM25, 512-token chunks; 300 easy + 300 hard dev questions with `x_ok`.

| query | hit@10 hard | hit@10 easy | top-10 in the query's language (hard / easy) | corpus share of that language |
|---|---|---|---|---|
| original question | 53% | 93% | 100% / 100% (= page language) | - |
| translated question | 0.7% | 35% | 99% / 89% | 3% |
| anchors only | 43% (46% if any) | 83% (94% if any) | - | - |

By kind of anchor, only questions that have that kind (n hard / easy):

| query | hard: hit@10 | hard: top-100 | easy: hit@10 |
|---|---|---|---|
| anchors only (278 / 265) | 46% | 64% | 94% |
| numbers only (259 / 10) | 27% | 48% | (10 questions) |
| names only (156 / 261) | 5% | 20% | 95% |

Hard names-only queries stay in the page's language (97% of the top 10) but miss the page: the names are regions. Numbers-only queries spread over languages (37% in the page's language): numbers are language-agnostic but, alone, not tied to a place. One run was killed by memory pressure on the shared server (index build peaks at ~4 GB); a rerun completed.

### Step 2b: realistic anchors

`translated_topk.py --engine duckdb`; realistic = numbers + capitalised words (not the first word) of the translated question alone; same 300 + 300 questions as step 2. Note the engine: step 2 used the pipeline's BM25, steps 2b and 3 the DuckDB BM25, which differ by 1-2 points (original question 53% vs 54%); compare within a step, not across engines.

| query | hit@10 hard | hit@10 easy |
|---|---|---|
| original question | 54% | 94% |
| translated question | 0.7% | 38% |
| realistic anchors (translated question only) | 42% | 89% |
| oracle anchors (shared with the original) | 43% | 82% |

By translation language (hard): German 35% (oracle 59%: every noun is capitalised), Greek/Bulgarian 36% (oracle 39%), other Latin-script 43% (oracle 43%). The heuristic sometimes beats the oracle: it keeps names the intersection missed through inflection or exonyms ("Byskeälven", "Ljusnan").

### Step 3: anchors removed from the full question

`removal.py` (DuckDB BM25); anchors deleted by kind from the original and the translated question; each variant scored only on questions that lose something, next to the full question on the same questions (paired). Hits on the gold page; gold + `relevant` changes almost nothing (the relevance pool was judged for the original question only).

| hard questions, original language | n | full | without |
|---|---|---|---|
| -numbers | 259 | 54% | 14% |
| -names | 156 | 55% | 42% |
| -both | 274 | 53% | 12% |

Easy questions, original language: -names (261) 96% -> 21%. Translated questions are near 0 in every variant (hard 0.7% -> 0%; easy 41% -> 0.4% without names). Limitation: anchors are tokens shared with the translation, so inflected names ("Ljusnans" vs "Ljusnan") stay in the question.

### Step 2c: fusion

`fusion.py --per-lang 100` (DuckDB BM25, gentle); for the original and the translated question alike: BM25 on the full question and on its realistic anchors (numbers + capitalised words of that question), page rankings fused with reciprocal rank fusion (k = 60); a question without anchors keeps its full ranking. hit@10 [95% interval].

| | n | full | anchors | fused |
|---|---|---|---|---|
| hard, original | 929 | 71% [68-73] | 53% [50-56] | 74% [72-77] |
| hard, translated | 929 | 2% [1-3] | 31% [28-34] | 25% [22-28] |
| easy, original | 1,678 | 92% [91-93] | 94% [93-95] | 96% [94-96] |
| easy, translated | 1,678 | 45% [42-47] | 82% [80-84] | 77% [75-79] |

By query language, translated hard: other Latin 2% -> 27%, Greek/Bulgarian 0% -> 18%, Swedish 0% -> 29%, German 0% -> 2% (anchors alone 10%). Same language, hard: Swedish 40% -> 52%, other Latin 72% -> 75%, German 89% -> 91%.

### Robustness: stratified sample, 95% intervals

Up to 100 dev questions per kind and language with `x_ok` (`--per-lang 100`): 929 hard (701 other Latin script, 100 German, 100 Swedish, 28 Greek/Bulgarian) and 1,678 easy. DuckDB BM25 run gently (4 queries in parallel, one thread each, `nice`); dense on the GPU. Removed words replaced by "…".

Hard questions, original language, full -> without numbers and names, hit@10 [95% interval] (n = questions that lose something):

| page language | BM25 | dense |
|---|---|---|
| other Latin (540) | 70% -> 56% [52-60] | 21% -> 14% [12-18] |
| German (90) | 88% -> 72% [62-80] | 19% -> 9% [5-17] |
| Greek / Bulgarian (17) | 65% -> 47% [26-69] | 18% -> 18% [6-41] |
| Swedish (94) | 41% -> 1% [0-6] | 0% -> 0% [0-4] |

Without numbers only, other Latin: BM25 68% -> 54%, dense 20% -> 16%; without names only: BM25 80% -> 75%, dense 29% -> 19%.

Translated questions, hit@10: hard BM25 2% (dense 7%); realistic anchors 31% [overall], by translation language: other Latin 33%, Greek/Bulgarian 22%, German 10% (oracle anchors 33%, 26%, 32%). Easy: translated 45%, realistic anchors 82%, oracle anchors 73%. Hard questions by page language with the realistic anchors: other Latin 29% [26-32], German 41% [32-51], Greek/Bulgarian 11% [4-27], Swedish 42% [33-52].

Chunk size, same sample at 256-token chunks (`CHUNK_SIZE=256`), hard questions, other Latin script, full -> without numbers and names: dense 27% [23-30] -> 18% [15-21] (512: 22% -> 14%), BM25 69% -> 54% (512: 72% -> 56%). German: dense 29% -> 17% (512: 19% -> 9%). Smaller chunks help dense by about 5 points (borderline: the intervals touch), consistent with details being diluted in larger chunks as part of the explanation; the 30-question pilot ("256 = 512") was too small to see it.

### Plan

1. ~~Correlational look~~ (done, step 1).
2. ~~Why translated hard questions fail~~ (done, step 2).
2b. ~~Realistic anchors~~ (done).
2c. ~~Fusion~~ (done; see *Step 2c* below).
3. ~~**Removal experiment**~~ (done, BM25; dense is step 4) (the opposite of step 2, which kept only the anchors): remove the anchors from the full question, e.g. "de plaats ontstaan in 1932 en is 30 km²" -> "de plaats ontstaan en is km²". {original, translated} x {anchors kept, removed}, levels numbers / names / both, ~500 easy + 500 hard. Most informative: hard in the original language (does BM25 still find them without names and numbers?) and easy translated (how much of their 35-43% is the name). Questions may now fit several places: report the gold rank, count `relevant` pages as hits where judged, and accept the ambiguity (see decisions).
4. ~~Dense on the same queries~~ (done, step 4).

### Step 4: dense vs BM25 without anchors

`removal.py --engine dense --placeholder "…"` and `--engine duckdb --placeholder "…"`; removed words become "…" so the sentence stays intact for dense; same 300 + 300 questions; paired (full on the same questions).

| | BM25 full -> without numbers & names | dense full -> without numbers & names |
|---|---|---|
| hard, non-Swedish (70; 54 lose something) | 65% -> 44% | 13% -> 6% |
| hard, Swedish (230; 220) | 51% -> 4% | 1% -> 0.5% |
| easy, original (300) | 94% | 99.7% |
| easy, translated (300) | 38% | 97% |
| easy, without names (261), original / translated | 21% / 0.4% | 17% / 5% |

The sample is not restricted to `balanced` pages, so 77% of its hard questions are Swedish; the 30-question pilot (balanced pages) had dense at 33%. BM25 with the placeholder reproduces the deletion run (step 3) within a point.
