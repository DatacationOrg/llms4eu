# Language anchors: what does retrieval rest on when the question changes language?

## In short (2026-10-07)

**Insights**

1. **BM25 locks onto the query's language.** A translated hard question is almost never found (0.7%), and 99% of
   its top-10 pages are in the query's language, which is 3% of the corpus. *Evidence: step 2, 300 questions.*
2. **The shared names and numbers would be enough, if they were alone.** The same questions reduced to the tokens
   both language versions share are found 43% of the time. The rest of the translated question drowns them.
   *Evidence: step 2. Caveat: uses knowledge a real system lacks; step 2b tests a realistic version.*
3. **Dense knows the neighbourhood, not the house.** On hard questions it returns the right kind of place in the
   right region (same category 76%, same country 88%) but rarely the right one: it loses the distinguishing
   details. *Evidence: 30-question pilot; dense 33% vs BM25 83% hit@10.*
4. So **"BM25 works well" holds only within one language**, and across languages only for named places. Hybrid
   inherits the language locking through its BM25 side.

**What it suggests** (not yet tested): for cross-lingual questions, feed BM25 the language-agnostic part (names,
numbers), or translate the question into the language of the region it is about (location resolution); give BM25
more weight than the default 30% for detail-rich questions; try a reranker for dense.

**Next:** step 2b, realistic anchors (CPU, minutes). **Parked:** see the table below.

## Decisions

- **Ambiguity is not an error.** Real tourists rarely ask questions with a unique answer (there are many castles).
  Report the gold rank; treat ambiguity as something the system must handle (asking back, filtering, a good set).
- **Location resolution as a lens** (idea): how precisely can a question's area be resolved (country, province,
  region, place)? The pipeline's geo score depends on it, and it could restore what BM25 loses across languages.
- **Anchors = tokens shared by a question and its translation**, not LLM-extracted: cheap, reproducible,
  language-agnostic by construction.
- **BM25 before dense**, because dense fails even in the same language on hard questions; **easy and hard apart**,
  because names survive translation.

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

- **Questions:** `question` is in the page's language; `question_x` is Ling's translation into one other EU language,
  checked by two LLM judges (`x_ok`).
- **Stored BM25** (`rank_o[3]`, `rank_x[3]`, step 1): by the dataset creator, `bm25s` over **whole pages** (title +
  text), one index for all languages, lowercased `\w+`, no stemming. Gold-page rank only.
- **Pipeline BM25** (`src/retrieval/retrievers/sparse.py`, step 2): over **512-token chunks**, raw chunk text
  (casefolded, no stemming); keeps top-k lists.
- **Dense** (Qwen3-Embedding-0.6B): same chunks, embedded as title + headings + text, Qwen's own query instruction.
- **Anchors:** casefolded `\w+` tokens in both versions, 3+ characters or containing a digit (drops shared function
  words like "de"); kind `number` (has a digit), `name` (capitalised inside the original question), `other`.

**Caveat: paraphrasing within the same language.** The hard-question generator had to paraphrase and avoid the
page's 25 most distinctive words, which sometimes bent facts. Part of the same-language difficulty of hard questions
is a generation artefact. It does not stop BM25: 62% of hard questions are found without any anchor.

### Dense failure analysis (30 pilot questions)

In the 20 failures, dense's top-10 pages share the gold page's category 76%, country 88%, language 88%: other
Spanish castles for a Spanish castle, other lakes in Kouvola for a lake in Kouvola. It misses "1451",
"Port-Vendres", "450 m wide", which BM25 matches (gold at rank 1 in 5 of the 6 inspected). Not a setup bug (Qwen's
own instruction; 30/30 easy questions found). Consequences: the default hybrid (0.7 dense) scored below BM25 alone;
a cross-encoder reranker could use the lost details; some apparent "query-language preference" of dense may be
region preference.

### Step 1: anchors and stored BM25 ranks

`shared_tokens.py`; dev, answer_ok, x_ok: 9,096 easy and 4,638 hard questions; hit@10.

| | anchors per question | has a number / name | original | translated |
|---|---|---|---|---|
| easy (named) | 2.0 | 3% / 86% | 97% | 43% |
| hard (described) | 3.3 | 85% / 56% | 62% | 0.6% |

- Easy: found after translation in proportion to anchors (12% with none, 72% with 4+, 88% without bot pages);
  across scripts (Greek, Bulgarian) 6%, within Latin script 46%.
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

### Plan

1. ~~Correlational look~~ (done, step 1).
2. ~~Why translated hard questions fail~~ (done, step 2).
2b. **Realistic anchors:** a heuristic from the translated query alone (numbers + capitalised words) instead of the
    oracle intersection; compare with the full translated query and the oracle. Does a simple query-side step
    recover cross-lingual BM25?
3. **Removal experiment:** {original, translated} x {anchors kept, removed}, levels numbers / names / both, ~500
   easy + 500 hard. Most informative: hard in the original language (does BM25 still find them without names and
   numbers?) and easy translated (how much of their 35-43% is the name).
4. **Dense on the same queries,** with placeholders instead of deleted anchors, after GPU approval.
