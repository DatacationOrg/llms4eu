# Experiments: overview (2026-10-07)

*This file: across experiments, for the team: the main caveat, a rating of every finding, what is ready, priorities. Each experiment's own README is the source of truth for its findings, evidence and decisions.*

## The question behind all of them

**What does retrieval rest on for tourist questions, and where does it break?** Three axes keep coming back:

- **Detail:** which details a question carries (names, numbers, descriptions) and how many.
- **Language:** whether the question is in the page's language.
- **Location:** how precisely the area a question is about can be resolved.

Each experiment is chosen for what its outcome would change in the system: make us **aware** of a failure case, point to a **targeted fix**, or show a technique worth **doubling down** on. Experiments whose outcome is predictable, or would not change a decision, are parked.

## The main caveat on everything below

Hard questions are findable mainly through rare numbers and specific details that the generator added to make each question unique. BM25 uses exactly those; dense loses them. So **"BM25 beats dense on hard questions" may partly be an artefact of how the questions were made**: a real tourist gives few exact numbers ("a castle near Utrecht from the Middle Ages"), and there dense's strength (the right kind of place in the right region) could matter more. The findings are **conditional on overspecified questions**. **First evidence (BM25 side):** removing the numbers from hard questions drops BM25 from 54% to 14%, numbers and names together to 12% (language anchors, step 3). The dense side of that comparison (step 4, GPU) and a small set of human-written questions are the remaining checks.

## How much each finding is worth

Every finding is rated on four dimensions, so it is clear what can go to the team and with which caveat:

- **Robust:** sample size, controls, agreement across methods.
- **General:** does it hold beyond this generated dataset (real users, other corpora)?
- **Surprise:** could we have predicted it beforehand? Low surprise can still be valuable (confirming and sizing a known risk).
- **Value for the team:** what it changes: *aware* (a failure case to design around), *fix* (a targeted remedy), *double down* (a technique worth investing in), or *evaluation* (how to read results on this dataset).

| finding | robust | general | surprise | value for the team |
|---|---|---|---|---|
| **BM25 locks onto the query's language:** a translated hard question is found 0.7% of the time; 99% of its top 10 is in the query's language (3% of the corpus) | **high:** 300 questions, two BM25 engines agree | **high:** mechanical, any lexical retriever on a multilingual corpus | low for the failure, medium for the mechanism (description words drown the shared names and numbers) and its size | **high, aware:** hybrid inherits it through its BM25 side; cross-lingual questions need their own handling |
| **A trivial anchor heuristic recovers it:** numbers + capitalised words of the translated question, 0.7% -> 42% (hard), 38% -> 89% (easy), close to the oracle | **medium:** 300 + 300 questions, as a diagnostic; fusion with the full query untested | **medium:** generated translations keep names neatly; real users write exonyms and variants; German capitalises all nouns | **high:** a regex comes close to the oracle | **high if fusion holds, fix:** cheap, no model (step 2c) |
| **BM25's success on hard questions rests on the numbers the generator added:** without numbers 54% -> 14%, without numbers and names 53% -> 12% (paired) | **high:** paired, 259-274 questions | **medium:** the mechanism is general (BM25 needs rare tokens); the size comes from how questions were generated | **medium:** the size | **high, evaluation:** "BM25 beats dense" on this test set is probably inflated; do not choose a retriever on it alone |
| **Hard-question specificity sits in the combination:** names 5%, numbers 27%, both 46% | **high** | **low:** real users rarely give exact numbers | medium | **medium, evaluation:** explains why hard questions are findable at all |
| **Dataset facts:** no popularity bias; ~5% of geo questions cross a border; stored dense ranks unusable; `realistic` label lenient and Swedish-only ([audit](../docs/reports/wiki-audit/audit-2026-10-06.md)) | **high:** full dev set | n/a: about this dataset | medium: the lenient label and missing cross-border cases | **high, evaluation:** what the test set can and cannot measure (no cross-border, no recommendation questions) |
| **Hand check:** 15 of 20 hard questions natural, only 3 of 20 tourist questions; the reason is overspecification | **low:** 20 questions, one annotator | unknown | medium | **medium, evaluation:** motivated treating detail as a variable |
| **Region density by category** ([location](location/README.md)): within 10 km, castles ~1 similar page, lakes ~12, Finnish lakes ~63 | **high:** metadata, 702 pages | **high:** real geography | low: lakes cluster, castles do not | **medium-high, double down / aware:** location resolution could settle castle questions, not lake questions |
| **Dense knows the neighbourhood, not the house:** right category 76%, right country 88%, wrong page | **low:** 30 questions, one model, one instruction, no reranker | unknown | low: a known weakness of single-vector retrieval | **medium, hypothesis:** if confirmed, a reranker or more BM25 weight; step 4 first |
| **Hybrid below BM25; chunk size does not matter for dense** | **low:** 30 questions | unknown | medium (hybrid worse than one of its parts) | **low until confirmed** |
| **BM25 splits numbers and ignores units** | high | high | low: known | **low-medium, fix:** number normalisation is cheap |

**Ready for the team now** (robust and valuable): BM25's language locking and why; that BM25's lead on hard questions rests on generated numbers, so retriever comparisons on this set are biased; what the dataset cannot measure; region density by category. **With a caveat:** the anchor heuristic (pending fusion). **Not yet:** the dense explanation, hybrid weighting, chunk size.

## What it means for the system (conditional on the caveat)

**Be aware:** dense loses distinguishing details; BM25 locks onto the query's language and hybrid inherits it; a region narrows castles but not lakes; BM25 mangles numbers; the test set overstates real use.

**Fix, to test:** an anchor query fused with the full query for cross-lingual BM25; a language-independent location filter; more BM25 weight for detail-rich questions; number normalisation; a reranker for dense.

**Double down:** BM25's strength on specific details and the anchor heuristic; location resolution for sparse kinds of places; designing for ambiguity (shortlists, asking back) instead of assuming unique answers.

## Next, by what it would tell us

1. **Dense on the anchor-less questions** (language anchors, step 4, GPU): BM25 drops to 12% without numbers and names (step 3, done); does dense hold up? Tests the main caveat.
2. **Own human-written questions** (20-30, Dutch and English, written without looking at the pages): the direct check of the same caveat. Template: `notebooks/own_questions.csv`.
3. **Fusion** (language anchors, step 2c) and an **oracle location filter** ([location](location/README.md)): turn the solid findings into tested design proposals.
4. **Reranker and a larger dense sample**: firm up the dense explanation.

## Experiments

| experiment | expected return | status |
|---|---|---|
| [language_anchors](language_anchors/README.md) | how cross-lingual retrieval works and fails; a cheap fix | steps 1-2b done; step 3, 2c, oracle location filter next |
| [location](location/README.md) | how much resolving the area can do: per kind of place, and as a filter against language locking | candidates per region done; oracle filter next |
| [wiki_audit](wiki_audit/audit.py) | what the dataset can and cannot measure | done ([report](../docs/reports/wiki-audit/audit-2026-10-06.md)) |
| [clue_ablation](clue_ablation/README.md) | how retrieval depends on the amount and type of detail | parked: LLM throughput; partly covered by the LLM-free anchor removal |
| reranker on hard questions | can a cross-encoder recover what dense loses? | not started (GPU) |
| dense / hybrid language study (resource tiers) | language bias of the retrievers the system actually uses | parked until dense is understood |
| own human-written questions | how far generated questions are from real ones | template ready |

## Open questions

- Why exactly dense fails: the look-alike explanation fits, but a bigger model, a task-specific instruction or a reranker are untested.
- How real tourist questions look: no real query data yet.
- Whether language and popularity bias reinforce each other: needs a corpus with several language versions per place (December data?).
