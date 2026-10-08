# Experiments: overview (2026-10-07)

*This file: across experiments, for the team: the main caveat, a rating of every finding, what is ready, priorities. Each experiment's own README is the source of truth for its findings, evidence and decisions.*

## The question behind all of them

**What does retrieval rest on for tourist questions, and where does it break?** Three axes keep coming back:

- **Detail:** which details a question carries (names, numbers, descriptions) and how many.
- **Language:** whether the question is in the page's language.
- **Location:** how precisely the area a question is about can be resolved.

Each experiment is chosen for what its outcome would change in the system: make us **aware** of a failure case, point to a **targeted fix**, or show a technique worth **doubling down** on. Experiments whose outcome is predictable, or would not change a decision, are parked.

## The main caveat on everything below

Hard questions were generated to be unique, so they carry rare numbers and specific details, and their wording copies the page (72% word overlap). **Tested** (stratified, 929 hard questions): removing numbers and names drops BM25 on Swedish hard questions from 41% to 1% and on other languages from 70% to 56%, but dense does not overtake it (other Latin-script languages: BM25 56%, dense 14% without them). So BM25's *scores* on this test set are inflated, but its *lead* over dense (Qwen3-0.6B) holds. Swedish hard questions are lake-register questions found only through numbers; they mainly measure the dataset. **Still untested:** whether the copied page wording inflates BM25 too; only human-written questions can tell.

## How much each finding is worth

Every finding is rated on four dimensions, so it is clear what can go to the team and with which caveat:

- **Robust:** sample size, controls, agreement across methods.
- **General:** does it hold beyond this generated dataset (real users, other corpora)?
- **Surprise:** could we have predicted it beforehand? Low surprise can still be valuable (confirming and sizing a known risk).
- **Value for the team:** what it changes: *aware* (a failure case to design around), *fix* (a targeted remedy), *double down* (a technique worth investing in), or *evaluation* (how to read results on this dataset).

| finding | robust | general | surprise | value for the team |
|---|---|---|---|---|
| **BM25 locks onto the query's language:** a translated hard question is found 0.7% of the time; 99% of its top 10 is in the query's language (3% of the corpus) | **high:** 300 questions, two BM25 engines agree | **high:** mechanical, any lexical retriever on a multilingual corpus | low for the failure, medium for the mechanism (description words drown the shared names and numbers) and its size | **high, aware:** hybrid inherits it through its BM25 side; cross-lingual questions need their own handling |
| **A trivial anchor heuristic recovers it:** numbers + capitalised words of the translated question; hard 2% -> 31% (oracle 33%), easy 45% -> 82% | **high:** stratified 929 + 1,678 questions, 95% intervals; fusion with the full query untested | **medium:** generated translations keep names neatly; real users write exonyms and variants; **German queries only 10%** (every noun is capitalised) | **high:** a regex comes close to the oracle | **high if fusion holds, fix:** cheap, no model (step 2c); needs a German exception |
| **Fusing the full question with its anchors works:** translated hard 2% -> 25%, translated easy 45% -> 77%; same-language questions gain a little (hard 71% -> 74%, easy 92% -> 96%) | **high:** stratified 929 + 1,678 questions, 95% intervals | **medium:** generated translations; weak for German queries | **medium:** that it costs nothing in the same language | **high, fix:** deployable without a model; a concrete proposal for the pipeline's BM25 side |
| **BM25's scores on hard questions depend on the numbers the generator added**, most for Swedish lake questions: without numbers and names Swedish 41% -> 1%, other languages 70% -> 56% (paired) | **high:** stratified, 929 hard questions, 95% intervals | **medium:** the mechanism is general (BM25 needs rare tokens); the size comes from how questions were generated | **medium:** the Swedish dependence | **high, evaluation:** absolute BM25 scores are inflated, Swedish hard questions mainly measure the dataset |
| **But dense does not overtake BM25 without them:** without numbers and names, other Latin-script languages BM25 56% [52-60] vs dense 14% [12-18]; German 72% vs 9% | **high:** 540 + 90 questions, intervals do not overlap | unknown: one dense model, one chunk size, generated wording | **medium:** the expected reversal did not happen | **high, aware:** this dense setup is weak on described places in general; BM25's lead is real here |
| **Dense has no language locking:** easy questions 97% after translation (BM25 38%) | **high:** 300 questions | **high:** multilingual embeddings | low: expected for a multilingual model | **high, double down:** dense for named places across languages, BM25 for detailed descriptions; hybrid combines both |
| **Hard-question specificity sits in the combination:** names 5%, numbers 27%, both 46% | **high** | **low:** real users rarely give exact numbers | medium | **medium, evaluation:** explains why hard questions are findable at all |
| **Dataset facts:** no popularity bias; ~5% of geo questions cross a border; stored dense ranks unusable; `realistic` label lenient and Swedish-only ([audit](../docs/reports/wiki-audit/audit-2026-10-06.md)) | **high:** full dev set | n/a: about this dataset | medium: the lenient label and missing cross-border cases | **high, evaluation:** what the test set can and cannot measure (no cross-border, no recommendation questions) |
| **Hand check:** 15 of 20 hard questions natural, only 3 of 20 tourist questions; the reason is overspecification | **low:** 20 questions, one annotator | unknown | medium | **medium, evaluation:** motivated treating detail as a variable |
| **Region density by category** ([location](location/README.md)): within 10 km, castles ~1 similar page, lakes ~12, Finnish lakes ~63 | **high:** metadata, 702 pages | **high:** real geography | low: lakes cluster, castles do not | **medium-high, double down / aware:** location resolution could settle castle questions, not lake questions |
| **Dense knows the neighbourhood, not the house:** right category 76%, right country 88%, wrong page | **low:** 30 questions, one model, one instruction, no reranker | unknown | low: a known weakness of single-vector retrieval | **medium, hypothesis:** if confirmed, a reranker or more BM25 weight; step 4 first |
| **Smaller chunks help dense a little:** at 256 instead of 512 tokens, dense 22% -> 27% on hard questions (other Latin script), BM25 unchanged; dense stays far behind. Hybrid below BM25 (pilot) | **medium:** 701 questions, intervals touch; hybrid only on 30 | unknown: one model | low-medium | **medium, aware:** chunk size is not the main cause of dense's weakness; smaller chunks are a small win for dense |
| **BM25 splits numbers and ignores units** | high | high | low: known | **low-medium, fix:** number normalisation is cheap |

**Ready for the team now** (robust and valuable): BM25's language locking and why, and that dense does not have it (complementary sides of hybrid); that BM25's scores on hard questions rest on generated numbers, yet dense does not overtake it without them; what the dataset cannot measure; region density by category. **A tested fix to propose:** BM25 on the full question fused with its numbers + capitalised words (cross-lingual 2% -> 25% hard, 45% -> 77% easy, no loss in the same language), with a German caveat. **Not yet:** the dense explanation, hybrid weighting, chunk size.

## What it means for the system (conditional on the caveat)

**Be aware:** dense loses distinguishing details; BM25 locks onto the query's language and hybrid inherits it; a region narrows castles but not lakes; BM25 mangles numbers; the test set overstates real use.

**Fix, to test:** an anchor query fused with the full query for cross-lingual BM25; a language-independent location filter; more BM25 weight for detail-rich questions; number normalisation; a reranker for dense.

**Double down:** BM25's strength on specific details and the anchor heuristic; location resolution for sparse kinds of places; designing for ambiguity (shortlists, asking back) instead of assuming unique answers.

## Next, by what it would tell us

1. ~~Dense on the anchor-less questions~~ (done: dense does not overtake BM25).
2. **Own human-written questions** (20-30, Dutch and English, written without looking at the pages): the direct check of the same caveat. Template: `notebooks/own_questions.csv`.
3. ~~Fusion~~ (done: works). Next: the **oracle location filter** ([location](location/README.md)), and fusion inside the pipeline's hybrid (dense + BM25 + anchors).
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
