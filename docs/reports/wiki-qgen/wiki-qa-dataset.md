# How the wiki places QA test set was made (2026-09-25 to 2026-10-05)

The test set in `/data/llms4eu/wiki/qa/` (models, columns and loader: `src/eval/wiki_qa.py`) is built in twelve stages on top of the Wikipedia places corpus. This report gives, per stage, what was
done, with which model and settings, which filter decides what is kept, and what the checks measured, so the set can
be judged and, roughly, rebuilt. Exact reruns are not possible: the free models are stochastic and change behind
their endpoints. The scripts, prompts and the full decision log are not in the repo: they live on thebeast in
`llms4eu/tmp_labeling/` (`CLEANING_LOG.md` is the day-by-day log, `xcheck/` every blind check's input and result).
The stage table names the script for each step and the columns it fills.

## Models and endpoints

| name | id, endpoint | used for |
|---|---|---|
| Space Bunny | `stealth/space-bunny-alpha`, OpenRouter (`openrouter.ai/api/v1`), free tier, JSON-schema output, <= 50 requests/min, stop on 429 | teacher: page tags, corpus questions, labels, repairs, hard questions, answer judge |
| Ling 3.1 Flash | `inclusionai/ling-3.1-flash-free`, Vercel AI Gateway (`ai-gateway.vercel.sh/v1`), free model, tool-calling output, 300-600 requests/min | labels, repairs, hard questions, answers, judge, every extra layer |
| Qwen3.5-4B + LoRA | `Qwen/Qwen3.5-4B`, LoRA r=8, 1000 steps (`qwen35_qg_s1000`), vLLM, bf16, JSON-schema decoding, temperature 0 | corpus questions (student) |
| Gemma 4 E4B + LoRA | `google/gemma-4-E4B-it`, LoRA r=8, 600 steps (`e4b_cls_s0600`) | page tags (student) |
| Nemotron 1B | `nvidia/Nemotron-3-Embed-1B-BF16`, first 512 dims, "query:"/"passage:" prefixes | near-duplicates, dense ranks |
| BM25 | `bm25s` over title + text of all pages, lowercased `\w+` tokens, no stemming, one index for all languages | distinctive words, ranks, relevance pools |
| Claude Sonnet | subagents, blind (never shown a model's labels) | quality checks only, plus the 200 table answers |

Bunny is a reasoning model: `max_tokens` 32,768 (8,192 made nearly every hard-question request fail); default
reasoning effort (effort=low dropped verdict F1 0.69 -> 0.55). Every intermediate row records the model that made
it, and no model judges its own output in the same call.

## Stages

| # | stage | script (tmp_labeling) | ends up in |
|---|---|---|---|
| 0 | corpus | `src/data_prep/wiki_places.py` (corpus PR) | `/data/llms4eu/wiki/pages.jsonl` |
| 1 | page tags | `tag_spec.py`, `train_lora.py --task cls`, `run_vllm.py` | `rag.page_tags` |
| 2 | corpus questions | `gen_questions.py`, `train_lora.py --task qg`, `run_vllm.py` | `clean` |
| 3 | rule checks | `rule_checks.py` | (suspect flags only) |
| 4 | quality labels, repairs, relabels | `qspec.py`, `bunny_label.py`, `bunny_fix.py`, `bunny_verify.py`, `make_queue.py` | `clean.labels`, `clean.repaired` |
| 5 | cross-page duplicates | `near_dups.py` | `clean.ambiguous_across_pages` |
| 6 | hard questions | `bm25.py`, `gen_challenge.py`, `dense_rank.py rank` | `challenge` |
| 7 | answers, evidence, cross-lingual | `gen_rag.py gen` | `rag` answers, evidence, `*_x` |
| 8 | answer judges | `gen_rag.py judge`, `qspec.RAG` | `rag.criteria`, `answer_ok`, `x_ok` |
| 9 | extra layers | `gen_extra.py` (variants, unans, compare, meta, tables), `tables.py` | `rag.variants`, `unanswerable`, `compare`, `meta`, `tables` |
| 10 | multi-page relevance | `gen_extra.py qrels` | `relevant`, `hits`, ... in `rag`, `unanswerable` |
| 11 | ranks, tags, splits | `dense_rank.py rag` / `bm25`, `build_clean.py` | `rank_*`, tags, `split` in `rag` |
| 12 | export | `build_clean.py` | `wiki_qa_*.parquet`, `manifest.json`, `SHA256SUMS` |

Intermediate results sit in one working SQLite file (`tmp_labeling/wiki_qa.db`), never read by users. Runners
(`run_*.sh`) loop each step 24/7 under `watchdog.sh` and resume where they stopped. Steps only add rows; originals
are never changed, and a replaced layer was first saved to `tmp_labeling/backup/`.

### 0. Corpus

Wikidata places of 12 types (castle, castle ruin, fortification, national park, nature reserve, natural monument,
forest, cave, waterfall, lake, mountain, garden) in the EU27, one SPARQL query per type and country, each place's
article in its own country's language; fetched with the repo's trafilatura pipeline (Markdown text). 133,164 pages,
one per Wikidata id, 24 languages plus a few others; sv 68,955, de 15,145, fi 13,003, fr 6,081, it 5,408, pl 5,351.
`language_ok` checks the declared page language and host; `machine_generated` flags 55,750 Swedish lake register
pages by their "maskinellt skapad" footer (kept).

### 1. Page tags

Eight tags per page: article_type (descriptive, registry_stub, list_or_disambiguation, other), information_richness
(low, medium, high), tourist_appeal (low, medium, high), primary_focus (description, history, visiting, data,
protection, other), and visitor_info, history, nature, culture (booleans). Bunny tagged 547 language-balanced
articles (437 train, 110 held out). The student, Gemma 4 E4B with a LoRA (r=8, alpha 16, dropout 0.05, q/k/v/o and
gate/up/down projections, batch 1, AdamW, lr 2e-4 one-cycle, loss on answer tokens only, 600 steps), agrees with the
teacher on 0.880 of held-out tags (Laya + LoRA 0.827, E4B zero-shot 0.789); it tagged all pages in ~5 h with vLLM.
In the export: `page_tags`.

### 2. Corpus questions (the easy set)

Up to 5 items per page (1-2 for stubs): `question` in the article's language, `question_en`, 1-3 `facts` from the
article, `query` (English keywords from the question only). Prompt rules: what a real person would ask; never
codes, ids, coordinates or register numbers; name or describe the place so it fits no other of ~100k pages; never
"the article"; never the answer in the question; paraphrase. Article cut at 6,000 chars, temperature 0.
Teacher set: Bunny on 1,028 articles, cleaned by Sonnet agents to ~3.4 items per article (`qg_clean.jsonl`, 905
train / 120 held out). Student: Qwen3.5-4B LoRA r=8, 1000 steps; held out: 95% of questions in the article
language, 4% answer leaks; chosen over Gemma E4B, whose Luxembourgish and Lithuanian were broken. Corpus run:
vLLM, ~0.36 s/page. Result: 354k items on 133k pages, 250k by the Qwen LoRA and 104k by Bunny (longer pages).

### 3. Rule checks

Free flags, used as suspects, not filters: near-duplicate on the page (token Jaccard >= 0.8), question or English
not in its language (lingua), numbers of the facts leaking into the query, a fact not found in the article, no word
shared with the title, padded (< 400 article chars per item). Only the duplicate flag has high precision against
gold labels; the others 0.1-0.3.

### 4. Quality labels, repairs, relabels

`qspec.SPEC`, 13 labels per item: duplicate, language_ok, fluent, realistic, self_answering, unique_place,
facts_supported (yes / partly / no), facts_answer, translation_ok, query_ok, answer_difficulty,
retrieval_difficulty, verdict (keep / fix / drop). Rules enforced on every reply (one retry with the errors): keep
with any flag is an error; a duplicate or unsupported item must be drop. One request per page (article <= 60k chars
plus all its items), temperature 0. Agreement with 100 gold pages (355 items, Sonnet one-article labels as gold):
verdict accuracy / macro-F1 Bunny 0.83 / 0.69 (Sonnet bulk 0.86 / 0.72, Sonnet ceiling 0.91 / 0.85); Ling the same
F1 (0.69, 86% agreement with Bunny), stronger on facts_answer, weaker on unique_place.
Repairs: FIX corrects only the listed problems, REPLACE writes a new item on an aspect the page's other items do not
cover (temperature 0.3); the page is then relabelled with the repairs in place, and a repair is used only if the
relabel says keep. Sonnet on 111 Bunny repairs: 94% better, 3.6% new errors; on 30 Ling repairs: 25 better, 3 new
errors. Queue: rule-flagged pages first, then Qwen pages outside Swedish lakes, Bunny pages, Swedish lakes; Bunny
from the front, Ling from the end. Kept easy item = verdict keep and not ambiguous across pages.

### 5. Cross-page duplicates

The same question asked about two pages cannot single out one (e.g. 19 Swedish lakes named Långtjärnen in one
parish). `ambiguous_across_pages` = exact text duplicate on another page, or nearest question on another page with
cosine >= 0.98 (Nemotron, mean-centred, exact neighbours; 0.85-0.95 is the same template about differently named
places). 8,547 items flagged, never removed.

### 6. Hard questions (the challenge set)

Pages >= 1,500 chars, languages round-robin, 5 items per page, temperature 0.6. The prompt gives FORBIDDEN words:
the title tokens and the article's 25 most distinctive words (highest idf x log(1+tf), longer than 2 chars, not
digits, in fewer than 200 pages): what keyword search would match on. Rules: vague and paraphrased, as by someone who
half-remembers the place; never the name or a forbidden word; always one locator (country, region, town, river)
plus properties that fit no other place ("ambiguous is worse than easy"); one question; stands alone (never "the
article", "this place", "here", a bare "it"). Each item: question, question_en, answer (1-3 sentences naming the
place, 2-3 facts), facts, query. Prompt versions v1 -> v3; blind Sonnet keep rate v1 46%, v2 58%; v3 standalone
40/40. Labelled with the same 13 labels; kept = `challenge_ok` (facts_supported yes, not self-answering; fluent,
unique_place, facts_answer, translation_ok, query_ok; duplicate and realistic ignored) and not `likely_ambiguous`
(BM25 rank > 10 and dense rank > 10: 63% of those were ambiguous to Sonnet). Generators: Bunny on hash half 0, Ling
on hash half 1 (`crc32(id) % 2`, language-balanced; blind check: good 26/39 vs 27/40). Rejected on the way: a BM25
hardening rewrite (Sonnet kept 13/30: names lost their uniqueness) and a Qwen LoRA for hard questions (40% used
forbidden words).

### 7. Answers, evidence, cross-lingual

One Ling call per page over its kept items: the answer (copied for hard items), 1-3 evidence sentences copied from
the article, the question type (identify, location, number, date, name, description, reason) and a translation of
question and answer into one of the 23 other EU languages (picked by hash, evenly spread). Evidence is located in the
article ignoring Markdown emphasis, `[n]` footnotes and whitespace, mapped back to raw char offsets (`spans`; 96% of
quotes found, `verbatim`). Answers are regenerated when a page's kept items change.

### 8. Answer judges

Two judges, Ling and Bunny, each judge the answer against context = title + evidence (what a chunk retriever would
return), with the `qspec.RAG` criteria: context_relevant, answerable, answer_relevant, answer_supported,
answer_complete, language_match, answer_fluent, plus x_ok (faithful, fluent translation). `criteria` = AND over the
judges that judged; `answer_ok` = every criterion but x_ok; `x_ok` only when both judged (Ling alone passes 95% of
translations, both 79%). Judges agree on 98.8-100% of answer criteria. Blind Sonnet: answers passing Bunny 28/30
clean; final check of 30 non-sv hard rows: answer correct 30/30, all claims in the quoted evidence 25/30 (a minor
detail from elsewhere in the article), translations are the weakest field (errors in ~1 of 7 passing ones).

### 9. Extra layers (Ling; each generation checked by a second Ling call, each layer blind-checked by Sonnet)

| layer | how | check that decides `ok` | blind Sonnet |
|---|---|---|---|
| query variants (balanced pages) | keyword, typo, verbose, history + followup per item; tags time_sensitive, reasoning | per-variant `*_ok`; followup only for non-identify questions | keyword 29, typo 30, verbose 28, followup 25 of 30 |
| unanswerable | 2 per page: false_premise (contradicted by the article), not_covered (absent) | article check + no pooled page answers it (stage 10) | 28/30 good |
| compare | partner: same language, country, category, by hash; question, answer, verbatim evidence from both | one_comparison, needs_both, answer_supported, identifiable, clean_text + verbatim | ok items 19/22 good |
| list / geo | gold from metadata: every place of a category in a country (2-20) / within 5, 10 or 25 km of an anchor page (1-15, haversine); Ling only phrases | matches_spec, fluent | spec 40/40, natural 36/40 |
| tables | `tables.py` finds data tables (>= 2 named header columns, >= 4 rows, a column >= 70% numeric); Ling picks row/column, totals to exclude, op (max, min, mean, sum, count, argmax, argmin); code computes | Sonnet recomputed 200: answer right 171/200; `answer_final` = computed if confirmed, else Sonnet's; ok = question sensible | 166 ok |

Lesson from these checks: Ling's self-checks are lenient on form (fragments, joined questions, garbled words); meta
and compare were redone with stricter prompts after the first Sonnet check.

### 10. Multi-page relevance (qrels)

For hard questions, easy questions of balanced pages and unanswerable items: pool = BM25 top 20 for the question +
the gold page, shuffled; Ling judges every candidate from an excerpt (lead 700 chars + the 6 sentences sharing the
most idf weight with the question, words matched on 5-letter prefixes): match yes / partly / no, answers. Gold page
judged yes 94-99% (a blind control). Sonnet agreement: 123/125 and 206/210 candidates; "fits another page" 30/30 and
29/30 questions. Derived: `relevant`, `partial`, `answering`, `hard_negatives`, `n_relevant`, `hits` (unique / few
/ many), `pool_saturated` (>= 10 of 20 match). Pages only a dense model would find are unjudged (BM25 pool).

### 11. Ranks, tags, splits

`rank_o` / `rank_x` = [dense rank, dense margin, BM25 rank] of the gold page for the original / translated
question (dense: title + first 1,000 chars per page, kept only where the answers row is unchanged since the GPU run).
Computed tags: `lex_overlap` (share of the question's idf weight found in the page), `title_in_question`, `stub`
(< 2,000 chars), `page_template_sim` (max cosine of lead embeddings), `evidence_pos`, `evidence_in_table`.
`split`: dev if `crc32(wikidata_id) % 5 == 0`, else test. `balanced`: the first 400 pages per language by id hash.

### 12. Export

`build_clean.py` reads the working file (never changing it), joins the layers, converts to zstd Parquet (one schema:
union of keys, absent = null; dict-valued columns as JSON strings) through a temp file and rename, then
`manifest.json` and `SHA256SUMS`. Frozen build: 2026-10-05 07:43.

## Result (2026-10-05 build)

| file | rows | ok |
|---|---|---|
| `wiki_qa_rag` | 530,875 | 498,550 (easy 307,144; hard 191,406: sv 98.3k, other languages 93.1k) |
| `wiki_qa_unanswerable` | 31,148 | 29,315 |
| `wiki_qa_compare` | 10,360 | 8,519 |
| `wiki_qa_meta` | 3,934 | 3,892 |
| `wiki_qa_tables` | 339 | 166 |

BM25 gold page top 1 / top 10: easy 86-93% / 95-100%; hard sv 28% / 57%; hard other 53% / 75%; cross-lingual hard
~0%. About 25% of Swedish hard questions fit several near-identical lake stubs (`hits` few / many); every other slice
is ~95-98% unique.

## Known limits

- Ling and Bunny are free preview models; outputs will differ on a rerun, and both endpoints change over time.
- One model family checks most extra layers; Sonnet checks are 30-200 item samples.
- The qrels pool is BM25 only, evidence does not always cover every detail of an answer, and translations stay
  noisy in small languages (mt, ga, bg, lt, lv).
- sv is the largest language (lake stubs): report per language or on `balanced`.
