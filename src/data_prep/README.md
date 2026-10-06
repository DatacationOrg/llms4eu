# Data prep

`wiki_places.py` collects the Wikipedia places corpus (`just wiki-collect`, `wiki-fetch`, `wiki-export`).
`wiki_qa.py` loads the QA test set built on it.

## Wiki QA test set

On thebeast (`thebeast.datacation.nl`), readable by the `llms4eu` group:

| path | what |
|---|---|
| `/data/llms4eu/wiki/pages.jsonl` | the index: 133,164 Wikipedia pages about EU places, 24 languages, one row per page with `id`, `title`, `text`, metadata |
| `/data/llms4eu/wiki/qa/wiki_qa_*.parquet` | the questions, one file per question type (below) |
| `/data/llms4eu/wiki/qa/manifest.json`, `SHA256SUMS` | row counts per file / split / language; checksums (`sha256sum -c SHA256SUMS`) |

The data is Wikipedia text and LLM-generated questions, not confidential: copy it where you need it
(set `WIKI_QA_DIR` to the copy). Do not publish it.

```python
from src.data_prep import wiki_qa

hard = wiki_qa.load("rag", ["id", "question", "relevant"],
                    {"answer_ok": True, "kind": "challenge", "split": "test"}).to_pandas()
for row in wiki_qa.rows("unanswerable", where={"ok": True}):   # json columns decoded
    ...
texts = wiki_qa.pages(hard.id.unique())                        # page id -> page row with text
```

Rules for every file: a page id looks like `svwiki/Q123` and points into `pages.jsonl`; test on `ok` rows
(`answer_ok` in `rag`), tune on `split == "dev"` (20%, by place), report on `"test"`. Columns marked *json* hold a
JSON string (`rows()` decodes them). How the set was made: `docs/reports/wiki-qgen/wiki-qa-dataset.md`.

### `rag`: 530,875 questions about one page (498,550 `answer_ok`)

| column | meaning |
|---|---|
| `id` | the page the question is about (gold) |
| `kind` | `corpus` (easy: names the place) or `challenge` (hard: describes it) |
| `question`, `answer`, `qtype` | question, reference answer, type (identify, location, number, date, name, description, reason) |
| `evidence`, `spans`, `verbatim` | quotes from the page, their [start, end) offsets in `text`, all quotes found |
| `answer_ok`, `criteria` *json*, `judges` | passed every answer check; per-check result; number of judges |
| `split`, `balanced` | dev / test; in the ≤ 400 pages per language subset |
| `n_relevant`, `hits`, `relevant`, `partial` | pages the question fits: count, unique / few / many, other full / partial matches |
| `answering`, `hard_negatives`, `qrels_judged`, `gold_match`, `gold_answers`, `pool_saturated` | pages whose text answers; similar wrong pages; were candidates judged; gold page's own judgement; more matches likely than judged |
| `x_lang`, `question_x`, `answer_x`, `x_ok` | the question in another EU language, and whether that translation passed |
| `variants` *json* | keyword, typo, verbose, history and followup rewrites with a check each (balanced pages) |
| `lang`, `n`, `wikidata_id`, `url`, `title`, `country`, `categories`, `latitude`, `longitude`, `sitelinks`, `char_count`, `machine_generated`, `stub` | page language, item number on the page, page metadata |
| `rank_o`, `rank_x` | gold page [dense rank, dense margin, BM25 rank] for the question / its translation |
| `labels` *json*, `page_tags` *json*, `reasoning`, `time_sensitive`, `title_in_question`, `lex_overlap`, `page_template_sim`, `evidence_pos`, `evidence_in_table` | question labels, page tags, single_fact / multi_fact / inference, answer may change, place named, keyword overlap 0-1, similarity to the nearest page, where the evidence sits |

### `unanswerable`: 31,148 questions the index cannot answer (29,315 `ok`)

`id`, `lang`, `title` (the page it was written from), `type` (`false_premise`: the page contradicts it;
`not_covered`: the page lacks it), `question`, `question_en`, `why`, `check` *json*, the qrels columns of `rag`
(`answering` is empty for `ok` rows), `ok`, `split`.

### `compare`: 10,360 questions needing two pages (8,519 `ok`)

`pages` (page A, page B), `id` (= page A), `lang`, `question`, `question_en`, `answer`, `qtype` (number, date,
attribute, common), `evidence_a`, `evidence_b`, `spans_a`, `spans_b`, `verbatim`, `check` *json*, `ok`, `split`.

### `meta`: 3,934 list and geo questions with a set of answers (3,892 `ok`)

`key`, `kind` (`list`: every place of a category in a country; `geo`: places within `radius_km` of `anchor`), `lang`,
`spec` (the task in English), `question`, `question_en`, `gold_pages` (page ids), `gold` (Wikidata ids), `n_gold`,
`anchor`, `radius_km`, `category`, `check` *json*, `ok`, `split`.

### `tables`: 339 aggregation questions over a table in a page (166 `ok`)

`id`, `title`, `in_language`, `table` (header row), `n_rows`, `axis` / `index` / `exclude` / `op` / `unit` (which row or
column, rows skipped, max / min / mean / sum / count / argmax / argmin), `question`, `question_en`, `answer_final`
*json* (the answer to use: `value`, `label` for argmax / argmin), `answer_source` (computed / sonnet / unverified),
`answer` *json* (computed by code), `sonnet` *json*, `sonnet_checked`, `n`, `ok`, `split`.

### `clean`, `challenge`: source items (provenance only, do not test on them)

The easy (`clean`, 354,250) and hard (`challenge`, 285,457) items before answers and judging, with every label:
`id`, `j`, `item` *json*, `labels` *json*; `clean` adds `source`, `repaired`, `original` *json*,
`ambiguous_across_pages`; `challenge` adds `prompt`, `bm25_rank`, `dense_rank`, `challenge_ok`, `likely_ambiguous`,
`question_hard`. `rag` already holds everything that passed.
