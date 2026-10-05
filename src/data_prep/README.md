# Data prep

`wiki_places.py` collects the Wikipedia places corpus (`just wiki-collect`, `wiki-fetch`, `wiki-export`).
`wiki_qa.py` loads the QA test set built on it.

## Wiki QA test set

On thebeast (`thebeast.datacation.nl`), readable by the `llms4eu` group:

| path | what |
|---|---|
| `/data/llms4eu/wiki/pages.jsonl` | the index: 133,164 Wikipedia pages about EU places, 24 languages |
| `/data/llms4eu/wiki/qa/` | the test set: `wiki_qa_*.parquet`, `README.md` (columns, examples, what each group tests), `manifest.json` (counts), `SHA256SUMS` |

The data is Wikipedia text and LLM-generated questions, not confidential: copy it where you need it
(set `WIKI_QA_DIR` to the copy). Do not publish it.

```python
from src.data_prep import wiki_qa

hard = wiki_qa.load("rag", ["id", "question", "relevant"],
                    {"answer_ok": True, "kind": "challenge", "split": "test"}).to_pandas()
for row in wiki_qa.rows("unanswerable", where={"ok": True}):   # JSON columns decoded
    ...
texts = wiki_qa.pages(hard.id.unique())                        # page id -> page row with text
```

Files (`load(name)`): `rag` (the main set), `unanswerable`, `compare`, `meta` (list/geo), `tables`, and the
source items `clean`, `challenge`. Test on rows with `ok` / `answer_ok`, tune on `split == "dev"`. A question's
correct pages are its `id` plus `relevant`. How the set was made: `docs/reports/wiki-qgen/wiki-qa-dataset.md`.
Check the files: `cd /data/llms4eu/wiki/qa && sha256sum -c SHA256SUMS`.
