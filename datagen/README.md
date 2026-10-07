# Data generation (not used by the pipeline)

How generated columns of the datasets were made, kept as a record. Nothing in `src/`
imports from here; the pipeline only reads the resulting Parquet files through
`src/db/dataset.py`.

| script | made | with |
|---|---|---|
| `notes.py` | `summary` in `wikipages.parquet`, `role` in `chunks.parquet` | Ling 3.1 Flash (free, Vercel AI Gateway) |

The wiki QA questions themselves (`qa/*.parquet`) were generated as described in
`docs/reports/wiki-qgen/wiki-qa-dataset.md`.
