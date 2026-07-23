# Archived OKF vs RAG Evidence Acquisition Pilot

> Archived result snapshot from 2026-07-15 09:34 local time.
>
> Provenance: recovered from `.local/reports/okf_rag_2026-07-15_093401.md` and
> `.local/reports/okf_rag_latest.md`. The files were byte-identical
> (`sha256: 813ad4c528604af6d2993104595e42b8fbb7bc7374a11b8538e487513fcb2744`).
> They were ignored local artifacts rather than committed Git files. Git history
> contains the benchmark implementation introduced by commit
> `3b2815721a4f2e42a29b46bbf27b0092d55f6951` (`OKF init`, 2026-07-15), but no
> committed copy of this report.
>
> This is a historical 19-question pilot. It is not directly comparable with the
> later 3,471-question chunk retrieval evaluation because its selection,
> retrieval methods, scoring unit, and OKF bundle differ.

Questions: 19 approved questions whose gold source pages occur in `data/okf/tourism`.

This pilot compares evidence acquisition, not final answer quality. RAG queries count retrieval attempts; OKF queries count Azure navigation actions.

| method              | seconds | ms/query | queries/query | queries |
|---------------------|---------|----------|---------------|---------|
| qwen_hybrid         | 0.10    | 5.1      | 1.00          | 19      |
| qwen_hybrid_agentic | 133.50  | 7026.1   | 2.32          | 44      |
| okf                 | 123.04  | 6475.7   | 3.11          | 59      |

## Page-level evidence

| method              | page hit | page MRR | failures |
|---------------------|----------|----------|----------|
| qwen_hybrid         | 0.579    | 0.363    | 0        |
| qwen_hybrid_agentic | 0.526    | 0.337    | 0        |
| okf                 | 1.000    | 0.974    | 0        |
