# Docs

What lives where. Two kinds of document, kept apart: **architecture** is durable
and hand-maintained; **reports** are measurements, dated, and mostly regenerable
by the scripts under `experiments/indexing/`.

## architecture/

| file | what it is |
|---|---|
| [decisions.md](architecture/decisions.md) | the architecture decision record: retrieval ownership, naming, hybrid and rerank history, chunk storage, indexing and config boundaries, OKF, agentic page tools, DCI, geographic scope, local-only inference |
| [geo-retrieval.md](architecture/geo-retrieval.md) | how geo-aware retrieval finds its answer: the five steps, the data model and standards, enrichment tiers, where the code lives, how to run and read it |
| [geo-literature-2023-2026.md](architecture/geo-literature-2023-2026.md) | one entry per 2023–2026 paper and vendor write-up behind the geo plan: citation, method, numbers, how much was read, relevance to this corpus |
| [geo-rationale.md](architecture/geo-rationale.md) | the argument for geo-aware retrieval and for scoring rather than filtering: corpus facts, what the first run showed, the four sources, the five principles, and what would refute it |
| [geo-improvement-plan.md](architecture/geo-improvement-plan.md) | the plan after the 2026-09-08 geo run: why the hard filter lost 15 of 72 scoped questions, what the GIR literature and search vendors do instead, and seven work packages with gates |
| [proposals.md](architecture/proposals.md) | ideas not yet adopted, with adopted ones marked and pointed at their decision |

## reports/

Dated measurements. Read the date and the question set before comparing two
reports: the four measurement eras do not share one (see the 2026-09-07 overview,
§0). Anything before 2026-08-11 was measured at the old 512-token cap.

| folder | contents | produced by |
|---|---|---|
| [retrieval/](reports/retrieval/) | the comprehensive overviews (2026-07-27, **2026-09-07** is the current head), the hand-made eval, and the output of the full comparison (`retrieval-results-full.md`) | `compare_qwen_modes.py`, `just full-comparison` |
| [agentic/](reports/agentic/) | agentic retrieval findings, the 2026-09-01 evaluation report, the tools and DCI runs, the DeepSeek run, the judge-model comparison | `compare_qwen_modes.py`, `compare_chunkings.py` |
| [chunking/](reports/chunking/) | the chunk token audit, the merged chunk-size table, and `sweeps/` with every chunk-size sweep and its `.cells.csv` | `audit_chunk_tokens.py`, `compare_chunkings.py`, `merge_chunk_sweeps.py` |
| [geo/](reports/geo/) | geo diagnostics: the soft-versus-strict comparison of 2026-09-09 (WP1 gate) and the classified gazetteer misses of 2026-09-09 (country hint, exonyms, no inflection) | `compare_qwen_modes.py --methods geo`, `just locate-pages --rejections` |
| [okf/](reports/okf/) | the Open Knowledge Format benchmark and its results; OKF was measured out and is no longer part of the comparison | `compare_okf_rag.py` |

Sidecars sit beside their report: `<report>.md.cells.csv` is the long-form merge
format, `<report>.md.checkpoint.json` is the resumable state (untracked when
large). A `Sources` section at the end of an overview names the reports its
numbers were copied from.

## slides/

Presentation decks (`.pptx`) about the RAG and OKF implementations.

## Conventions

- New reports go under `reports/<topic>/` with the date in the file name; the
  scripts' default output paths already point there.
- A decision that changes behaviour goes into `architecture/decisions.md` in the
  same change as the code; a proposal that is adopted gets a one-line pointer to
  its decision rather than being deleted.
- The glossary is `CONTEXT.md` at the repository root; the experiment protocol
  is `experiments/indexing/README.md`.
