# Docs

Two kinds of document, kept apart: **architecture** describes the code that
exists and changes with it; **reports** are dated measurements and stay as they
were written, including the commands and modules of their time.

## Architecture

| file | what it is |
|---|---|
| [architecture-decisions.md](architecture-decisions.md) | the decision record: what the pipeline does and why |
| [architecture-proposals.md](architecture-proposals.md) | ideas not yet adopted |

## reports/

Read the date and the question set before comparing two reports: the
measurement eras do not share one (see the 2026-09-07 overview, §0). Anything
before 2026-08-11 was measured at the old 512-token cap.

| folder | contents |
|---|---|
| [all-runs-unified-2026-09-14.md](reports/all-runs-unified-2026-09-14.md) | **start here**: every result with more than 150 samples, May to September 2026 |
| [retrieval/](reports/retrieval/) | the comprehensive overviews (**2026-09-07** is the current head), the hand-made eval, the geo runs |
| [chunking/](reports/chunking/) | the chunk token audit, the merged chunk-size table, every sweep |
| [agentic/](reports/agentic/) | agentic retrieval, page tools, DCI and judge comparisons; measured out |
| [geo/](reports/geo/) | the soft-versus-strict geo run, gazetteer misses, and the design, rationale, literature and plan of the full geo implementation that preceded today's minimal one |
| [okf/](reports/okf/) | the Open Knowledge Format benchmark; measured out |
| [scraping/](reports/scraping/) | seed-URL quality for the expanded corpus (the URL lists live under `/data/llms4eu`) |
| [wiki-qgen/](reports/wiki-qgen/) | the Wikipedia places QA test set: question generator and labels (2026-09-30), how the set was built (`wiki-qa-dataset.md`) |

Sidecars sit beside their report: `<report>.md.cells.csv` is the long-form table.

## slides/

Presentation decks about the RAG and OKF implementations.
