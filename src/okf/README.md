# Open Knowledge Format

Generates and consumes a minimal OKF v0.1 tourism bundle from complete scraped
pages. Generation reads complete Markdown from `data/db/pages.db`; navigation
reads only the generated OKF hierarchy and complete concept files.

## Generation

The resumable pipeline has three phases:

1. Azure proposes one primary concept for each complete source page.
2. Exact matching and batched Azure resolution canonicalize proposals against a
   persistent global catalog.
3. Azure creates or augments coherent Markdown concepts from assigned pages.

Private inventory, catalog, and enrichment checkpoints live under
`.local/okf/`. Concept and checkpoint writes use temporary sibling files and
atomic replacement. Completed pages are skipped on resume.

Pages above 60,000 characters are split at Markdown headings and paragraph
boundaries into target parts of at most 45,000 characters. Part-level discovery
is consolidated into one page concept, while part-level enrichment progress is
checkpointed. This remains necessary: the current source corpus contains a
264,863-character page and another page above 60,000 characters. Splitting is a
generation safeguard only; navigation reads complete OKF concepts.

Required environment variables:

- `AZURE_AI_ENDPOINT`
- `AZURE_AI_API_KEY`
- `AZURE_AI_MODEL`

Run generation and validation with:

```bash
just okf-pilot
just okf-generate
just okf-rebuild
just okf-validate
just okf-ask What is Rajhenburg Castle?
```

### Resume and clean rebuilds

- `just okf-generate` resumes from `.local/okf/` and leaves completed work alone.
- `just okf-rebuild` uses `--clean`: it deletes `data/okf/tourism/` plus the
   complete `.local/okf/` state directory, then rebuilds all pages from
   `pages.db`. Clean mode rejects source filters and limits to avoid replacing a
   full bundle with a partial one.

Use a clean rebuild after schema, taxonomy, or prompt changes. It is destructive
and invokes Azure for the full corpus, so keep a Git commit or external copy if
the previous bundle must remain recoverable. Azure configuration is checked
before any generated files are deleted.

## Documents and indexes

Generated concept frontmatter contains only:

- `type`
- `title`
- `description`
- `timestamp`
- `source_page_ids`, retained only for page-level evaluation compatibility

The body is coherent Markdown with a final standard `# Citations` section.
Aliases, tags, languages, synthetic search terms and queries, source-evidence
mappings, atomic-fact metadata, and retrieval-readiness metadata are not stored.

Indexes are regenerated from deepest directories upward. Entries expose only a
concept title and description. Collections above 20 entries are split into
bounded browse indexes. Validation requires parseable frontmatter, the four
required descriptive fields, and a non-empty body. Escaping links are errors;
missing internal links are warnings.

The generated manifest records source selection, resumability counts, failures,
concept counts, model deployment, and validation findings. It never stores
credentials.

## Navigation and evaluation

The answer command starts at the root index and follows only advertised relative
Markdown links. It may open several complete concept files within configured
step, document, and character budgets. Opened concept files are answer evidence;
indexes route but are not valid citations. Answers cite exact opened concept
paths. The navigator may open another advertised concept when the files read so
far do not contain enough information.

The evaluation adapter ranks cited concepts first and then other visited
concepts. Golden concepts are derived by mapping each gold chunk to its source
page and then through concept `source_page_ids`. RAG results use the same
projection, so retrieving a translated or duplicate sibling page gets full
concept credit. There is one `okf` mode: no BM25 metadata shortlist, raw SQLite
source window, or retrieval-readiness filter participates in navigation.

For fair comparison with chunk RAG, hold the question set, answer model, context
budget, and load constant and report concept retrieval separately from native
chunk retrieval and answer correctness. The complete protocol is in
[`experiments/indexing/README.md`](../../experiments/indexing/README.md).

## Tests

```bash
uv run --extra dev pytest tests/test_okf.py
uv run --extra dev ruff format src/okf tests/test_okf.py
uv run --extra dev ruff check src/okf tests/test_okf.py
```

Current limitations include one primary concept per page, model-assisted
canonicalization, source-page rather than claim-level benchmark attribution, no
automatic orphan pruning during resume runs, and prompts that can grow
when many pages enrich one concept. A clean rebuild removes orphans.
