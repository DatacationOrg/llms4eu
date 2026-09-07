# Context

## Terms

**Raw page**
A fetched source page stored in the local raw-pages SQLite database with metadata
and Markdown content.

**Markdown page**
The normalized text representation produced by scraping. It is the input to
chunking and should not depend on any retrieval method.

**Page chunk**
A heading-aware slice of a Markdown page, stored with its character span in that
page. The `base` variant targets 1,800 characters, allows up to 2,600 for long
paragraphs, and drops chunks below 300 characters when a page produces several.

**Chunk variant**
A way of cutting pages into chunks: size, overlap, and whether size is counted
in characters or in the embedder's own tokens. `base` is the historical cut that
the approved eval labels point at, and it keeps bare `{page_id}:{index}` ids;
every other variant is suffixed `{page_id}:{variant}:{index}`. Variants coexist
as rows in one `page_chunks` table, discriminated by a `variant` column.

**Chunk version**
How one chunk is turned into text for embedding: `v1` is title + heading + text,
`v2` adds labeled page metadata. Variant and version are independent — any
variant can be indexed with any version.

**Answer anchor**
The character span in a page holding the text that supports one answer, stored
in `eval_answer_anchors`. Placed once by asking the question model to quote its
source verbatim, which is checkable — a quote either appears in the gold chunk or
it does not. Because the span is in page coordinates it survives any re-chunking.

**Span metrics**
Retrieval quality measured in characters of target text rather than in whole
chunks: `char_recall` (how much of the target the retrieved chunks cover),
`char_precision` (how much of the retrieved text is target), and
`budget_recall@N` (recall when N characters of context are filled in rank order).
They exist because `hit@k` and `recall@k` count chunks, so a longer chunk is
likelier to contain any given answer and those metrics rank chunk *variants*
largely by size. Span metrics are the headline for any comparison across
variants; chunk metrics stay valid within one variant.

The target defaults to the **base chunk the question was generated from**, which
is ground truth by construction, needs no model and covers every approved
question. `base` therefore scores a trivial 1.000 and is the reference rather
than a competitor. An [[llms4eu-chunk-labels-are-positional]] answer anchor is
the optional tighter target, and the only reason to pay for one is to rank `base`
alongside the variants.

**Retrieval cost metrics**
What a variant's recall cost, so chunk size cannot decide a comparison on its
own. `store_share@k` is the percentage of a variant's *whole index* one query
returns at k — about 2.0% for a 1,024-token cut against 0.5% for a 256-token one,
measured on this corpus — and `recall_per_share@k` is `char_recall@k` divided by
it: answer coverage earned per percent of the index read. Lower is better for the
first, higher for the second.

They exist because every quality metric here can be bought with size. A cut that
returns four times the text covers more of any answer without retrieving one bit
better, and `char_recall` alone cannot tell the two apart. Read them beside
`budget_recall@N`, which fixes the cost instead of pricing it: agreement between
the two is the strong result, and a variant that wins `char_recall@k` while losing
both of these won on size. Chunk lengths are summed unmerged against every copy
the store holds, so an overlapping cut pays for its duplicates rather than
getting them free — characters of *reading*, where overlap is genuinely
deduplicated, are what `char_precision@k` measures.

**Question density**
How many questions a chunk is asked for. Under the legacy design it is one per
question type whatever the chunk's size, so a variant is probed once per *chunk*
and the smallest cut is probed most densely: measured on the 2026-08-14 sweep
database, 17.6 questions per 10,000 indexed characters for `tok256` against 4.2
for `tok1024` over the same pages — 7,806 questions against 1,456. The large cut
also gets the easier questions, since five questions squeezed out of 256 tokens
reach further down a chunk for facts than five out of 1,024.

`generate_dataset.py --density` scales the count with the chunk instead: one
question per 256 of its own tokens, so 256 gets one, 512 two and 1,024 four, and
every variant ends up with the same number of questions over the same corpus.
Types are then rotated across a variant's chunks rather than exhausted within one,
which is what keeps the category breakdown balanced when a chunk only gets one
question. `src/eval/sweep_status.py` prints the density per variant, so the claim
is checkable rather than assumed — and it needs checking: measured over 163 real
chunks the model delivered 49 of 49 requested for `tok256` but 172 of 215 for
`tok1024`, because one call asked for five pairs fails more often than one asked
for one. `--fill-missing --density` closes that gap; `--fill-missing` without the
flag offers every type for every chunk and silently restores the per-type design.
The budget is capped at five, the number of types, because one field per type is
what makes the request reliable — so the design holds to about 1,280 tokens and a
larger cut is reported as probed below density rather than capped quietly.

Density normalisation removes the sample-size and salience confounds. It does not
remove **home turf** — each variant's questions were still written from its own
chunks, with its own boundaries in view — which is what pooling removes: anchor
the questions, project them onto every variant, and score every variant on the
union. Equal counts are what make that pool balanced rather than dominated by
whichever cut produced the most questions. Declared to a sweep as
`--design per-variant-density`; the sweep tells the designs apart by whether one
question is labelled in more than one variant, not by counting rows, because
matching counts are exactly what density mode produces.

**Chunk variant labels**
Gold labels are per variant while question texts are shared. A question's answer
lives in a different chunk under every cutting, so `src/eval/relabel.py` projects
its anchor onto that variant's chunks by interval overlap — arithmetic, not
judgement, so labelling adds no noise to the scores being compared. Keeping the
questions fixed is what lets one method's score be compared across variants in a
single row.

Note that `eval_questions.approved` is a column default, not a record of human
review: nothing sets it to 0. The questions and answers are model-generated, and
a `base` gold chunk is correct by construction because its question was written
from that chunk.

**Chunk summary**
A short search-oriented description of a page chunk. It is derived content used
for past embedding and retrieval experiments, not a replacement for the chunk
text. Chunk summaries were removed from the steady-state database path after the
experiment.

**Indexer**
A provider-specific embedding backend behind a common interface. Current
indexers are English MiniLM, Qwen multilingual, Qwen 4B, Qwen 8B, and
Nemotron 3 Embed. Qwen 8B is the local stand-in for the removed Azure
`embed-v-4-0`, so a comparison against it is a comparison against the ceiling
the hosted provider used to hold.
A provider can also be declared entirely in `src/indexing/config.yaml` by adding
a `{name}_embedding_model` key. One model at two sequence lengths is two
providers (`qwen`, `qwen_s2048`), so each length's vectors get their own
collection instead of overwriting the other's.

**Source language**
`page_sources.language` is the configured default for a whole source;
`page_metadata.language` is the detected language of one page and wins where it
is set. Reads use `coalesce(m.language, s.language)`. Detection
(`src/preprocess/languages.py`) is read-only until `--apply`, because language
is embedded into the v2 representation and into chunk metadata.

**Ranking method**
A named retrieval strategy used in experiments to return ranked chunk ids for a
query. Examples: sparse, vector-only, hybrid vector+sparse, and reranked
variants.

**Retriever**
Reusable application code that implements a ranking method. Retrievers belong to
the retrieval/RAG layer; eval compares them but does not own their
implementation.

**Eval dataset**
Approved factual questions, answers, and gold chunk ids used to compare ranking
methods. Eval owns labels and metrics; it should not own reusable pipeline steps.
