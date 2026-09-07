"""Join the chunk-size sweep runs into one table per metric.

Parsed from the reports rather than retyped: the whole reason the runner emits a
long-form CSV is that unpivoting a markdown table by hand is where transcription
errors come from, and a merged table is exactly where one would hide.

Source-driven rather than two hard-coded runs. The sweep is built up one rung at a
time — 08-14 measured three methods, 08-17 added the pipeline rungs nemotron was
missing, 08-18 added the larger embedders and a larger reranker — and each addition
should be a line in SOURCES and its rows in ORDER, not another branch.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

DOCS = Path("docs/chunk-size-sweeps")
VARIANTS = ["base", "tok256", "tok512", "tok512ov", "tok1024"]
METRICS = ["hit@1", "hit@5", "hit@10", "recall@10", "mrr@10"]
CATEGORIES = [
    "crosslingual",
    "direct_long",
    "direct_short",
    "vague_long",
    "vague_short",
]

# The 08-14 report predates the long-form CSV, so it is parsed back out of its own
# markdown tables. Every run since emits `<output>.cells.csv` and is read from
# that, which is why there is one markdown parser and not one per run.
LEGACY_MD = DOCS / "chunk-size-sweep-2026-08-14.md"
LEGACY_LABEL = "08-14"
LEGACY_METHODS = ["qwen", "qwen_hybrid_rerank", "nemotron"]

# label -> (cell CSVs, methods it contributes). Several CSVs per label because a
# run is split across processes when it has to be — 08-18 measures three embedders
# whose weights do not fit on one card together, so it ran as four invocations of
# one grid rather than four different experiments. Unioning their rows here is what
# keeps that an implementation detail instead of four reports to read side by side.
#
# A CSV that is absent is reported and skipped rather than crashing the merge, so
# this stays runnable while a sweep is still measuring.
CSV_SOURCES = {
    "08-17": (
        [DOCS / "chunk-size-sweep-2026-08-17.md.cells.csv"],
        ["nemotron_hybrid", "nemotron_hybrid_rerank"],
    ),
    "08-18": (
        [
            DOCS / f"chunk-size-sweep-2026-08-18-{leg}.md.cells.csv"
            for leg in ("qwen4b", "qwen8b", "nemotron8b", "rerank4b")
        ],
        [
            "qwen4b",
            "qwen8b",
            "nemotron8b",
            "qwen4b_hybrid_rerank",
            "qwen8b_hybrid_rerank",
            "nemotron8b_hybrid_rerank",
            "qwen8b_hybrid_rerank_4b",
        ],
    ),
}

OUT = DOCS / "chunk-size-sweep-merged-2026-08-18.md"

# Bare dense first, then hybrid, then reranked: reading order is pipeline depth, so
# a jump down a column is the pipeline and a jump across a row is the chunking.
# Within a rung, ascending model size, so a scale effect reads as a trend instead
# of needing a parameter-count lookup. The comment on each row is the embedder's
# size, which the method name does not carry for `qwen`/`nemotron`.
ORDER = [
    ("qwen", "08-14"),  # 0.6B
    ("qwen4b", "08-18"),  # 4B
    ("qwen8b", "08-18"),  # 8B
    ("nemotron", "08-14"),  # 1B
    ("nemotron8b", "08-18"),  # 8B
    ("nemotron_hybrid", "08-17"),  # 1B, + BM25
    ("qwen_hybrid_rerank", "08-14"),  # 0.6B, + BM25 + 0.6B rerank
    ("qwen4b_hybrid_rerank", "08-18"),  # 4B
    ("qwen8b_hybrid_rerank", "08-18"),  # 8B
    ("qwen8b_hybrid_rerank_4b", "08-18"),  # 8B, + 4B rerank
    ("nemotron_hybrid_rerank", "08-17"),  # 1B
    ("nemotron8b_hybrid_rerank", "08-18"),  # 8B
]


# --- legacy markdown parsers ---------------------------------------------------


def parse_old_metrics(text: str) -> dict:
    """{(method, metric): {variant: value}} from the `### hit@5` style sections."""
    out, metric = {}, None
    for line in text.splitlines():
        if line.startswith("### "):
            metric = line[4:].strip()
            continue
        if metric is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in {"method", ""} or set(cells[0]) <= {"-"}:
            continue
        if len(cells) != len(VARIANTS) + 1:
            continue
        try:
            values = [float(c) for c in cells[1:]]
        except ValueError:
            continue
        out[(cells[0], metric)] = dict(zip(VARIANTS, values))
    return out


def parse_old_coverage(text: str) -> dict:
    """{(method, variant): (questions, seconds)} from the Coverage table."""
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5 or cells[0] not in VARIANTS:
            continue
        try:
            out[(cells[1], cells[0])] = (int(cells[2]), float(cells[4]))
        except ValueError:
            continue
    return out


def parse_old_profile(text: str) -> dict:
    """{variant: chunks} from the Chunk Profile table."""
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0] not in VARIANTS:
            continue
        try:
            out[cells[0]] = int(cells[1])
        except ValueError:
            continue
    return out


# --- load every source ---------------------------------------------------------

old_text = LEGACY_MD.read_text(encoding="utf-8")
old = parse_old_metrics(old_text)
old_cov = parse_old_coverage(old_text)
profile = parse_old_profile(old_text)

# Per label so a cell is always attributed to the run that measured it. Keying the
# metric tables by label as well as method means two runs measuring the same method
# cannot silently overwrite each other — which is the failure a merge invites.
values: dict[str, dict] = {LEGACY_LABEL: old}
questions: dict[str, dict] = {
    LEGACY_LABEL: {key: count for key, (count, _) in old_cov.items()}
}
ms_per_query: dict[str, dict] = {
    LEGACY_LABEL: {
        key: seconds * 1000 / count
        for key, (count, seconds) in old_cov.items()
        if count
    }
}
methods_by_label: dict[str, list[str]] = {LEGACY_LABEL: LEGACY_METHODS}
missing_sources: list[str] = []

for label, (paths, methods) in CSV_SOURCES.items():
    by_metric: dict = {}
    counts: dict = {}
    speeds: dict = {}
    for path in paths:
        if not path.exists():
            missing_sources.append(f"{label}: {path.name} not found, its rows omitted")
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                by_metric.setdefault((row["method"], row["metric"]), {})[
                    row["variant"]
                ] = float(row["value"])
                counts[(row["method"], row["variant"])] = int(row["questions"])
                speeds[(row["method"], row["variant"])] = float(row["ms_per_query"])
    if not by_metric:
        continue
    values[label] = by_metric
    questions[label] = counts
    ms_per_query[label] = speeds
    # Only the methods this label actually produced rows for, so a leg that has
    # not run yet does not make its rows look measured-and-empty.
    measured = {method for method, _ in by_metric}
    methods_by_label[label] = [m for m in methods if m in measured]

# Rows whose source never loaded, dropped here rather than rendering as an
# all-"-" row that reads like a measured miss.
rows = [(m, label) for m, label in ORDER if label in values]
labels = sorted({label for _, label in rows})


# --- mergeability: every run must have scored the same sample -------------------

problems = []
for variant in VARIANTS:
    per_label = {}
    for label in labels:
        counted = {
            questions[label][(m, variant)]
            for m in methods_by_label.get(label, [])
            if (m, variant) in questions[label]
        }
        if counted:
            per_label[label] = sorted(counted)
    distinct = {n for counted in per_label.values() for n in counted}
    if len(distinct) > 1:
        detail = ", ".join(f"{label} scored {per_label[label]}" for label in per_label)
        problems.append(f"{variant}: {detail}")

counts_by_variant = {}
for variant in VARIANTS:
    seen = [
        questions[label][(m, variant)]
        for label in labels
        for m in methods_by_label.get(label, [])
        if (m, variant) in questions[label]
    ]
    counts_by_variant[variant] = min(seen) if seen else "-"


# --- tables --------------------------------------------------------------------


def table(metric: str) -> str:
    head = "| method | run | " + " | ".join(VARIANTS) + " |"
    rule = "|---|---|" + "---:|" * len(VARIANTS)
    lines = [head, rule]
    # Bold the best cell per metric so the winner is visible without scanning.
    best = max(
        (
            values[label].get((m, metric), {}).get(v, float("-inf"))
            for m, label in rows
            for v in VARIANTS
        ),
        default=float("-inf"),
    )
    for method, label in rows:
        row = values[label].get((method, metric))
        if not row:
            continue
        cells = []
        for v in VARIANTS:
            if v not in row:
                cells.append("-")
                continue
            mark = "**" if row[v] == best else ""
            cells.append(f"{mark}{row[v]:.3f}{mark}")
        lines.append(f"| `{method}` | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def speed_table() -> str:
    lines = [
        "| method | run | " + " | ".join(VARIANTS) + " |",
        "|---|---|" + "---:|" * len(VARIANTS),
    ]
    for method, label in rows:
        cells = []
        for v in VARIANTS:
            rate = ms_per_query[label].get((method, v))
            cells.append(f"{rate:.1f}" if rate is not None else "-")
        lines.append(f"| `{method}` | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def category_table() -> str:
    """hit@5 per question category, for every run that published one.

    The 08-14 report published no category split, so its rows are simply absent
    rather than blank — a `-` row would read as a measured zero.
    """
    lines = [
        "| method | variant | " + " | ".join(CATEGORIES) + " |",
        "|---|---|" + "---:|" * len(CATEGORIES),
    ]
    any_row = False
    for method, label in rows:
        for v in VARIANTS:
            cells = []
            for cat in CATEGORIES:
                val = values[label].get((method, f"category_hit/{cat}"), {}).get(v)
                cells.append(f"{val:.3f}" if val is not None else "-")
            if any(c != "-" for c in cells):
                any_row = True
                lines.append(f"| `{method}` | {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines) if any_row else ""


# --- report --------------------------------------------------------------------

parts = [
    "# Chunk Size Sweep — Merged " + " + ".join(f"2026-{label}" for label in labels),
    "",
    f"One table per metric across {len(rows)} retrieval methods and",
    f"{len(VARIANTS)} chunk variants, joined from {len(labels)} runs. Derived:",
    "regenerate with `experiments/indexing/merge_chunk_sweeps.py` rather than",
    "editing by hand.",
    "",
    "| source | methods | measured |",
    "|---|---|---|",
]
for label in labels:
    shown = ", ".join(f"`{m}`" for m in methods_by_label.get(label, []))
    if label == LEGACY_LABEL:
        links = f"[`{LEGACY_MD.name}`]({LEGACY_MD.name})"
    else:
        # One link per leg, since a split run has no single report file.
        links = ", ".join(
            f"[`{path.name.removesuffix('.cells.csv')}`]"
            f"({path.name.removesuffix('.cells.csv')})"
            for path in CSV_SOURCES[label][0]
            if path.exists()
        )
    parts.append(f"| {links} | {shown} | 2026-{label} |")

parts += [
    "",
    "## Why these runs can be joined",
    "",
    "Every run used the **per-variant** question design over the same five cuttings",
    "in `.local/db/pages-variants.db`, the same vector store, and the same",
    "post-2026-08-11 sequence lengths, so no column is truncated. Questions scored",
    "per variant, identical in all runs:",
    "",
    "| variant | chunks | questions scored |",
    "|---|---:|---:|",
]
for v in VARIANTS:
    parts.append(f"| {v} | {profile.get(v, '-')} | {counts_by_variant[v]} |")

parts += [
    "",
    (
        "Verified equal across every run, so a row is not comparing two samples."
        if not problems
        else "**Sample mismatch — do not read across runs:**\n"
        + "\n".join(f"- {p}" for p in problems)
    ),
    "",
]
if missing_sources:
    parts += [
        "**Sources not merged:**",
        "",
        *(f"- {note}" for note in missing_sources),
        "",
    ]

parts += [
    "## What each run added",
    "",
    "**08-14** measured three methods, but gave only `qwen_hybrid_rerank` the",
    "hybrid+rerank pipeline; `nemotron` ran as bare dense retrieval. Those cells",
    "differ by 260x in cost (11.3s against 2,922.5s on `base`), so that table",
    "compared pipelines where it meant to compare chunkings.",
    "",
    "**08-17** added the two rungs nemotron was missing, on the same corpus, so",
    "nemotron's row shows its capability rather than its floor.",
    "",
    "**08-18** added scale. Both earlier runs ranked chunkings using only the two",
    "smallest embedders on the machine — Qwen3-Embedding-0.6B and",
    "Nemotron-3-Embed-1B — so every chunking conclusion rested on models small",
    "enough that they might not resolve a difference at all. This run adds",
    "Qwen3-Embedding-4B and 8B, Nemotron-3-Embed-8B, and one larger *reranker*",
    "rung (`qwen8b_hybrid_rerank_4b`, Qwen3-Reranker-4B in place of the 0.6B),",
    "because the finding both earlier runs agreed on is that the reranker dominates",
    "and the reranker doing that work was the smallest in its family.",
    "",
    "Read **down** a column for the pipeline and model-size effect, **across** a",
    "row for the chunking effect. Bold is the best cell in that metric.",
    "",
]

for metric in METRICS:
    parts += [f"## {metric}", "", table(metric), ""]

parts += [
    "## Speed",
    "",
    "Milliseconds per query. The 08-14 figures are derived from its Coverage table",
    "(total seconds / questions scored); later runs measure per query.",
    "",
    speed_table(),
    "",
]

cat = category_table()
if cat:
    parts += [
        "## hit@5 by category",
        "",
        "Only runs that published a category split appear here. This is the part of",
        "the data that disagrees with the aggregate.",
        "",
        cat,
        "",
    ]

parts += [
    "## Read this before quoting a winner",
    "",
    "**No span metric is available in any of these runs.** All used per-variant",
    "question sets, and the `gold` span target is read from `base` chunks",
    "(`src/eval/evaluate.py:load_span_labels`), which those questions have no link",
    "to — so `Anchored` is 0 in every cell and the character-overlap sections are",
    "empty. Everything here is ranked on `hit@k`/`recall@k`, which count whole",
    "chunks and therefore reward a variant for cutting large, independently of how",
    "well it was cut.",
    "",
    "**A variant with more chunks is also a harder haystack**, so `tok256` searches",
    f"{profile.get('tok256', '?')} chunks where `tok1024` searches",
    f"{profile.get('tok1024', '?')}. Read a large gap as real and a small one as",
    "possibly just haystack size.",
    "",
    "To rank these variants defensibly, anchor the answers",
    "(`just anchor-answers`), project the labels (`just relabel <variant>`), and",
    "re-run under `just shared-overview`, where every column scores one shared",
    "question set and the span metrics resolve.",
    "",
    "**Model size is confounded with nothing here, but language coverage is.**",
    "Nemotron-3-Embed's model card declares 42 languages and Slovenian is not one",
    "of them, at either 1B or 8B. This corpus is Slovenian. A nemotron-to-qwen gap",
    "is therefore not evidence about model scale, and the 1B-to-8B step is the row",
    "that separates the two readings.",
    "",
]

OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
for note in missing_sources:
    print(f"skipped {note}")
if problems:
    print("SAMPLE MISMATCH:", *problems, sep="\n  ")
    sys.exit(1)
print(f"sample check: OK, question counts equal across runs {counts_by_variant}")
