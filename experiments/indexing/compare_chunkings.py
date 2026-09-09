"""Vary the chunking instead of the retriever.

Every published retrieval number in this project was measured on one chunking:
1,800 target characters, no overlap, character sized. This runner makes chunk
size, overlap and size unit measured choices rather than assumptions.

Two axes move together, because they are one decision seen from two sides:

- **Chunk size.** How much text goes into one indexed document.
- **Sequence length.** How much of that document the embedder actually reads.
  `embedding_max_seq_length: 512` is a project setting, not a model limit —
  Qwen3-Embedding-0.6B declares 32,768 positions. The 2026-08-11 audit measured
  69.7% of v1 inputs over 512, so smaller chunks and a higher cap are competing
  fixes for the same truncation and cannot be judged apart. Sequence lengths
  enter the grid as separate providers (`qwen_s1024`, `qwen_s2048`), so each one
  indexes into its own collection.

Two labelling designs are supported, and `--design` declares which one the
database holds:

- `shared` — one question set, labels projected onto each variant by interval
  overlap (`src/eval/relabel.py`). Every column scores the same questions, so one
  method's score for every variant belongs in one row, and the `gold` span target
  resolves for free.
- `per-variant` — each variant owns questions generated from its own chunks. Gold
  is correct by construction, but the columns are differently sized samples and
  the `gold` span target is structurally empty, because that target is read from
  base chunks these questions have no link to.

Declaring it is not bureaucracy: the two are indistinguishable from any single
cell, and a grid that mixes them publishes rows that cannot be compared. The
2026-08-14 sweep ranked variants on `recall@10` precisely because it ran the
per-variant design with no span metric available, and `recall@10` rewards large
chunks by construction.

This runner reports methods against variants across every dimension the
single-variant comparison reported: span and chunk metrics, latency, the category
split, and — behind flags — `judge_hit@K` and the paired agentic diagnostics.

Each cell reports retrieval quality beside that variant's truncation, because a
variant whose chunks exceed the sequence limit is being silently cut, and its
scores describe a corpus the embedder never fully saw.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from _cli import run_cli
from src.eval.agentic_diagnostics import (
    baseline_for_agent,
    build_agentic_diagnostics,
    format_agentic_diagnostics,
)
from src.eval.equivalence import EvidenceEquivalenceJudge
from src.eval.evaluate import (
    add_judge_adjusted_scores,
    category_hit_scores,
    load_eval_rows,
    load_store_profile,
    run_eval,
)
from src.eval.evaluate import CONFIG as EVAL_CONFIG
from src.eval.metrics import bold_best_table, plain_table
from src.indexing.chunk_text import chunk_text_representation
from src.shared.tokenizers import (
    huggingface_tokenizer,
    provider_model_name,
    provider_token_limit,
)
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.retrieval.methods import missing_retriever_indexes
from src.shared.env import ROOT, load_local_env, load_yaml
from src.vector_store.chunks import load_chunks

INDEX_CONFIG = load_yaml(ROOT / "src" / "indexing" / "config.yaml")
EXPERIMENT_CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
RETRIEVAL_CONFIG = load_yaml(ROOT / "src" / "retrieval" / "config.yaml")

DEFAULT_VARIANTS = ("base", "tok512")
DEFAULT_METHODS = ("qwen", "qwen_hybrid_rerank")
DEFAULT_OUTPUT = ROOT / "docs" / f"chunk-size-sweep-{date.today().isoformat()}.md"

# Named method sets, so a run meant to line up with an earlier report does not
# depend on retyping six names correctly.
#
# `docs/reports/chunking/sweeps/chunk-size-sweep-2026-08-14.md` is the report these are built to extend.
# It measured three methods over the five variants:
#
#   qwen                  vector only
#   nemotron              vector only
#   qwen_hybrid_rerank    hybrid + cross-encoder rerank
#
# Only one of the three got the hybrid+rerank pipeline, so `nemotron`'s row there
# is an 11-second cell sitting beside a 2,922-second one: that table compares
# pipelines where it means to compare chunkings, and nemotron's numbers are its
# bare-dense floor rather than its capability.
#
# `sweep-completion` is what that report is missing — nemotron given the same
# pipeline qwen already had, plus an agent on each of the two providers already in
# the table. Run it on the same variants and the same per-variant question sets and
# its rows drop straight into that report's tables.
#
# Merging is a join on (variant, method, metric); `<output>.cells.csv` carries
# every cell in that long form, with each cell's own question count, so a merged
# table cannot hide a difference in sample size.
METHOD_GROUPS = {
    "sweep-completion": (
        "nemotron_hybrid",
        "nemotron_hybrid_rerank",
        "nemotron_hybrid_agentic",
        "qwen_hybrid_agentic",
    ),
    # Nemotron's half of the above, and the part that needs no LLM at all.
    "sweep-completion-cheap": ("nemotron_hybrid", "nemotron_hybrid_rerank"),
    "sweep-completion-agents": ("nemotron_hybrid_agentic", "qwen_hybrid_agentic"),
    # Every embedder on one pipeline, so a column difference is the chunking and
    # not the retrieval stack. `qwen8b` is the local stand-in for the removed
    # Azure `embed-v-4-0`, so this is also the closest thing to the ceiling that
    # provider used to hold.
    "symmetric": (
        "sparse_rerank",
        "qwen_hybrid_rerank",
        "qwen4b_hybrid_rerank",
        "qwen8b_hybrid_rerank",
        "nemotron_hybrid_rerank",
    ),
    "agentic": (
        "nemotron_hybrid_rerank",
        "nemotron_hybrid_agentic",
        "nemotron_hybrid_agentic_tools",
        "qwen8b_hybrid_rerank",
        "qwen8b_hybrid_agentic",
    ),
    "plain": ("sparse", "qwen", "qwen4b", "qwen8b", "nemotron"),
    # Reasoning effort as its own axis. The 2026-09-01 run measured the whole
    # agentic family at `low` only, and its two worst results were a tool agent
    # that failed to emit valid JSON on 7% of steps and a DCI agent that ran out
    # of steps without answering -- both failures reasoning effort plausibly
    # moves. Both agents at both rungs, plus the shared baseline, so a rung's
    # effect on the tools can be told apart from its effect on the judge under
    # them.
    "reasoning": (
        "qwen_hybrid_rerank",
        "qwen_hybrid_agentic",
        "qwen_hybrid_agentic_high",
        "qwen_hybrid_agentic_tools",
        "qwen_hybrid_agentic_tools_high",
    ),
    "reasoning-dci": ("sparse", "dci", "dci_high"),
    # The 2026-08-18 completion: the merged 08-14/08-17 report ranks chunkings
    # using only the two smallest embedders on the machine (0.6B and 1B), so
    # "which cutting wins" was measured on models that may be too small to show a
    # difference. These are the larger rungs of the same two families, on the same
    # per-variant question sets, so the rows join that report's tables directly.
    #
    # Both rungs per provider, deliberately. The bare-vector row is where the
    # embedder is the only thing acting, so it is the only place a scale effect is
    # visible at all; the reranked row is the published pipeline, and the merged
    # report's own finding is that reranking compresses embedder differences.
    # Reading one without the other is what let nemotron look 260x cheaper than
    # qwen in the 08-14 table.
    "large-embedders": (
        "qwen4b",
        "qwen8b",
        "nemotron8b",
        "qwen4b_hybrid_rerank",
        "qwen8b_hybrid_rerank",
        "nemotron8b_hybrid_rerank",
    ),
    # Bare vector only: every embedder in the house on the one rung where the
    # embedder is the only variable. Cheap enough to run over the whole grid.
    "embedder-ladder": (
        "qwen",
        "qwen4b",
        "qwen8b",
        "nemotron",
        "nemotron8b",
    ),
    # The reranker rung. Scaling the stage the merged report found dominant
    # (+0.131 hit@5 on `base` against a 0.069 spread across every chunking),
    # against the same first stage, so the only change is the cross-encoder.
    "reranker-ladder": (
        "qwen8b_hybrid_rerank",
        "qwen8b_hybrid_rerank_4b",
    ),
}

# Reported per cell, in this order. Anything a method does not produce shows as
# "-" rather than a real-looking zero.
#
# Span metrics lead, chunk metrics follow. `hit@k` and `recall@k` count whole
# chunks and so reward a variant for cutting large, which is the one thing this
# sweep must not conflate with retrieval quality; they stay in the report because
# they are the numbers every earlier report used, not because they settle it.
# Three, not the dozen this started as. `char_recall` is whether the answer's
# region came back, `char_precision` is what it cost to deliver it, and
# `budget_recall` is the two traded off under one context allowance. `iou` and a
# second budget said nothing the first three did not.
SPAN_METRICS = ("budget_recall@4000", "char_recall@10", "char_precision@10")
CHUNK_METRICS = ("hit@1", "hit@5", "hit@10", "recall@10", "mrr@10")
# What the recall cost. `store_share@10` is the percentage of the variant's whole
# index one query returns at k=10, and `recall_per_share@10` is `char_recall@10`
# divided by it — coverage earned per percent of the index read. They are here
# because every other metric in this grid can be bought with chunk size: a cut
# that returns four times the text covers more of any answer without retrieving
# any better, and until 2026-08-20 nothing in the report charged it for that.
COST_METRICS = ("store_share@10", "recall_per_share@10")
HEADLINE_METRICS = (*SPAN_METRICS, *COST_METRICS, *CHUNK_METRICS)
# Metrics a reader wants small, so the tables bold the cheapest cell instead of
# the most expensive one.
LOWER_IS_BETTER = ("store_share@1", "store_share@5", "store_share@10")

# What "best" means in the recommendation, first available wins. Recall at a fixed
# character budget is the decision-relevant one: the generator's context window,
# not k, is what is actually scarce downstream — and it is already size-neutral,
# which is why `recall_per_share@10` sits beside it in the report rather than
# ahead of it here. The ratio is how to read a `char_recall` or `hit@k` win; at a
# fixed budget the cost is fixed by construction and the recall is the answer.
PRIMARY_METRICS = ("budget_recall@4000", "recall@10")

# Variants scored on samples differing by more than this are not compared at all.
# Exact equality is too strict: an anchor on a page whose variant chunks have no
# resolved span drops one question from one column, which is not a reason to
# refuse a five-variant table.
SAMPLE_SPREAD = 0.01

# Which labelling design produced the grid. It is declared rather than detected
# because the two designs are indistinguishable from one cell: both put rows in
# `eval_relevant_chunks` keyed by question and chunk. Declaring it lets the run
# fail loudly when the database does not match, instead of publishing a table
# whose rows silently mix a shared question set with per-variant ones.
#
#   shared       one question set, labels projected per variant by
#                `src/eval/relabel.py`. Every column scores the same questions,
#                so `gold` span targets resolve and the rows are comparable.
#                A grid of per-variant questions that has been anchored and
#                relabelled onto every variant is this design too, and should be
#                declared as this: it is one pooled question set by then.
#   per-variant  each variant owns questions generated from its own chunks, one
#                question per type per chunk. Gold is correct by construction,
#                but the columns are differently sized samples — 7,801 questions
#                against 1,451 — and `gold` span targets are structurally empty,
#                the target being read from base chunks these questions have no
#                link to.
#   per-variant-density
#                each variant owns its questions, but a chunk is asked for one
#                question per 256 of its own tokens rather than one per type
#                (`generate_dataset.py --density`), so the columns are the same
#                size and probe the corpus at the same density. Removes the
#                sample-size confound from `per-variant`; leaves the home-turf
#                one, which only pooling removes.
DESIGNS = ("shared", "per-variant", "per-variant-density")

# Below this share of questions labelled in more than one variant, the database
# holds per-variant question sets rather than one projected set. It is a floor,
# not a target: the shared design should score near 1.0, and anything in between
# means relabelling stopped half way.
SHARED_QUESTION_SHARE = 0.5

# Used only by --dry-run, so an hours-long grid can be priced before it is
# started rather than after.
#
# Reranking is charged per token, not per query, because the cross-encoder reads
# every candidate: the same method costs 0.31 s/query on `tok256` and 1.01 s/query
# on `tok1024`, and a flat rate overestimated the whole grid by 68%. Rate against
# that variant's median chunk length is near-constant, so proportionality through
# the origin is the model.
#
# Refitted 2026-08-17 on the five `nemotron_hybrid_rerank` cells of
# docs/reports/chunking/sweeps/chunk-size-sweep-2026-08-17.md (per-cell 0.00119-0.00149, total
# seconds / total query-tokens = 0.001324). The previous 0.00116 came from a
# single cell of the 2026-08-14 sweep and ran 12-28% low against every cell of
# this one.
#
# Agentic cost is LLM latency and barely moves with chunk size, so it stays flat.
SECONDS_PER_RERANK_TOKEN = 0.001324
# Per-token rate for the named reranker rungs in
# src/retrieval/config.yaml:rerankers. Without this every rung is priced as the
# 0.6B, and a cross-encoder's cost scales with its parameter count: quoting a 4B
# row at the 0.6B rate understates it several-fold, which is exactly the mistake
# --dry-run exists to prevent.
#
# Measured rather than extrapolated from the parameter ratio, on the same
# `rerank_candidate_limit` of 30 candidates the sweep actually reranks. The 6.7x
# parameter ratio between 0.6B and 4B is not the throughput ratio: both models run
# the same 30 pairs per query and short candidates leave the GPU
# latency-bound rather than compute-bound.
SECONDS_PER_RERANK_TOKEN_BY_RERANKER = {"4b": 0.00816}
SECONDS_PER_QUERY = {"agentic": 5.1, "plain": 0.006}
JUDGE_SECONDS_PER_QUERY = 2.0
# Share of queries that miss at the report's cutoff and so reach the judge.
JUDGED_MISS_RATE = 0.2


@dataclass
class Cell:
    """One (variant, method) measurement, reduced to what the report needs.

    Aggregates only, deliberately. The 2026-07-27 comprehensive run persisted
    per-question rankings into its checkpoint and left a 96 MB JSON file in
    `docs/`; a five-variant grid would multiply that. Rankings are used while the
    cell is in memory and dropped afterwards, except under --judge-equivalence,
    which needs them post-hoc and gets its own sidecar file.
    """

    variant: str
    method: str
    scores: dict[str, float] = field(default_factory=dict)
    questions: int = 0
    seconds: float = 0.0
    span_questions: int = 0
    ms_per_query: float = 0.0
    queries_per_query: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def main() -> None:
    load_local_env()
    args = _parse_args()
    variants = _values(args.variants)
    methods = _resolve_methods(args.methods)
    if not variants or not methods:
        raise ValueError("At least one variant and one method are required")

    audit = {variant: _variant_audit(variant) for variant in variants}
    for variant in variants:
        if not audit[variant]["chunks"]:
            raise RuntimeError(
                f"No {variant} chunks in SQLite. Build them first with "
                f"`uv run python -m src.preprocess.chunks --variant {variant} ...`."
            )

    _check_design(variants, args.design)

    planned, skipped, refitted = _plan(variants, methods, audit, fit=not args.no_fit)
    if not planned:
        raise RuntimeError("Every cell was pruned or is missing its index")
    for note in refitted:
        print(f"refit: {note}")
    for note in skipped:
        print(f"skipped: {note}")

    if args.dry_run:
        print(_cost_estimate(planned, variants, methods, audit, args))
        return

    checkpoint = Path(args.checkpoint or f"{args.output}.checkpoint.json")
    state = _load_state(checkpoint, variants, methods, args)
    cells = _run(planned, state, checkpoint, args)

    report = _format_report(
        cells=cells,
        variants=variants,
        methods=methods,
        audit=audit,
        skipped=skipped,
        refitted=refitted,
        command=" ".join(sys.argv),
        args=args,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")
    print(report)
    rows_path = _write_cell_rows(Path(f"{args.output}.cells.csv"), cells, args)
    print(f"\nSweep report saved to: {output}")
    print(f"Mergeable cell rows saved to: {rows_path}")
    if not args.keep_checkpoint:
        checkpoint.unlink(missing_ok=True)


def _run(planned, state, checkpoint: Path, args) -> list[Cell]:
    cells: list[Cell] = []
    for index, (variant, method) in enumerate(planned, start=1):
        key = f"{variant}|{method}"
        if key in state["runs"]:
            stored = state["runs"][key]
            cells.append(Cell(variant, method, **stored))
            print(f"[{index}/{len(planned)}] {key}: resumed from checkpoint")
            continue

        started = time.perf_counter()
        run = run_eval(
            _cell_methods(method, args),
            limit=args.limit,
            category=args.category,
            warmup=args.warmup,
            variant=variant,
            span_metrics=args.span_metrics,
            span_target=args.span_target,
            store_metrics=True,
        )
        elapsed = time.perf_counter() - started
        if args.judge_equivalence and run.questions:
            run = add_judge_adjusted_scores(
                run,
                _judge_summaries(run, method, args.judge_k, Path(args.judge_cache)),
                args.judge_k,
            )
        cell = Cell(
            variant=variant,
            method=method,
            scores={
                name: value
                for name, value in run.scores.get(method, {}).items()
                if isinstance(value, (int, float))
            },
            questions=len(run.questions),
            seconds=elapsed,
            span_questions=run.span_questions,
            ms_per_query=run.timings.get(method, {}).get("ms_per_query", 0.0),
            queries_per_query=run.timings.get(method, {}).get("queries_per_query", 0.0),
            category_scores=(
                category_hit_scores(run).get(method, {}) if run.questions else {}
            ),
            diagnostics=_diagnostics(run, method),
        )
        cells.append(cell)
        state["runs"][key] = {
            "scores": cell.scores,
            "questions": cell.questions,
            "seconds": cell.seconds,
            "span_questions": cell.span_questions,
            "ms_per_query": cell.ms_per_query,
            "queries_per_query": cell.queries_per_query,
            "category_scores": cell.category_scores,
            "diagnostics": cell.diagnostics,
        }
        _write_state(checkpoint, state)
        print(
            f"[{index}/{len(planned)}] {key}: {cell.questions} questions "
            f"in {elapsed:.1f}s"
        )
    return cells


def _write_cell_rows(path: Path, cells, args) -> Path:
    """One tidy row per (variant, method, metric), for merging with other reports.

    The markdown tables are shaped for reading: metric per section, variants as
    columns. Merging those with a single-variant report means unpivoting them by
    hand, which is where transcription errors come from. This is the same numbers
    in long form, so combining runs is a join on (method, metric).

    `variant` is the column the older reports do not have; they are all one
    cutting, so joining them in means treating their rows as `base`.
    """
    rows = ["design,variant,method,metric,value,questions,span_questions,ms_per_query"]
    for cell in cells:
        measures = {**cell.scores}
        measures.update(
            {
                f"category_hit/{name}": value
                for name, value in cell.category_scores.items()
            }
        )
        for metric, value in measures.items():
            rows.append(
                f"{args.design},{cell.variant},{cell.method},{metric},{value:.6f},"
                f"{cell.questions},{cell.span_questions},{cell.ms_per_query:.3f}"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _check_design(variants: list[str], design: str) -> None:
    """Fail before the first cell when the database does not hold `design`.

    Sample size used to be the whole test: `shared` projected one set onto every
    cutting so the counts matched, `per-variant` sized each set by its own chunk
    count so they did not. `--density` breaks that test by design — its whole
    point is per-variant sets of matching size — so the primary observable is now
    whether the *same question* is labelled in more than one variant. That is
    structural rather than incidental: a per-variant question id derives from a
    chunk id and chunk ids carry the variant, so a per-variant question cannot
    appear in two columns however evenly the counts come out.

    Sample size stays as the secondary test, separating the two per-variant
    designs from each other: matching counts mean density mode ran, differing
    ones mean it did not. Getting that backwards is what a reader cannot see in
    the finished table.
    """
    labelled = {
        variant: {row["id"] for row in load_eval_rows(variant=variant)[0]}
        for variant in variants
    }
    counts = {variant: len(ids) for variant, ids in labelled.items()}
    missing = [variant for variant, count in counts.items() if count == 0]
    if missing:
        remedy = (
            "uv run python -m src.eval.relabel --variant <name>"
            if design == "shared"
            else "uv run python -m src.eval.generate_dataset --variant <name>"
            + (" --density" if design == "per-variant-density" else "")
        )
        raise RuntimeError(
            f"No labelled questions for {', '.join(missing)} under the "
            f"'{design}' design. Label them first: `{remedy}`. Counts: {counts}"
        )
    if len(variants) < 2:
        return

    union = set().union(*labelled.values())
    shared_share = sum(
        1
        for question_id in union
        if sum(question_id in ids for ids in labelled.values()) > 1
    ) / len(union)
    sizes = sorted(set(counts.values()))
    even = sizes[-1] - sizes[0] <= SAMPLE_SPREAD * sizes[-1]

    if design == "shared" and shared_share < SHARED_QUESTION_SHARE:
        raise RuntimeError(
            "--design shared expects one question set labelled in every "
            f"variant, but only {shared_share:.1%} of questions appear in more "
            f"than one. Either this database holds per-variant questions (use "
            f"--design per-variant, or per-variant-density if they were "
            f"generated with --density) or relabelling is unfinished (`uv run "
            f"python -m src.eval.relabel --variant <name>`). Counts: {counts}"
        )
    if design != "shared" and shared_share >= SHARED_QUESTION_SHARE:
        raise RuntimeError(
            f"--design {design} expects each variant to own its questions, but "
            f"{shared_share:.1%} of them are labelled in more than one variant. "
            "This database looks like the shared design; use --design shared."
        )
    if design == "per-variant" and even:
        raise RuntimeError(
            "--design per-variant expects each variant's set to be sized by its "
            f"own chunk count, but the counts match: {counts}. These look "
            "density-normalised; use --design per-variant-density."
        )
    if design == "per-variant-density" and not even:
        raise RuntimeError(
            "--design per-variant-density expects every variant probed at the "
            f"same question density, so matching counts, but they differ: "
            f"{counts}. Either generation is unfinished, or these are the "
            "legacy one-question-per-type sets (use --design per-variant)."
        )


def _cost_estimate(planned, variants, methods, audit, args) -> str:
    """What the grid will cost, before it is started rather than after.

    Prices the grid that was *asked for*, not the one that can run today: a cell
    whose vector index is missing is listed with its estimate and marked, because
    the point of pricing is to decide whether to build that index at all. `_plan`
    prunes those cells, so pricing only the planned ones would quote a grid for
    the run and a different one for the decision.
    """
    counts = {variant: len(load_eval_rows(variant=variant)[0]) for variant in variants}
    runnable = set(planned)
    lines = [
        f"Requested cells: {len(variants) * len(methods)} "
        f"({len(planned)} runnable now)",
        f"Design: {args.design} | span target: {args.span_target}",
        "",
        "| variant | method | questions | kind | median tok | estimated | status |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    total = 0.0
    blocked = 0.0
    for variant in variants:
        for method in methods:
            questions = (
                counts[variant]
                if args.limit is None
                else min(args.limit, counts[variant])
            )
            kind = _cost_kind(method)
            median = _median_tokens(variant, method, audit)
            reranker = _method_reranker(method)
            seconds = questions * _seconds_per_query(kind, median, reranker)
            if args.judge_equivalence:
                seconds += questions * JUDGED_MISS_RATE * JUDGE_SECONDS_PER_QUERY
            if args.agentic_diagnostics and baseline_for_agent(method):
                seconds += questions * _seconds_per_query("rerank", median, reranker)
            ready = (variant, method) in runnable
            if ready:
                total += seconds
            else:
                blocked += seconds
            lines.append(
                f"| {variant} | {method} | {questions} | "
                f"{kind if reranker is None else f'{kind}:{reranker}'} | "
                f"{median or '-'} | {_duration(seconds)} | "
                f"{'ready' if ready else 'needs index'} |"
            )
    lines.extend(["", f"**Estimated total, runnable now: {_duration(total)}**"])
    if blocked:
        lines.append(
            f"**Plus {_duration(blocked)} once the missing indexes are built** "
            f"(total {_duration(total + blocked)}). Build them with "
            "`just eval-index <provider> v1 <variant>`."
        )
    lines.extend(
        [
            "",
            "Reranking is priced against each variant's median chunk length "
            f"({SECONDS_PER_RERANK_TOKEN} s/token for the default reranker, "
            + ", ".join(
                f"{rate} for {name}"
                for name, rate in SECONDS_PER_RERANK_TOKEN_BY_RERANKER.items()
            )
            + "), which landed within 20% on "
            "every cell of the 2026-08-14 sweep; agentic cost is flat at "
            f"{SECONDS_PER_QUERY['agentic']} s/query. Treat it as an estimate. "
            "The run checkpoints per cell, so stopping it costs one cell.",
        ]
    )
    return "\n".join(lines)


def _cost_kind(method: str) -> str:
    if "agentic" in method or method.startswith("dci"):
        return "agentic"
    return "rerank" if "rerank" in method else "plain"


def _median_tokens(variant: str, method: str, audit: dict) -> int:
    """Median chunk length under the provider this cell actually embeds with.

    Capped at the reranker's own input limit, past which longer chunks are
    truncated and cost no more.

    A method with no provider in its name still reranks — `sparse_rerank` runs the
    same cross-encoder over the same chunk texts, and pricing it at zero because
    it has no embedder would quote a 40-minute cell as free. Fall back to the
    default provider's median, since what the reranker reads is the chunk, not
    whatever shortlisted it.
    """
    medians = audit.get(variant, {}).get("median_tokens", {})
    provider = _method_provider(method) or INDEX_CONFIG["default_provider"]
    median = medians.get(provider) or max(medians.values(), default=0)
    return min(median, RETRIEVAL_CONFIG["reranker_max_length"])


def _method_reranker(method: str) -> str | None:
    """The named reranker rung a method carries, or None for the default.

    Longest match first for the same reason as `_method_provider`: one configured
    name may be a suffix of another.
    """
    configured = RETRIEVAL_CONFIG.get("rerankers") or {}
    for name in sorted(configured, key=len, reverse=True):
        if method.endswith(f"_rerank_{name}") or f"_rerank_{name}_" in method:
            return name
    return None


def _seconds_per_query(kind: str, median_tokens: int, reranker=None) -> float:
    if kind == "rerank":
        rate = SECONDS_PER_RERANK_TOKEN_BY_RERANKER.get(
            reranker, SECONDS_PER_RERANK_TOKEN
        )
        return median_tokens * rate
    return SECONDS_PER_QUERY[kind]


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _cell_methods(method: str, args) -> list[str]:
    """The methods one cell runs: the requested one, plus a baseline if diagnosed.

    Agentic diagnostics are a paired measurement, so the baseline has to run in
    the same cell. That is a real extra cost — the baseline is re-measured once
    per variant — which is why it is opt-in rather than automatic.
    """
    if not args.agentic_diagnostics:
        return [method]
    baseline = baseline_for_agent(method)
    return [method] if baseline is None else [method, baseline]


def _diagnostics(run, method: str) -> dict:
    """Paired agent-versus-baseline diagnostics, empty when the pair is absent.

    `build_agentic_diagnostics` compares an agent with its non-agentic baseline,
    so it needs both in the same run. A cell that asked for diagnostics ran the
    pair together; a cell that did not returns nothing rather than a table of
    zeroes that would read as "the agent never retried".
    """
    if len(run.methods) < 2 or not run.action_logs.get(method):
        return {}
    # `rankings` must travel with the action log: the diagnostics align actions
    # to questions by zipping the log's attempt-1 groups against the ranking's
    # question ids. Without it every question gets an empty action list and the
    # table reports zero retries and zero tool calls for an agent that made
    # hundreds (the 2026-08-31 and 2026-09-01 reports).
    return build_agentic_diagnostics(
        questions=run.questions,
        relevance=run.relevance,
        rankings=run.rankings,
        method_states={
            name: {
                "action_log": run.action_logs.get(name, []),
                "rankings": run.rankings.get(name, {}),
            }
            for name in run.methods
        },
        method_names=run.methods,
        cutoff=EVAL_CONFIG["category_hit_k"],
    )


def _judge_summaries(run, method: str, cutoff: int, cache_path: Path) -> dict:
    """Equivalence-judge summaries for one cell, reusing the checkpoint judge.

    `judge_checkpoint` already owns the caching, the strict-hit short circuit and
    the audit report; it just expects the comparison runner's checkpoint shape.
    Building that shape from one cell's run is cheaper and far less error-prone
    than a second judging loop that would drift from the first.
    """
    from judge_retrieval_equivalence import _build_client, _write_json, judge_checkpoint

    client, model_id = _build_client(None)
    _report, cache, summaries = judge_checkpoint(
        state={
            "signature": {
                "timed_ids": [str(row["id"]) for row in run.questions],
                "scoring_unit": "chunk",
            },
            "methods": {name: {"rankings": run.rankings[name]} for name in run.methods},
        },
        judge=EvidenceEquivalenceJudge(client=client),
        model_id=model_id,
        cutoff=cutoff,
        cache_path=cache_path,
    )
    _write_json(cache_path, cache)
    return summaries


def fit_provider(variant: str, provider: str, audit: dict) -> str:
    """Smallest sequence length of the same model that reads `variant` whole.

    A chunk size is only meaningfully tested against a model that can actually
    read it. Pairing a 1024-token cut with a 512-token limit measures truncation,
    not chunk size, and reports it as if it were chunk size. Rather than keep a
    hand-written variant-to-provider table in step with the grid, derive it: take
    the variant's real peak token count under this model's own tokenizer and pick
    the shortest configured sibling that clears it.

    Returns `provider` unchanged when it already fits, or when nothing fits — the
    caller reports the shortfall rather than silently swapping in something else.
    """
    profile = audit.get(variant, {})
    model = profile.get("models", {}).get(provider)
    limit = profile.get("limits", {}).get(provider)
    peak = profile.get("max_tokens", {}).get(provider, 0)
    if model is None or limit is None or peak <= limit:
        return provider

    # Siblings share the model and therefore the tokenizer, so one peak applies.
    siblings = sorted(
        (sibling_limit, name)
        for name, sibling_limit in profile["limits"].items()
        if profile["models"].get(name) == model and peak <= sibling_limit
    )
    return siblings[0][1] if siblings else provider


def _refit(method: str, provider: str, fitted: str) -> str:
    """The same retrieval method on the fitted provider."""
    return fitted + method[len(provider) :]


def _plan(
    variants, methods, audit, fit: bool = True
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Cells to run, notes for cells left out, and notes for cells refitted.

    Three reasons a cell changes or disappears: its vector index is missing, a
    shorter sequence length already covers the same corpus without truncating, or
    the requested sequence length cannot read the variant whole and a longer one
    can. All three are reported — a silently shrunk or silently swapped grid
    reads as full coverage when it is not.
    """
    planned: list[tuple[str, str]] = []
    skipped: list[str] = []
    refitted: list[str] = []
    covered: dict[tuple[str, str, str], int] = {}

    for variant in variants:
        for method in methods:
            provider = _method_provider(method)
            if fit and provider:
                fitted = fit_provider(variant, provider, audit)
                if fitted != provider:
                    peak = audit[variant]["max_tokens"][provider]
                    refitted.append(
                        f"{variant}|{method} -> {_refit(method, provider, fitted)} "
                        f"(peak {peak} tokens exceeds {provider}'s "
                        f"{audit[variant]['limits'][provider]}; "
                        f"{fitted} reads it whole)"
                    )
                    method = _refit(method, provider, fitted)
                    provider = fitted
            missing = missing_retriever_indexes([method], variant)
            if missing:
                skipped.append(
                    f"{variant}|{method}: index not built ({missing[method]})"
                )
                continue
            limit = audit[variant]["limits"].get(provider)
            # Per provider: tokenizers disagree, and a cross-provider maximum
            # would judge one model's truncation by another's tokenizer.
            peak = audit[variant]["max_tokens"].get(provider, 0)
            model = audit[variant]["models"].get(provider)
            if fit and limit is not None and peak > limit:
                # Fitting was asked for and nothing longer exists, so this cell
                # would report truncation as if it were chunk size. Under
                # --no-fit the truncation is the point, and the cell is kept.
                skipped.append(
                    f"{variant}|{method}: {provider} truncates it "
                    f"({peak} tokens over a {limit} limit) and no configured "
                    "sequence length fits; its score would measure truncation"
                )
                continue
            if limit is not None and model is not None and peak <= limit:
                # Nothing is truncated at this length, so a longer one on the
                # same model produces byte-identical vectors. Keyed on the
                # retrieval family too — `qwen` and `qwen_hybrid_rerank` are
                # different methods on one provider, not two lengths of one
                # method, and collapsing them would gut the grid.
                family = method[len(provider) :]
                # Reaching here means this length does not truncate the variant,
                # and so did the one already recorded. Two non-truncating lengths
                # of one model give identical vectors whichever order they arrive
                # in, so no comparison of the lengths is needed.
                seen = covered.get((variant, model, family))
                if seen is not None:
                    skipped.append(
                        f"{variant}|{method}: duplicate of {model} at sequence "
                        f"length {seen} for the same retrieval method "
                        f"(peak {peak} tokens fits both)"
                    )
                    continue
                covered[(variant, model, family)] = limit
            planned.append((variant, method))
    return planned, skipped, refitted


def _variant_audit(variant: str) -> dict:
    """Token profile of one variant against every configured provider limit."""
    chunks = load_chunks(variant)
    # Read from the same place `store_share@k` divides by, so the denominator in
    # the metric and the size printed beside it cannot drift apart.
    _, store_chars = load_store_profile(variant)
    profile = {
        "chunks": len(chunks),
        "store_chars": store_chars,
        "max_tokens": {},
        "over_limit": {},
        "median_tokens": {},
        "limits": {},
        "models": {},
    }
    if not chunks:
        return profile

    representation = chunk_text_representation("v1")
    documents = [representation.text_for_embedding(chunk) for chunk in chunks]
    for provider in INDEX_CONFIG["providers"]:
        try:
            # Tokenizer only, never model weights: Nemotron needs a GPU to load
            # and would otherwise be dropped from the audit without a word.
            tokenizer = huggingface_tokenizer(provider)
            limit = provider_token_limit(provider)
            model_name = provider_model_name(provider)
        except Exception as error:
            print(f"audit: no local tokenizer for {provider} ({error})")
            continue
        counts = sorted(
            len(tokenizer(document, add_special_tokens=True)["input_ids"])
            for document in documents
        )
        profile["limits"][provider] = limit
        profile["models"][provider] = model_name
        profile["median_tokens"][provider] = counts[len(counts) // 2]
        profile["over_limit"][provider] = sum(
            1 for count in counts if count > limit
        ) / len(counts)
        profile["max_tokens"][provider] = counts[-1]
    return profile


def _method_provider(method: str) -> str:
    """The embedding provider a method name carries, longest match first."""
    for provider in sorted(INDEX_CONFIG["providers"], key=len, reverse=True):
        if method == provider or method.startswith(f"{provider}_"):
            return provider
    return ""


def _format_report(
    *, cells, variants, methods, audit, skipped, refitted, command, args
) -> str:
    by_key = {(cell.variant, cell.method): cell for cell in cells}
    # A refit changes which method actually ran, so the table rows are the
    # methods that were measured, not the ones that were requested.
    methods = list(dict.fromkeys(cell.method for cell in cells)) or methods
    lines = [
        f"# Chunk Size Sweep {date.today().isoformat()}",
        "",
        "## Run Configuration",
        "",
        f"- Command: `{command}`",
        f"- Variants: {', '.join(variants)}",
        f"- Methods: {', '.join(methods)}",
        f"- Questions: {args.limit if args.limit is not None else 'all'}"
        f" | Category: {args.category or 'all'} | Warmup: {args.warmup}",
        f"- Question-set design: **{args.design}**"
        + (
            " — one question set, labels projected onto each variant by "
            "`src/eval/relabel.py`, so every column scores the same questions "
            "and one method's row is comparable straight across."
            if args.design == "shared"
            else " — each variant owns questions generated from its own chunks "
            "at one question per 256 of a chunk's own tokens, so gold is correct "
            "by construction and every column probes the corpus at the same "
            "density. What remains is home turf: each variant's questions were "
            "written with its own boundaries in view."
            if args.design == "per-variant-density"
            else " — each variant owns questions generated from its own chunks, "
            "one per type per chunk, so gold is correct by construction but the "
            "columns are differently sized samples and the small-chunk variants "
            "were probed more densely."
        ),
        f"- Span target: **{args.span_target}**"
        + (
            " — the base chunk the question came from, so `base` scores a "
            "trivial 1.000 and is the ruler rather than a competitor."
            if args.span_target == "gold"
            else " — the quoted sentence stating the answer, so `base` competes "
            "on the same footing as every other cutting."
        ),
        "",
        "## Chunk Profile",
        "",
        _audit_table(variants, audit),
        "",
        "A variant whose inputs exceed a provider's sequence limit is being "
        "truncated: its scores describe a corpus the embedder never fully read.",
        "",
        "## Retrieval Quality (character overlap)",
        "",
        "Measured against the answer's character span, not against whole chunks. "
        "`char_recall` is how much of the answer the top-k chunks cover, "
        "`char_precision` how much of the retrieved text is answer, `iou` the two "
        "together. `budget_recall@N` fills N characters of context in rank order "
        "and asks how much of the answer got in — the comparison that decides an "
        "indexing choice, since the generator's context window is what is scarce.",
        "",
    ]
    lines.extend(_metric_sections(SPAN_METRICS, variants, methods, by_key))
    lines.extend(
        [
            "## Retrieval Cost (share of the index)",
            "",
            "What the recall above cost. `store_share@10` is the percentage of "
            "the variant's whole index one query returns at k=10 — about 2.0% "
            "for a 1,024-token cut against 0.5% for a 256-token one, because k "
            "slots of a larger chunk are simply more text. **Lower is "
            "better here, so the bold cell in that table is the cheapest method "
            "for that variant rather than the largest number.** "
            "`recall_per_share@10` divides `char_recall@10` by it: "
            "answer coverage earned per percent of the index read.",
            "",
            "This is the column that stops chunk size deciding the ranking on "
            "its own. Every other quality metric here can be bought with size — "
            "return four times the text and cover more of any answer without "
            "retrieving one bit better — and until this section existed nothing "
            "in the report charged a cut for the text it returned. Two cuts at "
            "equal `char_recall` are separated by what each had to read to get "
            "there. Read it beside `budget_recall@4000`, which fixes the cost "
            "instead of pricing it: agreement between the two is the strong "
            "result, and a variant that wins `char_recall@10` while losing both "
            "of these won on size.",
            "",
        ]
    )
    lines.extend(_metric_sections(COST_METRICS, variants, methods, by_key))
    lines.extend(["## Retrieval Quality (whole chunks)", ""])
    if args.design.startswith("per-variant"):
        lines.extend(
            [
                "Each variant is scored on questions generated from its own "
                "chunks, so its gold chunk is correct by construction and no "
                "variant is measured on questions written for another cutting.",
                "",
                *(
                    [
                        "Question density is normalised: a chunk is asked for "
                        "one question per 256 of its own tokens, so 256 gets "
                        "one, 512 two and 1,024 four, and every variant covers "
                        "the corpus with the same number of questions. Without "
                        "that, one question per type per chunk probes a "
                        "256-token cut four times more densely than a "
                        "1,024-token one over the same pages — and reaches "
                        "further down each chunk for the facts to ask about, so "
                        "the large cut gets the more salient questions as well "
                        "as the smaller sample.",
                        "",
                    ]
                    if args.design == "per-variant-density"
                    else []
                ),
                "The confound that replaces it: **a variant with more chunks is "
                "a harder haystack.** See the chunk counts above — finding one "
                "chunk among 2,003 is harder than among 501, independently of "
                "chunk quality, so a small-chunk variant is penalised for "
                "reasons that have nothing to do with how well it was cut. Read "
                "a large gap as real and a small one as possibly just haystack "
                "size.",
                "",
                "And the one no per-variant design removes: **each variant's "
                "questions were written from its own chunks, with its own "
                "boundaries in view.** Anchor these questions and relabel them "
                "onto every variant, then rerun as `--design shared`, and that "
                "goes too — density normalisation is what makes the pooled set "
                "balanced rather than dominated by whichever cut produced the "
                "most questions.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Every variant is scored on the same questions, with its labels "
                "projected from the shared answer anchors, so no column is "
                "measured on questions written for its own cutting and the "
                "sample size is identical across the row.",
                "",
                "The confound that remains: **a variant with more chunks is a "
                "harder haystack**, and `hit@k` counts whole chunks, so a "
                "larger chunk is likelier to contain any given answer. Both "
                "effects survive shared labelling. The span metrics above are "
                "the ones to rank on; these are here because they are the "
                "numbers every earlier report used.",
                "",
            ]
        )
    lines.extend(_metric_sections(CHUNK_METRICS, variants, methods, by_key))
    lines.extend(_speed_section(variants, methods, by_key))
    lines.extend(_category_sections(variants, methods, by_key, cells))
    lines.extend(_diagnostics_section(cells))

    if refitted:
        lines.extend(
            [
                "## Sequence Length Fit",
                "",
                "Each variant is measured on the shortest configured sequence "
                "length that reads its chunks whole. Pairing a cut with a limit "
                "below it would measure truncation and report it as chunk size.",
                "",
                *(f"- {note}" for note in refitted),
                "",
            ]
        )
    lines.extend(["## Coverage", "", _coverage_table(cells), ""])
    if skipped:
        lines.extend(
            [
                "## Skipped Cells",
                "",
                *(f"- {note}" for note in skipped),
                "",
            ]
        )
    lines.extend(
        ["## Recommendation", "", _recommendation(cells, variants, audit, args.design)]
    )
    return "\n".join(lines)


def _speed_section(variants, methods, by_key) -> list[str]:
    """Latency per cell, because an indexing choice is paid for at query time.

    Milliseconds per query rather than total seconds: the variants hold different
    numbers of questions under the per-variant design, so totals would rank the
    smallest question set fastest and say nothing about the method.
    """
    rows = []
    for method in methods:
        values = [by_key.get((variant, method)) for variant in variants]
        if all(cell is None or not cell.ms_per_query for cell in values):
            continue
        rows.append(
            [
                method,
                *(
                    f"{cell.ms_per_query:.1f}" if cell and cell.ms_per_query else "-"
                    for cell in values
                ),
            ]
        )
    if not rows:
        return []
    extra = [
        [
            method,
            *(
                f"{cell.queries_per_query:.2f}"
                if (cell := by_key.get((variant, method))) and cell.queries_per_query
                else "-"
                for variant in variants
            ),
        ]
        for method in methods
    ]
    lines = [
        "## Speed",
        "",
        "Milliseconds per query, so the columns stay comparable when the "
        "variants hold different numbers of questions.",
        "",
        plain_table(["method", *variants], rows),
        "",
    ]
    if any(any(value != "-" for value in row[1:]) for row in extra):
        lines.extend(
            [
                "Retrieval calls per query — above 1.00 means a method issued "
                "extra searches for some queries.",
                "",
                plain_table(["method", *variants], extra),
                "",
            ]
        )
    return lines


def _category_sections(variants, methods, by_key, cells) -> list[str]:
    """One table per question category, variants as columns.

    The category split is where a chunking choice shows its character:
    `crosslingual` questions are answered from Slovenian text, and `vague_short`
    is where retrieval is weakest, so a cutting that helps on average can still
    lose the cases that matter.
    """
    types = sorted({key for cell in cells for key in cell.category_scores})
    if not types:
        return []
    lines = [f"## hit@{EVAL_CONFIG['category_hit_k']} by category", ""]
    for question_type in types:
        rows = [
            [
                method,
                *(
                    f"{cell.category_scores[question_type]:.3f}"
                    if (cell := by_key.get((variant, method)))
                    and question_type in cell.category_scores
                    else "-"
                    for variant in variants
                ),
            ]
            for method in methods
        ]
        if all(all(value == "-" for value in row[1:]) for row in rows):
            continue
        lines.extend(
            [
                f"### {question_type}",
                "",
                bold_best_table(["method", *variants], rows),
                "",
            ]
        )
    return lines


def _diagnostics_section(cells) -> list[str]:
    """Paired agent-versus-baseline tables, one per cell that produced one."""
    diagnosed = [cell for cell in cells if cell.diagnostics]
    if not diagnosed:
        return []
    lines = ["## Agentic Diagnostics", ""]
    for cell in diagnosed:
        table = format_agentic_diagnostics(cell.diagnostics)
        if table:
            lines.extend([f"### {cell.variant} | {cell.method}", "", table, ""])
    return lines if len(lines) > 2 else []


def _metric_sections(metrics, variants, methods, by_key) -> list[str]:
    """One table per metric, skipping any no method produced."""
    lines: list[str] = []
    for metric in metrics:
        rows = [
            [
                method,
                *(
                    _cell_value(by_key.get((variant, method)), metric)
                    for variant in variants
                ),
            ]
            for method in methods
        ]
        if all(all(value == "-" for value in row[1:]) for row in rows):
            continue
        lines.extend(
            [
                f"### {metric}",
                "",
                bold_best_table(
                    ["method", *variants],
                    rows,
                    lower_is_better=metric in LOWER_IS_BETTER,
                ),
                "",
            ]
        )
    return lines


def _audit_table(variants, audit) -> str:
    providers = sorted(
        {provider for variant in variants for provider in audit[variant]["limits"]}
    )
    lines = [
        "| Variant | Chunks | Indexed chars | "
        + " | ".join(f"{p} max/over limit" for p in providers)
        + " |",
        "|---|---:|---:|" + "---:|" * len(providers),
    ]
    for variant in variants:
        profile = audit[variant]
        cells = " | ".join(
            f"{profile['max_tokens'].get(p, 0)} tok, "
            f"{profile['over_limit'].get(p, 0):.1%} over {profile['limits'][p]}"
            if p in profile["limits"]
            else "-"
            for p in providers
        )
        lines.append(
            f"| {variant} | {profile['chunks']} | "
            f"{profile.get('store_chars', 0):,} | {cells} |"
        )
    lines.append("")
    lines.append(
        "`Indexed chars` is what `store_share@k` is a share of: every copy the "
        "store holds, so an overlapping cut shows the text it duplicates rather "
        "than hiding it. Variants over the same pages should differ here only by "
        "overlap and by the chunks each cut drops as too short."
    )
    return "\n".join(lines)


def _coverage_table(cells) -> str:
    lines = [
        "| Variant | Method | Questions scored | Anchored | Seconds |",
        "|---|---|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell.variant} | {cell.method} | {cell.questions} | "
            f"{cell.span_questions} | {cell.seconds:.1f} |"
        )
    lines.append("")
    lines.append(
        "`Anchored` is the sample the character-overlap metrics average over. "
        "Under the `gold` span target it equals the labelled sample, because "
        "every question has a base chunk. Under `anchor` it is smaller, and the "
        "same shared sample for every variant — anchors are placed once, before "
        "any chunking — so the columns stay comparable and the shortfall costs "
        "statistical power, not validity. A zero here means the span metrics "
        "never ran: the target set was empty."
    )
    return "\n".join(lines)


def _cell_value(cell: Cell | None, metric: str) -> str | float:
    """The number, not a rendering of it.

    `bold_best_table` formats floats itself to the same three decimals and can
    only bold what it can compare, so stringifying here silently disabled bolding
    in every metric section — no published sweep report has a bold cell in one.
    That matters now that a cost metric is reported beside the quality ones: the
    bold has to be able to fall on the smallest cell of a `store_share` column.
    """
    if cell is None or metric not in cell.scores:
        return "-"
    return float(cell.scores[metric])


def _recommendation(cells, variants, audit, design: str = "shared") -> str:
    metric = next(
        (name for name in PRIMARY_METRICS if any(name in c.scores for c in cells)),
        None,
    )
    if metric is None:
        return (
            "No cell produced any of "
            f"{', '.join(PRIMARY_METRICS)}; nothing can be recommended."
        )
    scored = [cell for cell in cells if metric in cell.scores]
    caveat = (
        ""
        if metric != "recall@10"
        else (
            " **Ranked on `recall@10` because no span metric was available**, and "
            "that metric favours large chunks by construction: anchor the answers "
            "first (`uv run python -m src.eval.anchors`) before trusting this "
            "ordering."
        )
    )

    counts = {cell.variant for cell in scored}
    # Guard the sample the chosen metric actually averages over. A span metric
    # scores anchored questions, which is one set shared by every variant; a chunk
    # metric scores whatever each variant is labelled for, and `base` keeps its
    # original generated labels rather than anchor-derived ones, so those counts
    # differ by design and a mismatch there really does block comparison.
    sizes = sorted(
        {
            cell.span_questions if metric in SPAN_METRICS else cell.questions
            for cell in scored
        }
    )
    remedy = (
        "Anchor the remaining answers first: `uv run python -m src.eval.anchors`."
        if metric in SPAN_METRICS
        else "Finish relabelling every variant before reading this table: "
        "`uv run python -m src.eval.relabel --variant <name>`."
    )
    uneven = len(sizes) > 1 and sizes[-1] - sizes[0] > SAMPLE_SPREAD * sizes[-1]
    if uneven and metric in SPAN_METRICS:
        # Span metrics score one shared target set, so unequal samples there mean
        # the targets are incomplete rather than that the design differs.
        return (
            f"Variants were scored on different numbers of questions ({sizes}) "
            f"on `{metric}`, so the columns are not comparable. {remedy}"
        )
    spread = ""
    if len(sizes) > 1:
        spread = (
            # Not an error under per-variant questions: each variant owns a
            # question set sized by its own chunk count. Each question is scored
            # independently, so the scores stay comparable; the smaller samples
            # are simply noisier.
            f" Question counts differ across variants ({sizes}), which is "
            "expected when each variant owns its question set — the smaller "
            "samples are noisier, not biased."
            if design.startswith("per-variant")
            else f" Question counts differ across variants ({sizes}) even though "
            "the shared design projects one question set onto each — that means "
            "some variant has chunks with no resolved span, so treat the short "
            "columns as incomplete rather than as worse."
        )

    best = max(scored, key=lambda cell: cell.scores[metric])
    baseline = [
        cell
        for cell in scored
        if cell.variant == BASE_CHUNK_VARIANT and cell.method == best.method
    ]
    # The provider this cell actually used, not the worst of every configured
    # provider: reporting MiniLM's 256-token limit against a Qwen cell would
    # claim truncation that never happened.
    provider = _method_provider(best.method)
    profile = audit.get(best.variant, {})
    truncation = profile.get("over_limit", {}).get(provider)
    lead = ""
    if baseline:
        delta = best.scores[metric] - baseline[0].scores[metric]
        lead = (
            f" That is {delta:+.3f} {metric} against `base` on the same method."
            if abs(delta) >= 0.001
            else f" That is within noise of `base` on the same method, so the "
            f"current chunking is not measurably costing {metric}."
        )
    if truncation is None:
        note = " Truncation for that method's provider was not measured."
    elif truncation:
        note = (
            f" It truncates {truncation:.1%} of its inputs at {provider}'s "
            f"limit of {profile['limits'][provider]} tokens, so that score "
            "still describes a partly unread corpus."
        )
    else:
        note = f" Nothing is truncated at {provider}'s limit."
    sample = (
        f"{best.span_questions} anchored questions"
        if metric in SPAN_METRICS
        else f"{best.questions} questions"
    )
    return (
        f"Best measured cell: `{best.variant}` with `{best.method}` at "
        f"{metric} {best.scores[metric]:.3f} over {sample}.{lead}{note} "
        f"Compared variants: {', '.join(sorted(counts))}.{spread}{caveat}"
    )


def _retrieval_signature() -> dict:
    """Retrieval settings a cell's ranking depends on.

    Named keys rather than the whole config: hashing everything would invalidate a
    nemotron sweep because a DCI knob moved, and needless re-runs cost hours here.
    Anything that changes which chunks come back, or in what order, belongs in this
    list.
    """
    keys = (
        "final_result_limit",
        "rerank_candidate_limit",
        "reranker_model",
        "reranker_max_length",
        "reranker_prompt_name",
        "reranker_prompt",
        "hybrid_vector_weight",
        "hybrid_sparse_weight",
        "sparse_k1",
        "sparse_b",
        "agentic_judge_model",
        "agentic_max_attempts",
        "agentic_min_sufficient_chunks",
        "agentic_initial_limit",
        "agentic_limit_step",
        "agentic_max_limit",
    )
    return {key: RETRIEVAL_CONFIG.get(key) for key in keys}


def _load_state(checkpoint: Path, variants, methods, args) -> dict:
    signature = {
        "variants": variants,
        "methods": methods,
        "limit": args.limit,
        "category": args.category,
        "warmup": args.warmup,
        # Each of these changes what a stored cell means, so a resume that
        # ignored them would blend numbers measured against different targets.
        "design": args.design,
        "span_target": args.span_target if args.span_metrics else None,
        "judge_k": args.judge_k if args.judge_equivalence else None,
        "agentic_diagnostics": args.agentic_diagnostics,
        # The retrieval settings a cell's score depends on. Without these, halving
        # `rerank_candidate_limit` between runs would resume the cells measured at
        # the old value and put both in one table, which reads as a chunking
        # difference and is a reranker-depth difference.
        "retrieval": _retrieval_signature(),
    }
    if args.no_resume or not checkpoint.exists():
        return {"signature": signature, "runs": {}}
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    stored_signature = dict(stored.get("signature") or {})

    # Checkpoints written before retrieval settings were tracked carry no
    # `retrieval` key. Discarding them would throw away hours of completed cells
    # over a field that was not recorded, so they resume — but the one thing the
    # key exists to rule out cannot be ruled out for them, and that gets said out
    # loud rather than assumed away.
    untracked_retrieval = "retrieval" not in stored_signature
    if untracked_retrieval:
        stored_signature["retrieval"] = signature["retrieval"]

    if stored_signature != signature:
        changed = sorted(
            key
            for key in set(stored_signature) | set(signature)
            if stored_signature.get(key) != signature.get(key)
        )
        print(f"checkpoint signature changed ({', '.join(changed)}); starting fresh")
        return {"signature": signature, "runs": {}}

    runs = stored.get("runs", {})
    if untracked_retrieval and runs:
        print(
            f"warning: resuming {len(runs)} cells from a checkpoint that did not "
            "record retrieval settings. If rerank_candidate_limit, the reranker "
            "model, or the hybrid weights changed since those cells ran, this "
            "table will mix two configurations — rerun with --no-resume instead."
        )
    else:
        print(f"resuming {len(runs)} completed cells")
    return {"signature": signature, "runs": runs}


def _write_state(checkpoint: Path, state: dict) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _values(value: str | None) -> list[str]:
    return list(
        dict.fromkeys(part.strip() for part in (value or "").split(",") if part.strip())
    )


def _resolve_methods(value: str | None) -> list[str]:
    """Expand any named group in a comma-separated method list."""
    resolved: list[str] = []
    for name in _values(value):
        resolved.extend(METHOD_GROUPS.get(name, (name,)))
    return list(dict.fromkeys(resolved))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated method names, or a named group: "
            + ", ".join(f"`{name}`" for name in METHOD_GROUPS)
            + ". Groups and bare names can be mixed."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--warmup", type=int, default=EXPERIMENT_CONFIG.get("default_warmup", 5)
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help=(
            "Do not substitute a longer sequence length for variants that "
            "overflow the requested one. Use this to measure truncation "
            "deliberately, e.g. --methods qwen,qwen_s2048 --variants base."
        ),
    )
    parser.add_argument(
        "--span-metrics",
        action="store_true",
        help=(
            "Add character-overlap metrics. Only meaningful when every variant "
            "is scored on the shared base question set; with per-variant "
            "questions each variant's target is its own chunk, which is a "
            "tautology."
        ),
    )
    parser.add_argument(
        "--span-target",
        choices=("gold", "anchor"),
        default="gold",
        help=(
            "What the character-overlap metrics measure against. `gold` is the "
            "base chunk the question came from: free, complete, and it makes "
            "`base` score a trivial 1.000, so base is the ruler rather than a "
            "competitor. `anchor` is the quoted sentence that states the answer: "
            "tighter and it lets base compete, but it only covers anchored "
            "questions (`uv run python -m src.eval.anchors`)."
        ),
    )
    parser.add_argument(
        "--design",
        choices=DESIGNS,
        default="shared",
        help=(
            "Which labelling design the database holds. Declared, not detected, "
            "and validated before any cell runs: the designs are "
            "indistinguishable from a single cell, and a grid that mixes them "
            "publishes rows that are not comparable. `per-variant-density` is "
            "`per-variant` with each chunk's question count scaled to its size "
            "(`generate_dataset.py --density`), so the columns match in size."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned grid and an estimated cost, then exit.",
    )
    parser.add_argument(
        "--judge-equivalence",
        action="store_true",
        help=(
            "Add judge_hit@K, crediting a retrieved set the local equivalence "
            "judge rules can replace the gold evidence. One LLM call per missed "
            "question per cell; judgments are cached across cells and runs."
        ),
    )
    parser.add_argument(
        "--judge-k",
        type=int,
        default=int(EVAL_CONFIG["result_limit"]),
        help="Retrieval cutoff for equivalence judging (default: eval result_limit).",
    )
    parser.add_argument("--judge-cache", default=None)
    parser.add_argument(
        "--agentic-diagnostics",
        action="store_true",
        help=(
            "Add the paired agent-versus-baseline table. Runs each agent's "
            "non-agentic baseline in the same cell, so it costs one extra "
            "baseline measurement per variant."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.judge_cache is None:
        args.judge_cache = f"{args.output}.equivalence.json"
    if args.judge_k < 1:
        raise ValueError("--judge-k must be at least 1")
    return args


if __name__ == "__main__":
    run_cli(main)
