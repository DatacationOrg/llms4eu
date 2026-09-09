"""Compare retrieval methods across chunk variants, with a live report and resume.

The comprehensive comparison driver. A *cell* is one (variant, method) pair; every
cell keeps per-question rankings, action logs and observations in the checkpoint,
so the report can carry the elaborate diagnostics (paired agentic diagnostics,
`judge_hit@K`, per-question sidecars) that the aggregate-only chunk sweep cannot,
while also reporting the sweep's span and cost metrics (`budget_recall@4000`,
`char_recall@10`, `store_share@10`, `recall_per_share@10`).

Axes, all selectable in one run:

- chunk variants (`--variants base,tok256,...`), scored on the question set the
  database's `--design` declares;
- retrieval methods, mixing named groups and bare names (`--methods
  full`, `--methods agents,geo`, `--methods qwen_hybrid_rerank,...`): embedders,
  pipeline rungs, geo-scoped methods, agents with and
  without page tools, DCI, judge LLMs (`_gptoss`, `_gemma`) and reasoning rungs;
- an optional evidence-equivalence judge adding `judge_hit@K`.

Methods run grouped by embedding provider, one provider resident at a time, so a
grid over five embedders never holds two 15 GB checkpoints on the card together.
Within a group the methods interleave per question, so their timings see the same
machine. The checkpoint is written every `--save-every` queries and the report
after every query, so stopping the run at any point costs at most one query.

`--dry-run` checks the database against the requested design, span target and
geo methods, lists every missing index with its build command, and prices the
grid before anything is loaded.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import sys
import time
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Any

from _cli import run_cli
from src.db.pages import connect_pages, initialize_page_artifacts_db
from src.eval.agentic_diagnostics import (
    build_agentic_diagnostics,
    format_agentic_diagnostics,
)
from src.eval.equivalence import EvidenceEquivalenceJudge
from src.eval.evaluate import CONFIG as EVAL_CONFIG
from src.eval.evaluate import (
    EvalRun,
    add_judge_adjusted_scores,
    add_span_scores,
    add_store_share_scores,
    count_chunk_expansions,
    format_eval_report,
    load_eval_rows,
    load_store_profile,
    score_eval_rankings,
)
from src.eval.metrics import bold_best_table, plain_table, score_rankings
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    list_retrievers,
    method_provider,
    method_requires_geo,
    missing_retriever_indexes,
)
from src.shared.env import ROOT, load_local_env, load_yaml

CONFIG = load_yaml(ROOT / "experiments" / "indexing" / "config.yaml")
RETRIEVAL_CONFIG = load_yaml(ROOT / "src" / "retrieval" / "config.yaml")
DEFAULT_OUTPUT = Path("docs/reports/retrieval/retrieval-results-full.md")
STATE_VERSION = 4
DESIGNS = ("shared", "per-variant", "per-variant-density")
SPAN_TARGETS = ("anchor", "gold")

# The embedders the sweep store holds. `qwen8b` is the local stand-in for the
# removed Azure `embed-v-4-0`; `nemotron*` were not trained on Slovenian, which is
# the point of keeping them in the grid.
EMBEDDERS = ("qwen", "qwen4b", "qwen8b", "nemotron", "nemotron8b")
AGENT_PROVIDERS = ("qwen", "qwen8b", "nemotron")


def _per(providers, *templates):
    return tuple(t.format(p=p) for p in providers for t in templates)


# Named method sets. Groups and bare names mix freely in `--methods`, and every
# agent group carries its own reranked baseline so the paired diagnostics can
# pair. `full` is the union: the whole grid in one resumable run.
METHOD_GROUPS: dict[str, tuple[str, ...]] = {
    # Bare vectors: the only rung where the embedder is the sole variable.
    "embedders": EMBEDDERS,
    # Every embedder on the published pipeline, plus the lexical floor.
    "pipelines": (
        "sparse",
        "sparse_rerank",
        *_per(EMBEDDERS, "{p}_hybrid", "{p}_hybrid_rerank"),
    ),
    "reranker-ladder": ("qwen8b_hybrid_rerank", "qwen8b_hybrid_rerank_4b"),
    # Geo-aware first stage: soft (over-fetch, fuse text and geography; unknown
    # footprint neutral) with and without the reranker, plus the strict
    # filter-then-widen shape for comparison.
    "geo": _per(
        AGENT_PROVIDERS,
        "{p}_hybrid_rerank",
        "{p}_hybrid_geo",
        "{p}_hybrid_rerank_geo",
        "{p}_hybrid_rerank_geo_strict",
    ),
    # Agent generations on the default judge, each beside its baseline.
    "agents": (
        *_per(
            AGENT_PROVIDERS,
            "{p}_hybrid_rerank",
            "{p}_hybrid_agentic",
            "{p}_hybrid_agentic_tools",
        ),
        "sparse",
        "dci",
        "dci_k50",
    ),
    # The same agents on the geo-scoped first stage; the tool agent also gets the
    # geo tools (find_pages_near, pages_in_region).
    "agents-geo": _per(
        ("qwen", "nemotron"),
        "{p}_hybrid_rerank_geo",
        "{p}_hybrid_agentic_geo",
        "{p}_hybrid_agentic_tools_geo",
    ),
    # The judge LLM as an axis: the unsuffixed agents run the default judge
    # (azure / DeepSeek); these run the local judges from `agentic_judges`.
    "judges": (
        "qwen_hybrid_rerank",
        "qwen_hybrid_agentic",
        "qwen_hybrid_agentic_gptoss",
        "qwen_hybrid_agentic_gemma",
        "qwen_hybrid_agentic_tools",
        "qwen_hybrid_agentic_tools_gptoss",
        "qwen_hybrid_agentic_tools_gemma",
        "dci",
        "dci_gptoss",
    ),
    # Reasoning effort, gpt-oss only in practice (gemma's `think` is boolean).
    "reasoning": (
        "qwen_hybrid_rerank",
        "qwen_hybrid_agentic",
        "qwen_hybrid_agentic_high",
        "qwen_hybrid_agentic_tools",
        "qwen_hybrid_agentic_tools_high",
        "dci",
        "dci_high",
    ),
    # Kept so the historical reports' method lists still resolve.
    "legacy-default": (
        "sparse_rerank",
        "qwen4b_hybrid_rerank",
        "nemotron",
        "nemotron_hybrid_rerank",
        "qwen_hybrid_agentic",
        "nemotron_hybrid_agentic",
        "qwen_hybrid_agentic_tools",
    ),
}
METHOD_GROUPS["full"] = tuple(
    dict.fromkeys(
        name
        for group in (
            "embedders",
            "pipelines",
            "reranker-ladder",
            "geo",
            "agents",
            "agents-geo",
            "judges",
            "reasoning",
        )
        for name in METHOD_GROUPS[group]
    )
)
DEFAULT_METHODS = METHOD_GROUPS["legacy-default"]

# Reported across variants, in this order; anything a method does not produce
# shows as "-". Span metrics lead because `hit@k` rewards a variant for cutting
# large; cost metrics follow because every quality metric can be bought with size.
SPAN_METRICS = ("budget_recall@4000", "char_recall@10", "char_precision@10")
COST_METRICS = ("store_share@10", "recall_per_share@10")
CHUNK_METRICS = ("hit@1", "hit@5", "hit@10", "recall@10", "mrr@10")
LOWER_IS_BETTER = ("store_share@1", "store_share@5", "store_share@10")

# Cost model, from the 2026-08-14 sweep (within 20% on every reranked cell) and
# the 2026-09-01 agentic runs. Reranking is priced per median chunk token; the
# geo resolver is one cached LLM call per distinct question, so it is charged to
# the first geo method per variant only.
SECONDS_PER_RERANK_TOKEN = 0.001324
SECONDS_PER_RERANK_TOKEN_BY_RERANKER = {"4b": 0.00816}
SECONDS_PER_QUERY = {"agentic": 5.1, "dci": 8.0, "plain": 0.006, "geo": 0.05}
JUDGE_SECONDS_PER_QUERY = 2.0
JUDGED_MISS_RATE = 0.2
GEO_RESOLVER_SECONDS_PER_QUERY = 1.5
# Slovenian text under the Qwen tokenizer, from the chunk token audit.
TOKENS_PER_CHAR = 0.41

# Shared-design databases label one question set in every variant.
SHARED_QUESTION_SHARE = 0.5
SAMPLE_SPREAD = 0.01


def main() -> None:
    load_local_env()
    args = _parse_args()
    method_names = _resolve_methods(args.methods)
    variants = _resolve_variants(args.variants)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else _default_checkpoint_path(output_path)
    )
    judge_cache_root = (
        Path(args.judge_cache)
        if args.judge_cache
        else output_path.with_suffix(output_path.suffix + ".equivalence.json")
    )
    warmup = _resolve_warmup(args.warmup, args.limit)
    command = " ".join(shlex.quote(part) for part in sys.argv)

    initialize_page_artifacts_db()
    _check_design(variants, args.design)
    _check_prerequisites(method_names, variants, args)

    if args.dry_run:
        print(_dry_run_report(method_names, variants, args, command))
        return

    runs, state = _run_grid(
        method_names=method_names,
        variants=variants,
        args=args,
        warmup=warmup,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        judge_cache_root=judge_cache_root,
        command=command,
    )
    report = _build_report(
        state=state,
        runs=runs,
        variants=variants,
        method_names=method_names,
        args=args,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        command=command,
        status="complete",
    )
    print(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    _write_cell_rows(output_path, runs, args)
    _save_action_logs_from_state(state, output_path)
    _save_agentic_diagnostics(runs, state, output_path)
    if args.discard_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare retrieval methods across chunk variants with a live report "
            "and checkpoint resume."
        ),
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated method names and/or groups: "
            + ", ".join(METHOD_GROUPS)
            + ". `full` is the whole grid."
        ),
    )
    parser.add_argument(
        "--variants",
        default=BASE_CHUNK_VARIANT,
        help=(
            "Comma-separated chunk variants, or `all` for every variant in the "
            "database (default: base). Runs against PAGES_DB_PATH / CHROMA_PATH."
        ),
    )
    parser.add_argument(
        "--design",
        choices=DESIGNS,
        default="shared",
        help=(
            "Which question-set design the database holds; checked before the "
            "first cell. shared: one set relabelled onto every variant."
        ),
    )
    parser.add_argument(
        "--span-target",
        choices=SPAN_TARGETS,
        default="anchor",
        help=(
            "Character-overlap target: the anchored answer quote (needs "
            "eval_answer_anchors) or the gold base chunk."
        ),
    )
    parser.add_argument(
        "--no-span-metrics",
        action="store_true",
        help="Skip char_recall / budget_recall (store_share still reported).",
    )
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=(
            "Warmup questions per cell, excluded from timing. Defaults to 0 with "
            "--limit and to experiments/indexing/config.yaml:default_warmup otherwise."
        ),
    )
    parser.add_argument("--output", help=f"Report path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument(
        "--checkpoint", help="Checkpoint path (default: <output>.checkpoint.json)."
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--report-every",
        type=float,
        default=30.0,
        help="Seconds between live report rewrites (default 30; 0 = every query).",
    )
    parser.add_argument(
        "--discard-checkpoint",
        action="store_true",
        help=(
            "Delete the checkpoint when the run completes. Kept by default: it is "
            "the input to the post-hoc equivalence audit and to render_qwen_report."
        ),
    )
    parser.add_argument(
        "--catch-up-only",
        action="store_true",
        help=(
            "Run newly added cells only up to each variant's existing frontier, "
            "without advancing cells already in the checkpoint."
        ),
    )
    parser.add_argument(
        "--judge-equivalence",
        action="store_true",
        help="Judge strict misses for evidence equivalence and add judge_hit@K.",
    )
    parser.add_argument("--judge-k", type=int, default=int(EVAL_CONFIG["result_limit"]))
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Ollama model for the equivalence judge (default: config).",
    )
    parser.add_argument("--judge-cache")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check prerequisites and price the grid; run nothing.",
    )
    return parser.parse_args()


# --- resolution and checks -----------------------------------------------------


def _resolve_methods(raw: str) -> list[str]:
    names: list[str] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        names.extend(METHOD_GROUPS.get(token, (token,)))
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("No methods provided.")
    unknown = sorted(set(names) - set(list_retrievers()))
    if unknown:
        raise ValueError(
            f"Unknown methods: {', '.join(unknown)}. "
            f"Groups: {', '.join(METHOD_GROUPS)}."
        )
    return names


def _resolve_variants(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        with connect_pages() as conn:
            return [
                row["variant"]
                for row in conn.execute(
                    "select distinct variant from page_chunks order by variant"
                )
            ]
    variants = list(dict.fromkeys(v.strip() for v in raw.split(",") if v.strip()))
    if not variants:
        raise ValueError("No variants provided.")
    return variants


def _resolve_warmup(requested: int | None, limit: int | None) -> int:
    if requested is not None:
        return requested
    if limit is not None:
        return 0
    return int(CONFIG["default_warmup"])


def _default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.checkpoint.json")


def _check_design(variants: list[str], design: str) -> None:
    """Fail before the first cell when the database does not hold `design`.

    The primary observable is whether the same question is labelled in more than
    one variant: a shared set is, a per-variant set structurally cannot be.
    Sample size separates the two per-variant designs.
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
            f"'{design}' design in {_db_label()}. Label them first: `{remedy}`. "
            f"Counts: {counts}"
        )
    if len(variants) < 2:
        return
    union = set().union(*labelled.values())
    shared_share = sum(
        1 for qid in union if sum(qid in ids for ids in labelled.values()) > 1
    ) / len(union)
    sizes = sorted(set(counts.values()))
    even = sizes[-1] - sizes[0] <= SAMPLE_SPREAD * sizes[-1]
    if design == "shared" and shared_share < SHARED_QUESTION_SHARE:
        raise RuntimeError(
            "--design shared expects one question set labelled in every variant, "
            f"but only {shared_share:.1%} of questions appear in more than one. "
            "Either relabel (`uv run python -m src.eval.relabel --variant <name>`) "
            f"or declare the design this database holds. Counts: {counts}"
        )
    if design != "shared" and shared_share >= SHARED_QUESTION_SHARE:
        raise RuntimeError(
            f"--design {design} expects each variant to own its questions, but "
            f"{shared_share:.1%} are labelled in more than one variant; use "
            "--design shared."
        )
    if design == "per-variant" and even:
        raise RuntimeError(
            "--design per-variant expects differently sized sets, but the counts "
            f"match: {counts}. Use --design per-variant-density."
        )
    if design == "per-variant-density" and not even:
        raise RuntimeError(
            "--design per-variant-density expects matching counts, but they differ: "
            f"{counts}. Use --design per-variant, or finish generation."
        )


def _check_prerequisites(method_names: list[str], variants: list[str], args) -> None:
    """Data the requested cells depend on, checked before any model loads."""
    problems = []
    if not args.no_span_metrics and args.span_target == "anchor":
        if _anchor_count() == 0:
            problems.append(
                f"{_db_label()} has no eval_answer_anchors; span metrics need "
                "`just anchor-answers` first, or pass --span-target gold / "
                "--no-span-metrics."
            )
    if any(method_requires_geo(name, variants[0]) for name in method_names):
        if _location_count() == 0:
            problems.append(
                f"{_db_label()} has no page_locations; geo methods need "
                "`just locate-pages --apply` or `python -m src.preprocess.locations "
                "--copy-from data/db/pages.db` against this database first."
            )
    for problem in problems:
        print(f"warning: {problem}", flush=True)
    if problems and not args.dry_run:
        raise RuntimeError("Prerequisites missing; see warnings above.")


def _anchor_count() -> int:
    with connect_pages() as conn:
        return int(
            conn.execute("select count(*) from eval_answer_anchors").fetchone()[0]
        )


def _location_count() -> int:
    with connect_pages() as conn:
        return int(conn.execute("select count(*) from page_locations").fetchone()[0])


def _db_label() -> str:
    return os.getenv("PAGES_DB_PATH", "data/db/pages.db")


# --- dry run -------------------------------------------------------------------


def _dry_run_report(method_names, variants, args, command: str) -> str:
    counts = {
        v: len(load_eval_rows(limit=args.limit, category=args.category, variant=v)[0])
        for v in variants
    }
    medians = {v: _median_tokens(v) for v in variants}
    missing: dict[str, dict[str, str]] = {}
    for variant in variants:
        missing[variant] = missing_retriever_indexes(method_names, variant)
    lines = [
        f"# Dry run {date.today().isoformat()}",
        "",
        f"- Command: `{command}`",
        f"- Database: {_db_label()} | Chroma: {os.getenv('CHROMA_PATH', 'data/cache/chroma')}",
        f"- Variants: {', '.join(variants)} | design: {args.design} | "
        f"span target: {args.span_target}",
        f"- Methods ({len(method_names)}): {', '.join(method_names)}",
        f"- Cells: {len(variants) * len(method_names)}",
        "",
        "## Provider groups (loaded one at a time)",
        "",
    ]
    for provider, names in _provider_groups(method_names).items():
        lines.append(f"- {provider or 'no embedder'}: {', '.join(names)}")
    lines.extend(
        [
            "",
            "## Cells",
            "",
            "| variant | method | questions | kind | median tok | estimated | status |",
            "|---|---|---:|---|---:|---:|---|",
        ]
    )
    total = blocked = 0.0
    geo_charged: set[str] = set()
    for variant in variants:
        for method in method_names:
            questions = counts[variant]
            seconds = questions * _seconds_per_query(method, medians[variant])
            if args.judge_equivalence:
                seconds += questions * JUDGED_MISS_RATE * JUDGE_SECONDS_PER_QUERY
            if method_requires_geo(method, variant) and variant not in geo_charged:
                seconds += questions * GEO_RESOLVER_SECONDS_PER_QUERY
                geo_charged.add(variant)
            status = missing[variant].get(method)
            if status is None:
                total += seconds
                label = "ready"
            else:
                blocked += seconds
                label = f"needs index `{_build_command(status)}`"
            lines.append(
                f"| {variant} | {method} | {questions} | {_cost_kind(method)} | "
                f"{medians[variant]} | {_duration(seconds)} | {label} |"
            )
    lines.extend(["", f"**Estimated total, runnable now: {_duration(total)}**"])
    if blocked:
        lines.append(
            f"**Plus {_duration(blocked)} once the missing indexes are built** "
            f"(total {_duration(total + blocked)}). Geo methods also need the "
            "collection rebuilt with geo metadata (`just eval-index <provider> "
            "<version> <variant>` after `page_locations` is filled)."
        )
    lines.extend(
        [
            "",
            "Reranking is priced per median chunk token "
            f"({SECONDS_PER_RERANK_TOKEN} s/token; {SECONDS_PER_RERANK_TOKEN_BY_RERANKER}), "
            f"agents flat at {SECONDS_PER_QUERY['agentic']} s/query, DCI at "
            f"{SECONDS_PER_QUERY['dci']} s/query, the geo resolver once per question "
            f"per variant at {GEO_RESOLVER_SECONDS_PER_QUERY} s. An estimate: the run "
            "checkpoints every --save-every queries, so stopping costs one query.",
        ]
    )
    return "\n".join(lines)


def _build_command(requirement: str) -> str:
    head, _, variant = requirement.partition("#")
    provider, _, version = head.partition("@")
    return f"just eval-index {provider} {version or 'v1'} {variant or 'base'}"


def _cost_kind(method: str) -> str:
    if method.startswith("dci"):
        return "dci"
    if "agentic" in method:
        return "agentic"
    if "rerank" in method:
        return "rerank"
    return "geo" if "_geo" in method else "plain"


def _method_reranker(method: str) -> str | None:
    for name in sorted(RETRIEVAL_CONFIG.get("rerankers") or {}, key=len, reverse=True):
        if method.endswith(f"_rerank_{name}") or f"_rerank_{name}_" in method:
            return name
    return None


def _seconds_per_query(method: str, median_tokens: int) -> float:
    kind = _cost_kind(method)
    if kind == "rerank":
        rate = SECONDS_PER_RERANK_TOKEN_BY_RERANKER.get(
            _method_reranker(method), SECONDS_PER_RERANK_TOKEN
        )
        seconds = min(median_tokens, RETRIEVAL_CONFIG["reranker_max_length"]) * rate
        return seconds + (SECONDS_PER_QUERY["geo"] if "_geo" in method else 0.0)
    return SECONDS_PER_QUERY[kind]


def _median_tokens(variant: str) -> int:
    with connect_pages() as conn:
        counts = [
            int(row["char_count"])
            for row in conn.execute(
                "select char_count from page_chunks where variant = ?", (variant,)
            )
        ]
    return int(median(counts) * TOKENS_PER_CHAR) if counts else 0


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# --- provider grouping and model release ------------------------------------------


def _provider_groups(method_names: list[str]) -> dict[str | None, list[str]]:
    """Methods by embedding provider, cheapest group first, config order after."""
    groups: dict[str | None, list[str]] = {}
    for name in method_names:
        groups.setdefault(method_provider(name), []).append(name)
    order = [None, *[p for p in EMBEDDERS if p in groups]]
    order += [p for p in groups if p not in order]
    return {provider: groups[provider] for provider in order if provider in groups}


def _release_models() -> None:
    """Drop the memoised embedders so the next provider group starts on a free card.

    `EmbeddingIndexer._model` is a `functools.cache` on a method, which holds the
    indexer instance and its loaded model for the life of the process; clearing
    it is the only way a second 15 GB embedder fits.
    """
    import src.shared.indexers as indexers

    for attribute in vars(indexers).values():
        if isinstance(attribute, type):
            for member in vars(attribute).values():
                if hasattr(member, "cache_clear"):
                    member.cache_clear()
        elif hasattr(attribute, "cache_clear"):
            attribute.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --- the grid ------------------------------------------------------------------


def _cell_key(variant: str, method: str) -> str:
    return f"{variant}|{method}"


def _empty_cell(variant: str, method: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "method": method,
        "warmup_index": 0,
        "next_index": 0,
        "elapsed_seconds": 0.0,
        "timed_total_queries": 0.0,
        "rankings": {},
        "action_log": [],
        "observations": {},
        "failures": 0,
        "geo": {"resolved": 0, "unresolved": 0, "widenings": 0, "levels": {}},
    }


def _retrieval_signature() -> dict:
    """Retrieval settings a cell's ranking depends on.

    Named keys rather than the whole config, so a DCI knob does not invalidate a
    nemotron cell. Includes the judge provider, model and structured-output
    settings: under `azure` the model name is the deployment in the environment,
    so that is recorded too, or a judge swap would resume the other judge's cells.
    """
    keys = (
        "final_result_limit",
        "rerank_candidate_limit",
        "reranker_model",
        "reranker_max_length",
        "reranker_prompt_name",
        "reranker_prompt",
        "reranker_dtype",
        "rerankers",
        "hybrid_vector_weight",
        "hybrid_sparse_weight",
        "sparse_k1",
        "sparse_b",
        "agentic_judge_provider",
        "agentic_judge_model",
        "agentic_judge_reasoning",
        "agentic_judge_structured_method",
        "agentic_tools_structured_method",
        "dci_structured_method",
        "agentic_judges",
        "agentic_reasoning_levels",
        "agentic_max_attempts",
        "agentic_min_sufficient_chunks",
        "agentic_initial_limit",
        "agentic_limit_step",
        "agentic_max_limit",
        "agentic_tools_max_tool_calls",
        "geo_resolver_provider",
        "geo_default_radius_km",
        "geo_min_candidates",
        "geo_boost_weight",
        "geo_decay_km",
        "geo_include_null",
    )
    signature = {key: RETRIEVAL_CONFIG.get(key) for key in keys}
    signature["azure_model"] = os.getenv("AZURE_AI_MODEL")
    return signature


def _run_grid(
    *,
    method_names: list[str],
    variants: list[str],
    args,
    warmup: int,
    checkpoint_path: Path,
    output_path: Path,
    judge_cache_root: Path,
    command: str,
) -> tuple[dict[str, EvalRun], dict[str, Any]]:
    questions_by_variant: dict[str, tuple[list[dict], list[dict]]] = {}
    signature_questions = {}
    for variant in variants:
        questions, relevance = load_eval_rows(
            limit=args.limit, category=args.category, variant=variant
        )
        if not questions:
            raise RuntimeError(f"No eval questions for variant {variant}")
        cut = min(warmup, max(len(questions) - 1, 0))
        questions_by_variant[variant] = (questions, relevance)
        signature_questions[variant] = {
            "warmup_ids": [row["id"] for row in questions[:cut]],
            "timed_ids": [row["id"] for row in questions[cut:]],
        }
    signature = {
        "version": STATE_VERSION,
        "variants": variants,
        "method_names": method_names,
        "category": args.category,
        "limit": args.limit,
        "warmup": warmup,
        "design": args.design,
        "span_target": None if args.no_span_metrics else args.span_target,
        "result_limit": EVAL_CONFIG["result_limit"],
        "questions": signature_questions,
        "retrieval": _retrieval_signature(),
    }
    catch_up = (
        _catch_up_plan(checkpoint_path, variants, method_names)
        if args.catch_up_only and not args.no_resume
        else None
    )
    if args.catch_up_only and catch_up is None:
        raise ValueError("--catch-up-only requires a compatible existing checkpoint")
    state = _load_or_initialize_state(
        checkpoint_path, signature, variants, method_names, resume=not args.no_resume
    )
    audits = {
        variant: _build_equivalence_audit(
            state, variant, method_names, args, judge_cache_root
        )
        for variant in variants
    }
    reporter = _LiveReporter(
        state=state,
        variants=variants,
        method_names=method_names,
        args=args,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        command=command,
        questions_by_variant=questions_by_variant,
        audits=audits,
        every=args.report_every,
    )
    reporter.write(force=True)

    for variant in variants:
        questions, relevance = questions_by_variant[variant]
        cut = len(signature_questions[variant]["warmup_ids"])
        warmup_questions, timed_questions = questions[:cut], questions[cut:]
        for provider, names in _provider_groups(method_names).items():
            pending = [
                name
                for name in names
                if state["cells"][_cell_key(variant, name)]["next_index"]
                < (catch_up or {}).get(_cell_key(variant, name), len(timed_questions))
                or state["cells"][_cell_key(variant, name)]["warmup_index"] < cut
            ]
            if not pending:
                continue
            missing = missing_retriever_indexes(pending, variant)
            if missing:
                raise RuntimeError(
                    f"Missing indexes for variant {variant}: "
                    + ", ".join(
                        f"{m} ({_build_command(r)})" for m, r in missing.items()
                    )
                )
            print(
                f"== {variant} | {provider or 'no embedder'}: {', '.join(pending)}",
                flush=True,
            )
            retrievers = {name: build_retriever(name, variant) for name in pending}
            for name in pending:
                _run_warmup(
                    state["cells"][_cell_key(variant, name)],
                    retrievers[name],
                    warmup_questions,
                    checkpoint_path,
                    state,
                )
            _run_timed_round_robin(
                variant=variant,
                method_names=pending,
                retrievers=retrievers,
                timed_questions=timed_questions,
                state=state,
                checkpoint_path=checkpoint_path,
                save_every=max(1, args.save_every),
                log_every=max(1, args.log_every),
                targets=catch_up,
                audit=audits[variant],
                reporter=reporter,
                runs_writer=lambda: _write_cell_rows(
                    output_path,
                    _all_runs(
                        state,
                        variants,
                        method_names,
                        questions_by_variant,
                        args,
                        audits,
                    ),
                    args,
                ),
            )
            del retrievers
            _release_models()
    _write_state(checkpoint_path, state)
    runs = _all_runs(state, variants, method_names, questions_by_variant, args, audits)
    return runs, state


def _run_warmup(cell, retriever, warmup_questions, checkpoint_path, state) -> None:
    for index in range(int(cell["warmup_index"]), len(warmup_questions)):
        retriever.retrieve(
            warmup_questions[index]["question"], EVAL_CONFIG["result_limit"]
        )
        cell["warmup_index"] = index + 1
        _write_state(checkpoint_path, state)


def _run_timed_round_robin(
    *,
    variant: str,
    method_names: list[str],
    retrievers: dict[str, Retriever],
    timed_questions: list[dict[str, Any]],
    state: dict[str, Any],
    checkpoint_path: Path,
    save_every: int,
    log_every: int,
    targets: dict[str, int] | None,
    audit,
    reporter,
    runs_writer,
) -> None:
    total = len(timed_questions)
    target_of = {
        name: (targets or {}).get(_cell_key(variant, name), total)
        for name in method_names
    }
    cells = {name: state["cells"][_cell_key(variant, name)] for name in method_names}
    while any(cells[name]["next_index"] < target_of[name] for name in method_names):
        for name in method_names:
            cell = cells[name]
            index = int(cell["next_index"])
            if index >= target_of[name]:
                continue
            retriever = retrievers[name]
            row = timed_questions[index]
            should_log = (index + 1) % log_every == 0
            if should_log:
                print(f"{variant}|{name}: running {index + 1}/{total}", flush=True)
            before_queries = _read_total_queries(retriever)
            before_failures = int(getattr(retriever, "failures", 0))
            before_geo = _geo_snapshot(retriever)
            action_log = _retriever_action_log(retriever)
            prev_len = len(action_log) if action_log is not None else 0
            started = time.perf_counter()
            chunks = retriever.retrieve(row["question"], EVAL_CONFIG["result_limit"])
            elapsed = time.perf_counter() - started
            new_entries = []
            if action_log is not None:
                new_entries = [dict(entry) for entry in action_log[prev_len:]]
                for entry in new_entries:
                    entry["question_id"] = str(row["id"])
                cell.setdefault("action_log", []).extend(new_entries)
            queries = _query_delta(before_queries, _read_total_queries(retriever))
            cell["elapsed_seconds"] += elapsed
            cell["timed_total_queries"] += queries
            cell["failures"] = int(cell.get("failures", 0)) + max(
                int(getattr(retriever, "failures", 0)) - before_failures, 0
            )
            cell["rankings"][str(row["id"])] = [chunk.id for chunk in chunks]
            cell.setdefault("observations", {})[str(row["id"])] = {
                "elapsed_seconds": elapsed,
                "query_count": queries,
                "actions": new_entries,
            }
            _accumulate_geo(cell, before_geo, _geo_snapshot(retriever))
            cell["next_index"] = index + 1
            if audit is not None:
                audit.judge_prediction(
                    name, str(row["id"]), cell["rankings"][str(row["id"])]
                )
            reporter.write()
            if should_log:
                print(
                    f"{variant}|{name}: completed {index + 1}/{total} in {elapsed:.2f}s "
                    f"({len(chunks)} results)",
                    flush=True,
                )
            if (index + 1) % save_every == 0 or index + 1 == total:
                _write_state(checkpoint_path, state)
                runs_writer()


def _all_runs(
    state, variants, method_names, questions_by_variant, args, audits
) -> dict[str, EvalRun]:
    return {
        variant: run
        for variant in variants
        if (
            run := _variant_run(
                state,
                variant,
                method_names,
                *questions_by_variant[variant],
                args,
                audits.get(variant),
            )
        )
        is not None
    }


def _variant_run(
    state, variant, method_names, questions, relevance, args, audit
) -> EvalRun | None:
    """Score one variant on the questions every one of its cells has answered."""
    cut = len(state["signature"]["questions"][variant]["warmup_ids"])
    timed = questions[cut:]
    cells = {
        name: state["cells"].get(_cell_key(variant, name)) for name in method_names
    }
    started = [name for name, cell in cells.items() if cell and cell["next_index"] > 0]
    if not started:
        return None
    aligned = min(int(cells[name]["next_index"]) for name in started)
    scored = timed[:aligned]
    scored_ids = {str(row["id"]) for row in scored}
    rankings = {
        name: {
            qid: ranked
            for qid, ranked in cells[name]["rankings"].items()
            if qid in scored_ids
        }
        for name in started
    }
    timed_relevance = [
        row for row in relevance if str(row["question_id"]) in scored_ids
    ]
    timings = {}
    for name in started:
        cell = cells[name]
        elapsed = float(cell["elapsed_seconds"])
        total_queries = float(cell.get("timed_total_queries", 0.0))
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / aligned if aligned else 0.0,
            "queries_per_query": total_queries / aligned if aligned else 0.0,
            "total_queries": total_queries,
            "chunk_expansions": count_chunk_expansions(cell.get("action_log", [])),
        }
    score_names, scores = score_eval_rankings(timed_relevance, rankings, started)
    metric_names, category_scores = _score_categories(
        scored, timed_relevance, rankings, started
    )
    run = EvalRun(
        questions=scored,
        relevance=timed_relevance,
        methods=started,
        warmup_count=cut,
        rankings=rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
        category_metric_names=metric_names,
        category_scores=category_scores,
    )
    if not args.no_span_metrics:
        run = add_span_scores(run, variant, args.span_target)
    run = add_store_share_scores(run, variant)
    if audit is not None:
        summaries = audit.summaries(started, [str(row["id"]) for row in scored])
        run = add_judge_adjusted_scores(run, summaries, audit.cutoff)
    return run


def _score_categories(questions, relevance, rankings, method_names):
    type_by_id = {str(row["id"]): row["question_type"] for row in questions}
    types = sorted(set(type_by_id.values()))
    cutoff = int(EVAL_CONFIG["category_hit_k"])
    metric_names = {name: f"hit@{cutoff}" for name in method_names}
    scores: dict[str, dict[str, float]] = {}
    for name in method_names:
        scores[name] = {}
        for question_type in types:
            ids = {qid for qid, t in type_by_id.items() if t == question_type}
            scores[name][question_type] = score_rankings(
                [row for row in relevance if str(row["question_id"]) in ids],
                {qid: r for qid, r in rankings[name].items() if qid in ids},
                ks=(cutoff,),
            )[f"hit@{cutoff}"]
    return metric_names, scores


# --- state -----------------------------------------------------------------------


def _load_or_initialize_state(
    checkpoint_path, signature, variants, method_names, resume
) -> dict:
    if resume and checkpoint_path.exists():
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = None
            print("Checkpoint file is corrupted; starting fresh.")
        if state is not None:
            if state.get("signature") == signature:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                return state
            if _extend_compatible_state(state, signature, variants, method_names):
                print(f"Extending compatible checkpoint: {checkpoint_path}")
                _write_state(checkpoint_path, state)
                return state
            print("Ignoring incompatible checkpoint and starting fresh.")
    state = {
        "signature": signature,
        "cells": {
            _cell_key(v, m): _empty_cell(v, m) for v in variants for m in method_names
        },
    }
    _write_state(checkpoint_path, state)
    return state


def _extend_compatible_state(state, signature, variants, method_names) -> bool:
    """Accept a checkpoint that differs only by added variants or methods.

    Anything that changes what a cell measures (question ids, warmup, design,
    span target, retrieval settings) must match; a changed retrieval fingerprint
    means the old cells measured a different pipeline and are not extended.
    """
    previous = state.get("signature", {})
    stable = (
        "version",
        "category",
        "limit",
        "warmup",
        "design",
        "span_target",
        "result_limit",
        "retrieval",
    )
    if any(previous.get(key) != signature.get(key) for key in stable):
        changed = [k for k in stable if previous.get(k) != signature.get(k)]
        print(f"checkpoint differs in: {', '.join(changed)}")
        return False
    for variant, ids in previous.get("questions", {}).items():
        if variant in signature["questions"] and signature["questions"][variant] != ids:
            print(f"checkpoint question set for {variant} differs")
            return False
    existing = set(state.get("cells", {}))
    requested = {_cell_key(v, m) for v in variants for m in method_names}
    if not existing or not existing.issubset(requested):
        return False
    for variant in variants:
        for method in method_names:
            state["cells"].setdefault(
                _cell_key(variant, method), _empty_cell(variant, method)
            )
    state["signature"] = signature
    return True


def _catch_up_plan(checkpoint_path, variants, method_names) -> dict[str, int] | None:
    if not checkpoint_path.exists():
        return None
    try:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    cells = state.get("cells", {})
    if not cells:
        return None
    plan = {}
    for variant in variants:
        existing = {k: c for k, c in cells.items() if c.get("variant") == variant}
        frontier = min((int(c["next_index"]) for c in existing.values()), default=0)
        for method in method_names:
            key = _cell_key(variant, method)
            plan[key] = int(cells[key]["next_index"]) if key in cells else frontier
    return plan


def _write_state(checkpoint_path: Path, state: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")
    temporary.replace(checkpoint_path)


# --- instrumentation -------------------------------------------------------------


def _read_total_queries(retriever: Retriever) -> float | None:
    if hasattr(retriever, "total_queries"):
        return float(retriever.total_queries())
    return None


def _retriever_action_log(retriever: Retriever) -> list | None:
    batch_stats = getattr(retriever, "batch_stats", None)
    if batch_stats is not None and isinstance(
        getattr(batch_stats, "action_log", None), list
    ):
        return batch_stats.action_log
    action_log = getattr(retriever, "action_log", None)
    return action_log if isinstance(action_log, list) else None


def _query_delta(before: float | None, after: float | None) -> float:
    if before is None or after is None:
        return 1.0
    delta = after - before
    return float(delta) if delta > 0 else 1.0


def _geo_stats(retriever) -> Any | None:
    """The GeoStats object behind a retriever, however deep it is wrapped."""
    seen = 0
    current = retriever
    while current is not None and seen < 6:
        stats = getattr(current, "stats", None)
        if (
            stats is not None
            and hasattr(stats, "widenings")
            and hasattr(stats, "levels")
        ):
            return stats
        current = getattr(current, "base_retriever", None)
        seen += 1
    return None


def _geo_snapshot(retriever) -> dict | None:
    stats = _geo_stats(retriever)
    if stats is None:
        return None
    return {
        "resolved": stats.resolved,
        "unresolved": stats.unresolved,
        "widenings": stats.widenings,
        "levels": dict(stats.levels),
    }


def _accumulate_geo(cell, before: dict | None, after: dict | None) -> None:
    if before is None or after is None:
        return
    geo = cell.setdefault(
        "geo", {"resolved": 0, "unresolved": 0, "widenings": 0, "levels": {}}
    )
    for key in ("resolved", "unresolved", "widenings"):
        geo[key] += after[key] - before[key]
    for level, count in after["levels"].items():
        geo["levels"][level] = (
            geo["levels"].get(level, 0) + count - before["levels"].get(level, 0)
        )


# --- equivalence judge -------------------------------------------------------------


def _build_equivalence_audit(state, variant, method_names, args, cache_root: Path):
    if not args.judge_equivalence:
        return None
    if args.judge_k < 1:
        raise ValueError("--judge-k must be at least 1")
    from judge_retrieval_equivalence import (
        IncrementalEquivalenceAudit,
        _build_client,
        checkpoint_view,
    )

    client, model_id = _build_client(args.judge_model)
    # The audit reads `signature.timed_ids` and `methods`; give it this variant's
    # view of the checkpoint. Cell dicts are shared, so its judgments land in the
    # same state the report reads.
    view = checkpoint_view(state, variant)
    view["methods"] = {
        name: cell for name, cell in view["methods"].items() if name in method_names
    }
    cache_path = cache_root.with_name(f"{cache_root.stem}.{variant}{cache_root.suffix}")
    audit = IncrementalEquivalenceAudit(
        state=view,
        judge=EvidenceEquivalenceJudge(client=client),
        model_id=model_id,
        cutoff=args.judge_k,
        cache_path=cache_path,
    )
    audit.backfill()
    return audit


# --- report ------------------------------------------------------------------------


class _LiveReporter:
    def __init__(
        self,
        *,
        state,
        variants,
        method_names,
        args,
        output_path,
        checkpoint_path,
        command,
        questions_by_variant,
        audits,
        every: float,
    ):
        self.state = state
        self.variants = variants
        self.method_names = method_names
        self.args = args
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.command = command
        self.questions_by_variant = questions_by_variant
        self.audits = audits
        self.every = every
        self._last = 0.0

    def write(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and self.every > 0 and now - self._last < self.every:
            return
        self._last = now
        runs = _all_runs(
            self.state,
            self.variants,
            self.method_names,
            self.questions_by_variant,
            self.args,
            self.audits,
        )
        report = _build_report(
            state=self.state,
            runs=runs,
            variants=self.variants,
            method_names=self.method_names,
            args=self.args,
            output_path=self.output_path,
            checkpoint_path=self.checkpoint_path,
            command=self.command,
            status="in-progress",
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(f".{self.output_path.name}.tmp")
        temporary.write_text(report + "\n", encoding="utf-8")
        temporary.replace(self.output_path)


def _build_report(
    *,
    state,
    runs,
    variants,
    method_names,
    args,
    output_path,
    checkpoint_path,
    command,
    status,
) -> str:
    signature = state["signature"]
    lines = [
        f"# Retrieval Comparison {date.today().isoformat()}",
        "",
        f"- Command: `{command}`",
        f"- Database: {_db_label()} | design: **{args.design}** | span target: "
        f"**{signature.get('span_target') or 'none'}**",
        f"- Variants: {', '.join(variants)}",
        f"- Methods ({len(method_names)}): {', '.join(method_names)}",
        f"- Questions: {args.limit if args.limit is not None else 'all'} | "
        f"Category: {args.category or 'all'} | Warmup: {signature['warmup']}",
        f"- Output: {output_path} | Checkpoint: {checkpoint_path}",
        f"- Updated: {datetime.now().isoformat(timespec='seconds')} | Status: {status}",
    ]
    if args.judge_equivalence:
        lines.append(
            f"- Equivalence judge: {args.judge_model or CONFIG['equivalence_judge_model']} "
            f"at k={args.judge_k}"
        )
    lines.extend(["", "## Progress", ""])
    for variant in variants:
        total = len(signature["questions"][variant]["timed_ids"])
        done = [
            f"{m} {state['cells'][_cell_key(variant, m)]['next_index']}/{total}"
            for m in method_names
            if _cell_key(variant, m) in state["cells"]
        ]
        lines.append(f"- {variant}: " + ", ".join(done))
    if not runs:
        lines.extend(["", "No scored queries yet."])
        return "\n".join(lines)

    for variant in variants:
        run = runs.get(variant)
        if run is None:
            continue
        lines.extend(["", f"## Variant `{variant}`", ""])
        lines.append(format_eval_report(run, include_categories=args.category is None))
        diagnostics = _agentic_diagnostics(run, state, variant)
        formatted = format_agentic_diagnostics(diagnostics)
        if formatted:
            lines.extend(["", formatted])
        geo = _geo_table(state, variant, run.methods)
        if geo:
            lines.extend(["", geo])
        if run.span_questions:
            lines.extend(
                [
                    "",
                    f"Span metrics average over {run.span_questions} anchored questions "
                    f"of {len(run.questions)} scored; `store_share@k` is a share of "
                    f"{run.store_chars:,} indexed characters in {run.store_chunks} chunks.",
                ]
            )
    if len(variants) > 1:
        lines.extend(["", "## Across variants", ""])
        lines.extend(_metric_sections(runs, variants, method_names))
    lines.extend(["", "## Chunk profile", "", _profile_table(variants, runs, state)])
    lines.extend(
        ["", "## Coverage", "", _coverage_table(state, variants, method_names, runs)]
    )
    lines.extend(["", "## Notes", "", *_notes(state, variants, method_names, args)])
    return "\n".join(lines)


def _agentic_diagnostics(run: EvalRun, state, variant: str) -> dict[str, Any]:
    return build_agentic_diagnostics(
        questions=run.questions,
        relevance=run.relevance,
        rankings=run.rankings,
        method_states={m: state["cells"][_cell_key(variant, m)] for m in run.methods},
        method_names=run.methods,
        cutoff=int(EVAL_CONFIG["category_hit_k"]),
    )


def _geo_table(state, variant: str, method_names: list[str]) -> str:
    rows = []
    for name in method_names:
        geo = state["cells"][_cell_key(variant, name)].get("geo") or {}
        asked = geo.get("resolved", 0) + geo.get("unresolved", 0)
        if not asked:
            continue
        levels = ", ".join(f"{k}={v}" for k, v in sorted(geo.get("levels", {}).items()))
        rows.append(
            [
                name,
                asked,
                geo["resolved"],
                f"{geo['resolved'] / asked:.0%}",
                geo["widenings"],
                levels or "-",
            ]
        )
    if not rows:
        return ""
    return (
        "Geo scope\n"
        + plain_table(
            ["method", "questions", "scoped", "scoped %", "widened", "levels used"],
            rows,
        )
        + (
            "\n\n`scoped` questions resolved to a place; `widened` needed a wider scope "
            "before enough candidates came back; `levels used` is the filter level that "
            "produced the final candidates (radius, nuts3, nuts2, country, none)."
        )
    )


def _metric_sections(runs: dict[str, EvalRun], variants, method_names) -> list[str]:
    lines: list[str] = []
    for metric in (*SPAN_METRICS, *COST_METRICS, *CHUNK_METRICS):
        rows = []
        for method in method_names:
            row = [method]
            for variant in variants:
                run = runs.get(variant)
                value = run.scores.get(method, {}).get(metric) if run else None
                row.append("-" if value is None else float(value))
            rows.append(row)
        if all(all(v == "-" for v in row[1:]) for row in rows):
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


def _profile_table(variants, runs, state) -> str:
    rows = []
    for variant in variants:
        _, store_chars = load_store_profile(variant)
        with connect_pages() as conn:
            chunks = int(
                conn.execute(
                    "select count(*) from page_chunks where variant = ?", (variant,)
                ).fetchone()[0]
            )
        run = runs.get(variant)
        rows.append(
            [
                variant,
                chunks,
                f"{store_chars:,}",
                _median_tokens(variant),
                len(state["signature"]["questions"][variant]["timed_ids"]),
                run.span_questions if run else "-",
            ]
        )
    return plain_table(
        [
            "variant",
            "chunks",
            "indexed chars",
            "median tok (est.)",
            "timed questions",
            "anchored",
        ],
        rows,
    )


def _coverage_table(state, variants, method_names, runs) -> str:
    rows = []
    for variant in variants:
        run = runs.get(variant)
        for method in method_names:
            cell = state["cells"].get(_cell_key(variant, method))
            if cell is None:
                continue
            rows.append(
                [
                    variant,
                    method,
                    int(cell["next_index"]),
                    len(run.questions) if run and method in run.methods else 0,
                    f"{float(cell['elapsed_seconds']):.1f}",
                    int(cell.get("failures", 0)),
                ]
            )
    return plain_table(
        ["variant", "method", "answered", "scored (aligned)", "seconds", "failures"],
        rows,
    )


def _notes(state, variants, method_names, args) -> list[str]:
    notes = [
        "- Every variant's `Overall` table scores only the questions all of its started "
        "cells have answered (`scored (aligned)`), so a partially run cell never "
        "lowers a finished one.",
        "- Agentic diagnostics pair each agent with its reranked baseline in the same "
        "variant; a missing pair means the baseline was not in `--methods`.",
    ]
    if not args.judge_equivalence:
        notes.append(
            "- `judge_hit@K` is absent: run with --judge-equivalence to add it."
        )
    if args.no_span_metrics:
        notes.append("- Span metrics were disabled (--no-span-metrics).")
    failures = [
        f"{k} ({c['failures']})" for k, c in state["cells"].items() if c.get("failures")
    ]
    if failures:
        notes.append(
            "- Judge failures counted (check the log before quoting): "
            + ", ".join(failures)
        )
    return notes


# --- sidecars ------------------------------------------------------------------------


def _write_cell_rows(output_path: Path, runs: dict[str, EvalRun], args) -> Path:
    """One tidy row per (variant, method, metric), the merge format across reports."""
    path = output_path.with_suffix(output_path.suffix + ".cells.csv")
    rows = ["design,variant,method,metric,value,questions,span_questions,ms_per_query"]
    for variant, run in runs.items():
        for method in run.methods:
            measures = dict(run.scores[method])
            measures.update(
                {
                    f"category_hit/{k}": v
                    for k, v in (run.category_scores or {}).get(method, {}).items()
                }
            )
            for metric, value in measures.items():
                if not isinstance(value, (int, float)):
                    continue
                rows.append(
                    f"{args.design},{variant},{method},{metric},{value:.6f},"
                    f"{len(run.questions)},{run.span_questions},"
                    f"{run.timings[method]['ms_per_query']:.3f}"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _save_action_logs_from_state(state, output_path: Path) -> None:
    log = {
        key: cell["action_log"]
        for key, cell in state["cells"].items()
        if cell.get("action_log")
    }
    if not log:
        return
    path = output_path.with_name(
        output_path.stem.replace(".", "_") + "-judge-actions.json"
    )
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Judge action log saved to: {path}")


def _save_agentic_diagnostics(runs, state, output_path: Path) -> None:
    diagnostics = {
        variant: _agentic_diagnostics(run, state, variant)
        for variant, run in runs.items()
    }
    diagnostics = {v: d for v, d in diagnostics.items() if d.get("summaries")}
    if not diagnostics:
        return
    path = output_path.with_name(f"{output_path.stem}-agentic-diagnostics.json")
    path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Agentic diagnostics saved to: {path}")


if __name__ == "__main__":
    run_cli(main)
