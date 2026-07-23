from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from datetime import datetime
from typing import Any

from _cli import run_cli
from src.db.pages import initialize_page_artifacts_db
from src.eval.evaluate import CONFIG as EVAL_CONFIG
from src.eval.evaluate import (
    EvalRun,
    add_judge_adjusted_scores,
    count_chunk_expansions,
    format_eval_report,
    load_eval_rows,
    score_eval_rankings,
)
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    ensure_retrievers_ready,
    list_retrievers,
)
from src.retrieval.retrievers.agentic import AgenticRetriever
from src.shared.env import ROOT, load_local_env, load_yaml
from src.eval.equivalence import EvidenceEquivalenceJudge

DEFAULT_METHODS = (
    "sparse_rerank",
    "qwen4b_hybrid_rerank",
    "nemotron",
    "nemotron_hybrid_rerank",
    "azure_hybrid_rerank",
    "azure_hybrid_agentic",
    "nemotron_hybrid_agentic",
)
DEFAULT_OUTPUT = Path("docs/retrieval-results-chunks-okf.md")
CONFIG = load_yaml(ROOT / "experiments" / "indexing" / "config.yaml")


def main() -> None:
    load_local_env()
    args = _parse_args()
    method_names = _resolve_methods(args.methods)
    limit = args.limit
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else _default_checkpoint_path(output_path)
    )
    judge_cache_path = (
        Path(args.judge_cache)
        if args.judge_cache
        else output_path.with_suffix(output_path.suffix + ".equivalence.json")
    )
    warmup = _resolve_warmup(args.warmup, limit)
    run, state = _run_eval_with_checkpoint(
        method_names=method_names,
        limit=limit,
        category=args.category,
        warmup=warmup,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        resume=not args.no_resume,
        save_every=max(1, args.save_every),
        judge_cutoff=args.judge_k if args.judge_equivalence else None,
        judge_cache_path=judge_cache_path,
    )
    judge_details = None
    if args.judge_equivalence:
        run, judge_details = _add_equivalence_judgments(
            run=run,
            state=state,
            output_path=output_path,
            cutoff=args.judge_k,
            cache_path=judge_cache_path,
        )
    report = _format_report(
        run,
        method_names=method_names,
        output_path=output_path,
        include_categories=args.category is None,
        state=state,
        judge_details=judge_details,
    )
    print(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    _save_action_logs_from_state(state, method_names, output_path)
    if checkpoint_path.exists() and not args.keep_checkpoint:
        checkpoint_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval methods with live report and checkpoint resume.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated retriever names to compare. Defaults to the primary "
            "sparse, Qwen4B, Nemotron, Azure, and agentic benchmark suite. Use "
            "'all-agentic' to include every registered agentic method."
        ),
    )
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=(
            "Optional number of warmup questions. If omitted, defaults to 0 when "
            "--limit is set and to experiments/indexing/config.yaml:default_warmup "
            "for full runs."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional output file path for the full text report. Defaults to "
            f"{DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Optional checkpoint file path used for crash-safe resume. "
            "Defaults to <output>.checkpoint.json."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and start from scratch.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist checkpoint every N timed queries per method (default: 10).",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep completed rankings for post-hoc evidence-equivalence auditing.",
    )
    parser.add_argument(
        "--judge-equivalence",
        action="store_true",
        help=(
            "Use the Azure evidence-equivalence judge after each retrieved ranking "
            "and add judge_hit@K to the overall table. Judgments are cached for resume."
        ),
    )
    parser.add_argument(
        "--judge-k",
        type=int,
        default=int(EVAL_CONFIG["result_limit"]),
        help="Retrieval cutoff for evidence-equivalence judging (default: result_limit).",
    )
    parser.add_argument(
        "--judge-cache",
        help="Optional equivalence-judgment cache path.",
    )
    return parser.parse_args()


def _agentic_retrievers() -> list[str]:
    return sorted(name for name in list_retrievers() if "agentic" in name)


def _resolve_methods(raw_methods: str) -> list[str]:
    if raw_methods.strip().lower() in {"all", "all-agentic"}:
        return _agentic_retrievers()

    methods = [name.strip() for name in raw_methods.split(",") if name.strip()]
    if not methods:
        raise ValueError(
            "No methods provided. Use --methods all-agentic or a comma-separated list."
        )

    available = set(list_retrievers())
    unknown = sorted(set(methods) - available)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def _format_report(
    run,
    method_names: list[str],
    output_path: Path,
    include_categories: bool,
    state: dict[str, Any],
    judge_details: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"Methods: {', '.join(method_names)}",
        "Scoring unit: chunk",
        f"Warmup: {run.warmup_count} queries",
        f"Output: {output_path}",
    ]
    if judge_details:
        lines.extend(
            [
                f"Equivalence judge: {judge_details['model_id']}",
                f"Equivalence cutoff: {judge_details['cutoff']}",
                f"Equivalence cache: {judge_details['cache_path']}",
            ]
        )
    lines.extend(["", format_eval_report(run, include_categories=include_categories)])
    return "\n".join(lines)


def _add_equivalence_judgments(
    *,
    run: EvalRun,
    state: dict[str, Any],
    output_path: Path,
    cutoff: int,
    cache_path: Path,
) -> tuple[EvalRun, dict[str, Any]]:
    if cutoff < 1:
        raise ValueError("--judge-k must be at least 1")

    from judge_retrieval_equivalence import (
        _build_client,
        _write_json,
        judge_checkpoint,
    )

    client, model_id = _build_client("azure", None)
    judge = EvidenceEquivalenceJudge(client=client)
    judge_report, cache, summaries = judge_checkpoint(
        state=state,
        judge=judge,
        model_id=model_id,
        cutoff=cutoff,
        cache_path=cache_path,
    )
    _write_json(cache_path, cache)
    audit_path = output_path.with_name(f"{output_path.stem}-equivalence.md")
    audit_path.write_text(judge_report + "\n", encoding="utf-8")

    judged_run = add_judge_adjusted_scores(run, summaries, cutoff)
    return judged_run, {
        "model_id": model_id,
        "cutoff": cutoff,
        "cache_path": cache_path,
        "audit_path": audit_path,
    }


def _resolve_warmup(requested_warmup: int | None, limit: int | None) -> int:
    if requested_warmup is not None:
        return requested_warmup
    if limit is not None:
        return 0
    return int(CONFIG["default_warmup"])


def _default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.checkpoint.json")


def _save_action_logs_from_state(
    state: dict[str, Any],
    method_names: list[str],
    output_path: Path,
) -> None:
    log = {
        name: state["methods"][name].get("action_log", [])
        for name in method_names
        if state["methods"][name].get("action_log")
    }
    if not log:
        return
    log_path = output_path.with_name(
        output_path.stem.replace(".", "_") + "-judge-actions.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Judge action log saved to: {log_path}")


def _run_eval_with_checkpoint(
    method_names: list[str],
    limit: int | None,
    category: str | None,
    warmup: int,
    checkpoint_path: Path,
    output_path: Path,
    resume: bool,
    save_every: int,
    retrievers_out: dict[str, Retriever] | None = None,
    judge_cutoff: int | None = None,
    judge_cache_path: Path | None = None,
) -> tuple[EvalRun, dict[str, Any]]:
    initialize_page_artifacts_db()
    questions, relevance = load_eval_rows(limit=limit, category=category)
    if not questions:
        return EvalRun([], [], [], 0, {}, {}, [], {}), {}

    warmup = min(warmup, max(len(questions) - 1, 0))
    warmup_questions = questions[:warmup]
    timed_questions = questions[warmup:]
    signature = {
        "version": 2,
        "method_names": method_names,
        "category": category,
        "limit": limit,
        "warmup": warmup,
        "warmup_ids": [row["id"] for row in warmup_questions],
        "timed_ids": [row["id"] for row in timed_questions],
        "result_limit": EVAL_CONFIG["result_limit"],
        "scoring_unit": "chunk",
        "okf_bundle": None,
        "okf_covered_pages": 0,
        "okf_concepts": 0,
        "eligible_questions": len(questions),
    }
    state = _load_or_initialize_state(
        checkpoint_path=checkpoint_path,
        signature=signature,
        method_names=method_names,
        resume=resume,
    )
    equivalence_audit = _build_incremental_equivalence_audit(
        state=state,
        cutoff=judge_cutoff,
        cache_path=judge_cache_path,
    )
    if equivalence_audit is not None:
        equivalence_audit.backfill()

    _write_live_report(
        state=state,
        method_names=method_names,
        warmup=warmup,
        timed_questions=timed_questions,
        relevance=relevance,
        category=category,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        equivalence_audit=equivalence_audit,
    )

    ensure_retrievers_ready(method_names)
    retrievers = {name: build_retriever(name) for name in method_names}
    if retrievers_out is not None:
        retrievers_out.update(retrievers)

    for name in method_names:
        _run_warmup_with_checkpoint(
            method_name=name,
            retriever=retrievers[name],
            warmup_questions=warmup_questions,
            state=state,
            checkpoint_path=checkpoint_path,
        )

    _run_timed_round_robin(
        method_names=method_names,
        retrievers=retrievers,
        timed_questions=timed_questions,
        state=state,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        relevance=relevance,
        warmup=warmup,
        category=category,
        save_every=save_every,
        equivalence_audit=equivalence_audit,
    )

    # Question ids are UUID strings; keep them as-is so scoring keys line up.
    method_rankings = {
        name: {
            question_id: ranked
            for question_id, ranked in state["methods"][name]["rankings"].items()
        }
        for name in method_names
    }

    timings: dict[str, dict[str, float]] = {}
    for name in method_names:
        elapsed = float(state["methods"][name]["elapsed_seconds"])
        total_queries = float(state["methods"][name].get("timed_total_queries", 0.0))
        question_count = len(timed_questions)
        queries_per_query = total_queries / question_count if question_count else 0.0
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / len(timed_questions),
            "queries_per_query": queries_per_query,
            "total_queries": total_queries,
            "chunk_expansions": count_chunk_expansions(
                state["methods"][name].get("action_log", [])
            ),
        }

    timed_question_ids = {row["id"] for row in timed_questions}
    timed_relevance = [
        row for row in relevance if row["question_id"] in timed_question_ids
    ]
    score_names, scores = score_eval_rankings(
        timed_relevance,
        method_rankings,
        method_names,
    )
    return EvalRun(
        questions=timed_questions,
        relevance=timed_relevance,
        methods=method_names,
        warmup_count=warmup,
        rankings=method_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
    ), state


def _run_warmup_with_checkpoint(
    method_name: str,
    retriever: Retriever,
    warmup_questions: list[dict[str, Any]],
    state: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    method_state = state["methods"][method_name]
    for index in range(method_state["warmup_index"], len(warmup_questions)):
        retriever.retrieve(
            warmup_questions[index]["question"], EVAL_CONFIG["result_limit"]
        )
        method_state["warmup_index"] = index + 1
        _write_state(checkpoint_path, state)


def _run_timed_round_robin(
    method_names: list[str],
    retrievers: dict[str, Retriever],
    timed_questions: list[dict[str, Any]],
    state: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    relevance: list[dict[str, Any]],
    warmup: int,
    category: str | None,
    save_every: int,
    equivalence_audit=None,
) -> None:
    total_timed = len(timed_questions)
    for name in method_names:
        state["methods"][name].setdefault("timed_total_queries", 0.0)

    while any(
        state["methods"][name]["next_index"] < total_timed for name in method_names
    ):
        for method_name in method_names:
            method_state = state["methods"][method_name]
            next_index = int(method_state["next_index"])
            if next_index >= total_timed:
                continue

            retriever = retrievers[method_name]
            row = timed_questions[next_index]
            before_queries = _read_total_queries(retriever)
            before_failures = int(getattr(retriever, "failures", 0))
            action_log = _retriever_action_log(retriever)
            prev_log_len = len(action_log) if action_log is not None else 0
            started = time.perf_counter()
            chunks = retriever.retrieve(row["question"], EVAL_CONFIG["result_limit"])
            elapsed = time.perf_counter() - started
            after_queries = _read_total_queries(retriever)
            after_failures = int(getattr(retriever, "failures", 0))

            if action_log is not None:
                new_entries = action_log[prev_log_len:]
                method_state.setdefault("action_log", []).extend(new_entries)

            method_state["elapsed_seconds"] += elapsed
            method_state["timed_total_queries"] += _query_delta(
                before_queries, after_queries
            )
            method_state["failures"] = int(method_state.get("failures", 0)) + max(
                after_failures - before_failures, 0
            )
            method_state["rankings"][str(row["id"])] = [chunk.id for chunk in chunks]
            method_state["next_index"] = next_index + 1
            if equivalence_audit is not None:
                equivalence_audit.judge_prediction(
                    method_name,
                    str(row["id"]),
                    method_state["rankings"][str(row["id"])],
                )

            _write_live_report(
                state=state,
                method_names=method_names,
                warmup=warmup,
                timed_questions=timed_questions,
                relevance=relevance,
                category=category,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                equivalence_audit=equivalence_audit,
            )

            completed = int(method_state["next_index"])
            if completed % save_every == 0 or completed == total_timed:
                print(
                    f"{method_name}: processed {completed}/{total_timed} timed questions"
                )
                _write_state(checkpoint_path, state)


def _build_incremental_equivalence_audit(
    *,
    state: dict[str, Any],
    cutoff: int | None,
    cache_path: Path | None,
):
    if cutoff is None:
        return None
    if cache_path is None:
        raise ValueError("An equivalence cache path is required")

    from judge_retrieval_equivalence import (
        IncrementalEquivalenceAudit,
        _build_client,
    )

    client, model_id = _build_client("azure", None)
    return IncrementalEquivalenceAudit(
        state=state,
        judge=EvidenceEquivalenceJudge(client=client),
        model_id=model_id,
        cutoff=cutoff,
        cache_path=cache_path,
    )


def _load_or_initialize_state(
    checkpoint_path: Path,
    signature: dict[str, Any],
    method_names: list[str],
    resume: bool,
) -> dict[str, Any]:
    if resume and checkpoint_path.exists():
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if state.get("signature") == signature:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                return state
            if _extend_compatible_state(state, signature, method_names):
                print(f"Extending compatible checkpoint: {checkpoint_path}")
                _write_state(checkpoint_path, state)
                return state
            print("Ignoring incompatible checkpoint and starting fresh.")
        except json.JSONDecodeError:
            print("Checkpoint file is corrupted; starting fresh.")

    state = {
        "signature": signature,
        "methods": {
            name: {
                "warmup_index": 0,
                "next_index": 0,
                "elapsed_seconds": 0.0,
                "timed_total_queries": 0.0,
                "rankings": {},
                "action_log": [],
                "failures": 0,
            }
            for name in method_names
        },
    }
    _write_state(checkpoint_path, state)
    return state


def _extend_compatible_state(
    state: dict[str, Any],
    signature: dict[str, Any],
    method_names: list[str],
) -> bool:
    previous = state.get("signature", {})
    stable_keys = (
        "version",
        "category",
        "limit",
        "warmup",
        "warmup_ids",
        "timed_ids",
        "result_limit",
    )
    if any(previous.get(key) != signature.get(key) for key in stable_keys):
        return False

    existing_methods = set(state.get("methods", {}))
    if not existing_methods or not existing_methods.issubset(method_names):
        return False

    for name in method_names:
        state["methods"].setdefault(name, _empty_method_state())
    state["signature"] = signature
    return True


def _empty_method_state() -> dict[str, Any]:
    return {
        "warmup_index": 0,
        "next_index": 0,
        "elapsed_seconds": 0.0,
        "timed_total_queries": 0.0,
        "rankings": {},
        "action_log": [],
        "failures": 0,
    }


def _write_state(checkpoint_path: Path, state: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _read_total_queries(retriever: Retriever) -> float | None:
    if hasattr(retriever, "total_queries"):
        return float(retriever.total_queries())
    return None


def _retriever_action_log(retriever: Retriever) -> list | None:
    if isinstance(retriever, AgenticRetriever):
        return retriever.batch_stats.action_log
    action_log = getattr(retriever, "action_log", None)
    return action_log if isinstance(action_log, list) else None


def _query_delta(before: float | None, after: float | None) -> float:
    if before is None or after is None:
        return 1.0
    delta = after - before
    if delta <= 0:
        return 1.0
    return float(delta)


def _write_live_report(
    state: dict[str, Any],
    method_names: list[str],
    warmup: int,
    timed_questions: list[dict[str, Any]],
    relevance: list[dict[str, Any]],
    category: str | None,
    output_path: Path,
    checkpoint_path: Path,
    equivalence_audit=None,
) -> None:
    report = _build_live_report(
        state=state,
        method_names=method_names,
        warmup=warmup,
        timed_questions=timed_questions,
        relevance=relevance,
        include_categories=category is None,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        equivalence_audit=equivalence_audit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")


def _build_live_report(
    state: dict[str, Any],
    method_names: list[str],
    warmup: int,
    timed_questions: list[dict[str, Any]],
    relevance: list[dict[str, Any]],
    include_categories: bool,
    output_path: Path,
    checkpoint_path: Path,
    equivalence_audit=None,
) -> str:
    total_timed = len(timed_questions)
    progress_by_method = {
        name: int(state["methods"][name]["next_index"]) for name in method_names
    }
    aligned = min(progress_by_method.values()) if progress_by_method else 0

    header = [
        f"Methods: {', '.join(method_names)}",
        "Scoring unit: chunk",
        f"Warmup: {warmup} queries",
        f"Output: {output_path}",
        f"Checkpoint: {checkpoint_path}",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "Status: in-progress",
        "",
        "Progress",
    ]
    if equivalence_audit is not None:
        header[5:5] = [
            f"Equivalence judge: {equivalence_audit.model_id}",
            f"Equivalence cutoff: {equivalence_audit.cutoff}",
            f"Equivalence cache: {equivalence_audit.cache_path}",
        ]
    for name in method_names:
        header.append(
            f"- {name}: {progress_by_method[name]}/{total_timed} timed queries"
        )
    header.append(f"- aligned_for_scoring: {aligned}/{total_timed}")
    if aligned == 0:
        header.extend(
            [
                "",
                "No shared scored queries yet. Detailed metrics will appear once all methods complete at least one timed query.",
            ]
        )
        return "\n".join(header)

    partial_questions = timed_questions[:aligned]
    partial_ids = {row["id"] for row in partial_questions}
    partial_ids_as_str = {str(question_id) for question_id in partial_ids}
    partial_relevance = [row for row in relevance if row["question_id"] in partial_ids]
    partial_rankings = {
        name: {
            question_id: rankings
            for question_id, rankings in state["methods"][name]["rankings"].items()
            if question_id in partial_ids_as_str
        }
        for name in method_names
    }
    score_names, scores = score_eval_rankings(
        partial_relevance,
        partial_rankings,
        method_names,
    )
    timings: dict[str, dict[str, float]] = {}
    for name in method_names:
        elapsed = float(state["methods"][name]["elapsed_seconds"])
        total_queries = float(state["methods"][name].get("timed_total_queries", 0.0))
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / aligned,
            "queries_per_query": total_queries / aligned,
            "total_queries": total_queries,
            "chunk_expansions": count_chunk_expansions(
                state["methods"][name].get("action_log", [])
            ),
        }

    partial_run = EvalRun(
        questions=partial_questions,
        relevance=partial_relevance,
        methods=method_names,
        warmup_count=warmup,
        rankings=partial_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
    )
    if equivalence_audit is not None:
        summaries = equivalence_audit.summaries(
            method_names,
            [str(row["id"]) for row in partial_questions],
        )
        partial_run = add_judge_adjusted_scores(
            partial_run,
            summaries,
            equivalence_audit.cutoff,
        )

    body = format_eval_report(partial_run, include_categories=include_categories)
    header.extend(["", body])
    return "\n".join(header)


if __name__ == "__main__":
    run_cli(main)
