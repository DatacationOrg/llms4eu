from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from datetime import datetime
from typing import Any

from src.db.pages import initialize_page_artifacts_db
from src.eval.equivalence import EvidenceEquivalenceJudge
from src.eval.evaluate import CONFIG as EVAL_CONFIG
from src.eval.evaluate import (
    EvalRun,
    add_judge_adjusted_scores,
    format_eval_report,
    load_eval_rows,
    score_eval_rankings,
)
from src.eval.metrics import score_rankings
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    ensure_retrievers_ready,
    list_retrievers,
)
from src.shared.env import data_path, load_local_env

DEFAULT_METHODS = (
    "sparse_rerank",
    "qwen4b_hybrid_rerank",
    "nemotron",
    "nemotron_hybrid_rerank",
)
# v1 vs v2 chunk representation, same retriever otherwise.
PHASE2_METHODS = ("nemotron_hybrid_rerank",)
DEFAULT_OUTPUT = Path("docs/retrieval-results-chunks.md")
DEFAULT_WARMUP = 5


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
        else data_path("judge-cache", f"{output_path.name}.equivalence.json")
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
        log_every=max(1, args.log_every),
        catch_up_only=args.catch_up_only,
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
        judge_details=judge_details,
    )
    print(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
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
            "sparse, Qwen4B and Nemotron benchmark suite. Use 'all' for the whole "
            "catalog, or 'phase2' / 'phase2-nemotron' for v1/v2 comparisons."
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
            f"--limit is set and to {DEFAULT_WARMUP} "
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
        "--log-every",
        type=int,
        default=1,
        help="Log progress every N timed queries per method (default: 1).",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep completed rankings for post-hoc evidence-equivalence auditing.",
    )
    parser.add_argument(
        "--catch-up-only",
        action="store_true",
        help=(
            "Run newly added methods only until they reach the existing checkpoint's "
            "shared progress, without advancing methods already in the checkpoint."
        ),
    )
    parser.add_argument(
        "--judge-equivalence",
        action="store_true",
        help=(
            "Use the local evidence-equivalence judge after each retrieved ranking "
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


def _resolve_methods(raw_methods: str) -> list[str]:
    group = raw_methods.strip().lower()
    if group == "all":
        return list_retrievers()
    if group in {"phase2", "phase2-nemotron"}:
        return list(PHASE2_METHODS)

    methods = [name.strip() for name in raw_methods.split(",") if name.strip()]
    if not methods:
        raise ValueError(
            "No methods provided. Use --methods all or a comma-separated list."
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
    formatted_run = format_eval_report(run, include_categories=include_categories)
    lines.extend(["", formatted_run])
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

    # Sibling script, not a package: this resolves because running
    # experiments/indexing/*.py puts that directory on sys.path.
    from judge_retrieval_equivalence import (
        _build_client,
        _write_json,
        judge_checkpoint,
    )

    client, model_id = _build_client(None)
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
    return DEFAULT_WARMUP


def _default_checkpoint_path(output_path: Path) -> Path:
    return data_path("checkpoints", f"{output_path.name}.checkpoint.json")


def _run_eval_with_checkpoint(
    method_names: list[str],
    limit: int | None,
    category: str | None,
    warmup: int,
    checkpoint_path: Path,
    output_path: Path,
    resume: bool,
    save_every: int,
    log_every: int = 1,
    catch_up_only: bool = False,
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
        "version": 3,
        "method_names": method_names,
        "category": category,
        "limit": limit,
        "warmup": warmup,
        "warmup_ids": [row["id"] for row in warmup_questions],
        "timed_ids": [row["id"] for row in timed_questions],
        "result_limit": EVAL_CONFIG["result_limit"],
        "scoring_unit": "chunk",
        "eligible_questions": len(questions),
    }
    catch_up = (
        _catch_up_plan(checkpoint_path, method_names)
        if catch_up_only and resume
        else None
    )
    if catch_up_only and catch_up is None:
        raise ValueError("--catch-up-only requires a compatible existing checkpoint")
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
        log_every=log_every,
        targets=catch_up,
        equivalence_audit=equivalence_audit,
    )

    aligned = min(
        (int(state["methods"][name]["next_index"]) for name in method_names),
        default=0,
    )
    scored_questions = timed_questions[:aligned]
    scored_ids = {str(row["id"]) for row in scored_questions}

    # Question ids are UUID strings; keep them as-is so scoring keys line up.
    method_rankings = {
        name: {
            question_id: ranked
            for question_id, ranked in state["methods"][name]["rankings"].items()
            if question_id in scored_ids
        }
        for name in method_names
    }

    timed_question_ids = {row["id"] for row in scored_questions}
    timed_relevance = [
        row for row in relevance if row["question_id"] in timed_question_ids
    ]
    return _build_eval_run(
        state=state,
        questions=scored_questions,
        relevance=timed_relevance,
        rankings=method_rankings,
        method_names=method_names,
        warmup=warmup,
    ), state


def _build_eval_run(
    *,
    state: dict[str, Any],
    questions: list[dict],
    relevance: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
    warmup: int,
    divisor: int | None = None,
) -> EvalRun:
    """Score one set of rankings and wrap it with per-method timings."""
    count = divisor if divisor is not None else len(questions)
    timings = {}
    for name in method_names:
        elapsed = float(state["methods"][name]["elapsed_seconds"])
        total_queries = float(state["methods"][name].get("timed_total_queries", 0.0))
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / count if count else 0.0,
            "queries_per_query": total_queries / count if count else 0.0,
            "total_queries": total_queries,
        }
    score_names, scores = score_eval_rankings(relevance, rankings, method_names)
    category_metric_names, category_scores = _score_method_categories(
        questions=questions,
        relevance=relevance,
        rankings=rankings,
        method_names=method_names,
    )
    return EvalRun(
        questions=questions,
        relevance=relevance,
        methods=method_names,
        warmup_count=warmup,
        rankings=rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
        category_metric_names=category_metric_names,
        category_scores=category_scores,
    )


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
    log_every: int,
    targets: dict[str, int] | None = None,
    equivalence_audit=None,
) -> None:
    total_timed = len(timed_questions)
    target_by_method = targets or {name: total_timed for name in method_names}
    for name in method_names:
        state["methods"][name].setdefault("timed_total_queries", 0.0)

    while any(
        state["methods"][name]["next_index"] < target_by_method[name]
        for name in method_names
    ):
        for method_name in method_names:
            method_state = state["methods"][method_name]
            next_index = int(method_state["next_index"])
            if next_index >= target_by_method[method_name]:
                continue

            retriever = retrievers[method_name]
            row = timed_questions[next_index]
            should_log = (next_index + 1) % log_every == 0
            if should_log:
                print(
                    f"{method_name}: running {next_index + 1}/{total_timed}",
                    flush=True,
                )
            before_queries = _read_total_queries(retriever)
            started = time.perf_counter()
            chunks = retriever.retrieve(row["question"], EVAL_CONFIG["result_limit"])
            elapsed = time.perf_counter() - started
            after_queries = _read_total_queries(retriever)

            method_state["elapsed_seconds"] += elapsed
            method_state["timed_total_queries"] += _query_delta(
                before_queries, after_queries
            )
            method_state["rankings"][str(row["id"])] = [chunk.id for chunk in chunks]
            method_state["next_index"] = next_index + 1
            if equivalence_audit is not None:
                equivalence_audit.judge_prediction(
                    method_name,
                    str(row["id"]),
                    method_state["rankings"][str(row["id"])],
                )

            completed = int(method_state["next_index"])
            if should_log:
                print(
                    f"{method_name}: completed {completed}/{total_timed} "
                    f"in {elapsed:.2f}s ({len(chunks)} results)",
                    flush=True,
                )
            if completed % save_every == 0 or completed == total_timed:
                _write_state(checkpoint_path, state)
                # Rescoring the whole run is expensive; refresh the live report on
                # the checkpoint cadence, not once per question.
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

    client, model_id = _build_client(None)
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
        "methods": {name: _empty_method_state() for name in method_names},
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
        "scoring_unit",
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
    }


def _catch_up_plan(
    checkpoint_path: Path,
    method_names: list[str],
) -> dict[str, int] | None:
    if not checkpoint_path.exists():
        return None
    try:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    existing = state.get("methods", {})
    if not existing or not set(existing).issubset(method_names):
        return None
    frontier = min(int(method["next_index"]) for method in existing.values())
    return {
        name: int(existing[name]["next_index"]) if name in existing else frontier
        for name in method_names
    }


def _score_method_categories(
    *,
    questions: list[dict],
    relevance: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    question_type_by_id = {row["id"]: row["question_type"] for row in questions}
    question_types = sorted(set(question_type_by_id.values()))
    cutoff = int(EVAL_CONFIG["category_hit_k"])
    metric_names: dict[str, str] = {}
    category_scores: dict[str, dict[str, float]] = {}
    for method_name in method_names:
        method_relevance = relevance
        method_rankings = rankings[method_name]
        metric_name = f"hit@{cutoff}"
        metric_names[method_name] = metric_name
        category_scores[method_name] = {}
        for question_type in question_types:
            question_ids = {
                question_id
                for question_id, current_type in question_type_by_id.items()
                if current_type == question_type
            }
            typed_relevance = [
                row for row in method_relevance if row["question_id"] in question_ids
            ]
            typed_rankings = {
                question_id: ranked
                for question_id, ranked in method_rankings.items()
                if question_id in question_ids
            }
            category_scores[method_name][question_type] = score_rankings(
                typed_relevance,
                typed_rankings,
                ks=(cutoff,),
                mrr_k=cutoff,
            )[f"hit@{cutoff}"]
    return metric_names, category_scores


def _write_state(checkpoint_path: Path, state: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _read_total_queries(retriever: Retriever) -> float | None:
    return (
        float(retriever.total_queries())
        if hasattr(retriever, "total_queries")
        else None
    )


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
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(report + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


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
    partial_run = _build_eval_run(
        state=state,
        questions=partial_questions,
        relevance=partial_relevance,
        rankings=partial_rankings,
        method_names=method_names,
        warmup=warmup,
        divisor=aligned,
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
    main()
