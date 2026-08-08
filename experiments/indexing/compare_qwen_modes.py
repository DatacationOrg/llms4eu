from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from datetime import datetime
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
    count_chunk_expansions,
    format_eval_report,
    load_eval_rows,
    score_eval_rankings,
)
from src.eval.metrics import score_rankings
from src.okf.evidence import (
    OKFConceptRetriever,
    invert_page_map,
    load_bundle_page_map,
    project_pages_to_concepts,
)
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    ensure_retrievers_ready,
    list_retrievers,
)
from src.retrieval.retrievers.agentic import AgenticRetriever
from src.shared.env import ROOT, load_local_env, load_yaml

DEFAULT_METHODS = (
    "sparse_rerank",
    "qwen4b_hybrid_rerank",
    "nemotron",
    "nemotron_hybrid_rerank",
    "embed_v4_hybrid_rerank",
    "embed_v4_hybrid_agentic",
    "nemotron_hybrid_agentic",
)
PHASE2_METHODS = {
    "phase2-nemotron": (
        "nemotron_hybrid_rerank",
        "nemotron_hybrid_rerank_v2",
        "nemotron_hybrid_agentic",
        "nemotron_hybrid_agentic_v2",
    ),
    "phase2-embed-v4": (
        "embed_v4_hybrid_rerank",
        "embed_v4_hybrid_rerank_v2",
        "embed_v4_hybrid_agentic",
        "embed_v4_hybrid_agentic_v2",
    ),
}
OKF_METHOD = "okf"
OKF_METHODS = {OKF_METHOD}
COMPREHENSIVE_OKF_METHODS = (
    "sparse_rerank",
    "qwen4b_hybrid_rerank",
    "nemotron_hybrid_rerank",
    "embed_v4_hybrid_rerank",
    "embed_v4_hybrid_agentic",
    "nemotron_hybrid_agentic",
    OKF_METHOD,
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
        log_every=max(1, args.log_every),
        okf_bundle=Path(args.okf_bundle),
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
        state=state,
        judge_details=judge_details,
    )
    print(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    _save_action_logs_from_state(state, method_names, output_path)
    _save_agentic_diagnostics(run, state, method_names, output_path)
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
            "'all-agentic' to include every registered agentic method, or "
            "'comprehensive-okf' for the full chunk+OKF benchmark, 'okf-only' "
            "for concept-level OKF scoring, or "
            "'phase2', 'phase2-nemotron', or 'phase2-embed-v4' for v1/v2 comparisons."
        ),
    )
    parser.add_argument(
        "--okf-bundle",
        default="data/okf/tourism",
        help="OKF bundle used when the okf method is selected.",
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


def _has_okf(method_names: list[str]) -> bool:
    return bool(OKF_METHODS.intersection(method_names))


def _uses_okf(method_names: list[str]) -> bool:
    return bool(method_names) and all(name in OKF_METHODS for name in method_names)


def _scoring_unit_label(method_names: list[str]) -> str:
    if _uses_okf(method_names):
        return "OKF concept"
    if _has_okf(method_names):
        return "per method (RAG: chunk; OKF: concept)"
    return "chunk"


def _build_okf_retriever(bundle_root: Path) -> Retriever:
    return OKFConceptRetriever(bundle_root)


def _resolve_methods(raw_methods: str) -> list[str]:
    group = raw_methods.strip().lower()
    if group in {"all", "all-agentic"}:
        return _agentic_retrievers()
    if group == "phase2":
        return list(dict.fromkeys(sum(PHASE2_METHODS.values(), ())))
    if group == "comprehensive-okf":
        return list(COMPREHENSIVE_OKF_METHODS)
    if group == "okf-only":
        return [OKF_METHOD]
    if group in PHASE2_METHODS:
        return list(PHASE2_METHODS[group])

    methods = [name.strip() for name in raw_methods.split(",") if name.strip()]
    if not methods:
        raise ValueError(
            "No methods provided. Use --methods all-agentic or a comma-separated list."
        )

    available = set(list_retrievers()) | OKF_METHODS
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
        f"Scoring unit: {_scoring_unit_label(method_names)}",
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
    if _has_okf(method_names):
        signature = state["signature"]
        lines.extend(
            [
                f"OKF bundle: {signature['okf_bundle']}",
                f"OKF coverage: {signature['okf_covered_pages']} pages in "
                f"{signature['okf_concepts']} concepts",
                f"OKF-eligible questions: {signature['eligible_questions']}",
                "OKF failures: "
                + ", ".join(
                    f"{name}={state['methods'][name].get('failures', 0)}"
                    for name in method_names
                    if name in OKF_METHODS
                ),
            ]
        )
    formatted_run = format_eval_report(run, include_categories=include_categories)
    lines.extend(["", formatted_run])
    diagnostics = _agentic_diagnostics(run, state, method_names)
    formatted_diagnostics = format_agentic_diagnostics(diagnostics)
    if formatted_diagnostics:
        lines.extend(["", formatted_diagnostics])
    return "\n".join(lines)


def _agentic_diagnostics(
    run: EvalRun,
    state: dict[str, Any],
    method_names: list[str],
) -> dict[str, Any]:
    return build_agentic_diagnostics(
        questions=run.questions,
        relevance=run.relevance,
        rankings=run.rankings,
        method_states=state.get("methods", {}),
        method_names=method_names,
        cutoff=int(EVAL_CONFIG["category_hit_k"]),
    )


def _save_agentic_diagnostics(
    run: EvalRun,
    state: dict[str, Any],
    method_names: list[str],
    output_path: Path,
) -> None:
    diagnostics = _agentic_diagnostics(run, state, method_names)
    if not diagnostics.get("summaries"):
        return
    path = output_path.with_name(f"{output_path.stem}-agentic-diagnostics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Agentic diagnostics saved to: {path}")


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

    judged_run = _add_method_judge_scores(run, summaries, cutoff)
    return judged_run, {
        "model_id": model_id,
        "cutoff": cutoff,
        "cache_path": cache_path,
        "audit_path": audit_path,
    }


def _add_method_judge_scores(
    run: EvalRun,
    summaries: dict[str, dict[str, int]],
    cutoff: int,
) -> EvalRun:
    return add_judge_adjusted_scores(run, summaries, cutoff)


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
    log_every: int = 1,
    okf_bundle: Path = Path("data/okf/tourism"),
    catch_up_only: bool = False,
    retrievers_out: dict[str, Retriever] | None = None,
    judge_cutoff: int | None = None,
    judge_cache_path: Path | None = None,
) -> tuple[EvalRun, dict[str, Any]]:
    initialize_page_artifacts_db()
    bundle_root = (ROOT / okf_bundle).resolve()
    questions, relevance, chunk_pages, page_concepts = _load_comparison_rows(
        method_names=method_names,
        limit=limit,
        category=category,
        bundle_root=bundle_root,
    )
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
        "scoring_unit": (
            "mixed"
            if _has_okf(method_names) and not _uses_okf(method_names)
            else "okf_concept"
            if _uses_okf(method_names)
            else "chunk"
        ),
        "okf_bundle": str(bundle_root) if _has_okf(method_names) else None,
        "okf_covered_pages": len(page_concepts),
        "okf_concepts": len(
            {concept for concepts in page_concepts.values() for concept in concepts}
        ),
        "okf_methods": sorted(OKF_METHODS.intersection(method_names)),
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
        chunk_pages=chunk_pages,
        page_concepts=page_concepts,
        equivalence_audit=equivalence_audit,
    )

    rag_methods = [name for name in method_names if name not in OKF_METHODS]
    ensure_retrievers_ready(rag_methods)
    retrievers = {
        name: (
            _build_okf_retriever(bundle_root)
            if name in OKF_METHODS
            else build_retriever(name)
        )
        for name in method_names
    }
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
        chunk_pages=chunk_pages,
        page_concepts=page_concepts,
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

    timings: dict[str, dict[str, float]] = {}
    for name in method_names:
        elapsed = float(state["methods"][name]["elapsed_seconds"])
        total_queries = float(state["methods"][name].get("timed_total_queries", 0.0))
        question_count = len(scored_questions)
        queries_per_query = total_queries / question_count if question_count else 0.0
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / question_count if question_count else 0.0,
            "queries_per_query": queries_per_query,
            "total_queries": total_queries,
            "chunk_expansions": count_chunk_expansions(
                state["methods"][name].get("action_log", [])
            ),
        }

    timed_question_ids = {row["id"] for row in scored_questions}
    timed_relevance = [
        row for row in relevance if row["question_id"] in timed_question_ids
    ]
    score_names, scores = _score_method_rankings(
        relevance=timed_relevance,
        rankings=method_rankings,
        method_names=method_names,
        chunk_pages=chunk_pages,
        page_concepts=page_concepts,
    )
    category_metric_names, category_scores = _score_method_categories(
        questions=scored_questions,
        relevance=timed_relevance,
        rankings=method_rankings,
        method_names=method_names,
        chunk_pages=chunk_pages,
        page_concepts=page_concepts,
    )
    return EvalRun(
        questions=scored_questions,
        relevance=timed_relevance,
        methods=method_names,
        warmup_count=warmup,
        rankings=method_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
        category_metric_names=category_metric_names,
        category_scores=category_scores,
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
    log_every: int,
    chunk_pages: dict[str, str],
    page_concepts: dict[str, list[str]],
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
            before_failures = int(getattr(retriever, "failures", 0))
            action_log = _retriever_action_log(retriever)
            prev_log_len = len(action_log) if action_log is not None else 0
            started = time.perf_counter()
            chunks = retriever.retrieve(row["question"], EVAL_CONFIG["result_limit"])
            elapsed = time.perf_counter() - started
            after_queries = _read_total_queries(retriever)
            after_failures = int(getattr(retriever, "failures", 0))

            if action_log is not None:
                new_entries = [dict(entry) for entry in action_log[prev_log_len:]]
                for entry in new_entries:
                    entry["question_id"] = str(row["id"])
                method_state.setdefault("action_log", []).extend(new_entries)

            method_state["elapsed_seconds"] += elapsed
            method_state["timed_total_queries"] += _query_delta(
                before_queries, after_queries
            )
            method_state["failures"] = int(method_state.get("failures", 0)) + max(
                after_failures - before_failures, 0
            )
            method_state["rankings"][str(row["id"])] = [chunk.id for chunk in chunks]
            method_state.setdefault("observations", {})[str(row["id"])] = {
                "elapsed_seconds": elapsed,
                "query_count": _query_delta(before_queries, after_queries),
                "actions": new_entries if action_log is not None else [],
            }
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
                chunk_pages=chunk_pages,
                page_concepts=page_concepts,
                equivalence_audit=equivalence_audit,
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
                "observations": {},
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

    previous_scoring = previous.get("scoring_unit", "chunk")
    scoring = signature.get("scoring_unit", "chunk")
    upgrading_to_okf = (
        previous_scoring == "chunk"
        and scoring in {"okf_concept", "mixed"}
        and not existing_methods.intersection(OKF_METHODS)
        and bool(set(method_names).intersection(OKF_METHODS))
    )
    if previous_scoring != scoring and not upgrading_to_okf:
        return False
    if scoring == "okf_concept" and not upgrading_to_okf:
        okf_keys = ("okf_bundle", "okf_covered_pages", "okf_concepts")
        if any(previous.get(key) != signature.get(key) for key in okf_keys):
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
        "observations": {},
        "failures": 0,
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


def _load_comparison_rows(
    *,
    method_names: list[str],
    limit: int | None,
    category: str | None,
    bundle_root: Path,
) -> tuple[list[dict], list[dict], dict[str, str], dict[str, list[str]]]:
    if not _has_okf(method_names):
        questions, relevance = load_eval_rows(limit=limit, category=category)
        return questions, relevance, {}, {}

    page_concepts = invert_page_map(load_bundle_page_map(bundle_root))
    if not page_concepts:
        raise RuntimeError(
            f"OKF bundle contains no source page provenance: {bundle_root}"
        )

    questions, relevance = load_eval_rows(limit=None, category=category)
    with connect_pages() as conn:
        chunk_pages = {
            str(row["id"]): str(row["page_id"])
            for row in conn.execute("select id, page_id from page_chunks")
        }
    covered_question_ids = {
        row["question_id"]
        for row in relevance
        if chunk_pages.get(row["chunk_id"]) in page_concepts
    }
    questions = [row for row in questions if row["id"] in covered_question_ids]
    if limit is not None:
        questions = questions[:limit]
    selected_ids = {row["id"] for row in questions}
    relevance = [row for row in relevance if row["question_id"] in selected_ids]
    return questions, relevance, chunk_pages, page_concepts


def _scoring_inputs(
    *,
    relevance: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
    chunk_pages: dict[str, str],
    page_concepts: dict[str, list[str]],
) -> tuple[list[dict], dict[str, dict[str, list[str]]]]:
    if not _uses_okf(method_names):
        return relevance, rankings

    concept_relevance = []
    seen_relevance = set()
    for row in relevance:
        page_id = chunk_pages.get(row["chunk_id"])
        for concept in page_concepts.get(page_id, []):
            key = (row["question_id"], concept)
            if key not in seen_relevance:
                concept_relevance.append(
                    {"question_id": row["question_id"], "chunk_id": concept}
                )
                seen_relevance.add(key)

    concept_rankings = {}
    for name in method_names:
        concept_rankings[name] = {}
        for question_id, ranked in rankings[name].items():
            concepts = (
                ranked
                if name in OKF_METHODS
                else project_pages_to_concepts(
                    [chunk_pages[item] for item in ranked if item in chunk_pages],
                    page_concepts,
                )
            )
            concept_rankings[name][question_id] = list(dict.fromkeys(concepts))
    return concept_relevance, concept_rankings


def _method_scoring_inputs(
    *,
    method_name: str,
    relevance: list[dict],
    rankings: dict[str, list[str]],
    chunk_pages: dict[str, str],
    page_concepts: dict[str, list[str]],
) -> tuple[list[dict], dict[str, list[str]]]:
    if method_name not in OKF_METHODS:
        return relevance, rankings

    concept_relevance, concept_rankings = _scoring_inputs(
        relevance=relevance,
        rankings={method_name: rankings},
        method_names=[method_name],
        chunk_pages=chunk_pages,
        page_concepts=page_concepts,
    )
    return concept_relevance, concept_rankings[method_name]


def _score_method_rankings(
    *,
    relevance: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
    chunk_pages: dict[str, str],
    page_concepts: dict[str, list[str]],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    score_names: list[str] = []
    scores: dict[str, dict[str, float]] = {}
    for method_name in method_names:
        method_relevance, method_rankings = _method_scoring_inputs(
            method_name=method_name,
            relevance=relevance,
            rankings=rankings[method_name],
            chunk_pages=chunk_pages,
            page_concepts=page_concepts,
        )
        names, method_scores = score_eval_rankings(
            method_relevance,
            {method_name: method_rankings},
            [method_name],
        )
        score_names.extend(name for name in names if name not in score_names)
        scores[method_name] = method_scores[method_name]
    return score_names, scores


def _score_method_categories(
    *,
    questions: list[dict],
    relevance: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
    chunk_pages: dict[str, str],
    page_concepts: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    question_type_by_id = {row["id"]: row["question_type"] for row in questions}
    question_types = sorted(set(question_type_by_id.values()))
    cutoff = int(EVAL_CONFIG["category_hit_k"])
    metric_names: dict[str, str] = {}
    category_scores: dict[str, dict[str, float]] = {}
    for method_name in method_names:
        method_relevance, method_rankings = _method_scoring_inputs(
            method_name=method_name,
            relevance=relevance,
            rankings=rankings[method_name],
            chunk_pages=chunk_pages,
            page_concepts=page_concepts,
        )
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
            )[f"hit@{cutoff}"]
    return metric_names, category_scores


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
    chunk_pages: dict[str, str],
    page_concepts: dict[str, list[str]],
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
        chunk_pages=chunk_pages,
        page_concepts=page_concepts,
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
    chunk_pages: dict[str, str],
    page_concepts: dict[str, list[str]],
    equivalence_audit=None,
) -> str:
    total_timed = len(timed_questions)
    progress_by_method = {
        name: int(state["methods"][name]["next_index"]) for name in method_names
    }
    aligned = min(progress_by_method.values()) if progress_by_method else 0

    header = [
        f"Methods: {', '.join(method_names)}",
        f"Scoring unit: {_scoring_unit_label(method_names)}",
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
    if _has_okf(method_names):
        concept_count = len(
            {concept for concepts in page_concepts.values() for concept in concepts}
        )
        header.extend(
            [
                "",
                f"OKF coverage: {len(page_concepts)} pages in {concept_count} concepts",
                "OKF failures: "
                + ", ".join(
                    f"{name}={state['methods'][name].get('failures', 0)}"
                    for name in method_names
                    if name in OKF_METHODS
                ),
            ]
        )
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
    score_names, scores = _score_method_rankings(
        relevance=partial_relevance,
        rankings=partial_rankings,
        method_names=method_names,
        chunk_pages=chunk_pages,
        page_concepts=page_concepts,
    )
    category_metric_names, category_scores = _score_method_categories(
        questions=partial_questions,
        relevance=partial_relevance,
        rankings=partial_rankings,
        method_names=method_names,
        chunk_pages=chunk_pages,
        page_concepts=page_concepts,
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
        category_metric_names=category_metric_names,
        category_scores=category_scores,
    )
    if equivalence_audit is not None:
        summaries = equivalence_audit.summaries(
            method_names,
            [str(row["id"]) for row in partial_questions],
        )
        partial_run = _add_method_judge_scores(
            partial_run,
            summaries,
            equivalence_audit.cutoff,
        )

    body = format_eval_report(partial_run, include_categories=include_categories)
    diagnostics = _agentic_diagnostics(partial_run, state, method_names)
    formatted_diagnostics = format_agentic_diagnostics(diagnostics)
    if formatted_diagnostics:
        body = f"{body}\n\n{formatted_diagnostics}"
    header.extend(["", body])
    return "\n".join(header)


if __name__ == "__main__":
    run_cli(main)
