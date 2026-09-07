from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db as initialize_eval_db,
)
from src.eval.metrics import (
    bold_best_table,
    plain_table,
    score_rankings,
    score_retrieval_efficiency,
    score_span_rankings,
    score_store_share,
    span_coverage,
)
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    ensure_retrievers_ready,
    list_retrievers,
)
from src.shared.env import ROOT, load_local_env, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
AGENTIC_METHODS = ["qwen_agentic", "qwen_hybrid_agentic"]
REPORTS_DIR = ROOT / ".local" / "reports"


@dataclass(frozen=True)
class EvalRun:
    questions: list[dict]
    relevance: list[dict]
    methods: list[str]
    warmup_count: int
    rankings: dict[str, dict[str, list[str]]]
    timings: dict[str, dict[str, float]]
    score_names: list[str]
    scores: dict[str, dict[str, float]]
    category_metric_names: dict[str, str] | None = None
    category_scores: dict[str, dict[str, float]] | None = None
    # Span metrics are averaged over anchored questions only, which is a smaller
    # set than the labelled ones. Reported so a table cannot quietly compare two
    # differently sized samples.
    span_questions: int = 0
    # Per-method agent action log, empty for methods that keep none. Exposed here
    # rather than harvested by each caller because `run_eval` owns the retriever
    # instances and is the only place the log can be read after the timed loop.
    action_logs: dict[str, list[dict]] = field(default_factory=dict)
    # Size of the variant's index, carried so a report can show what the
    # `store_share@k` percentages are a share *of*. A bare percentage hides that
    # tok256 and tok1024 hold the same corpus in stores of very different shape.
    store_chars: int = 0
    store_chunks: int = 0


def evaluate(
    method_names: list[str],
    show_ranks: bool = False,
    limit: int | None = None,
    category: str | None = None,
) -> None:
    run = run_eval(method_names, limit=limit, category=category)
    if not run.questions:
        print("No eval questions found. Run src.eval.generate_dataset first.")
        return

    report = _build_eval_report(
        run,
        show_ranks=show_ranks,
        category=category,
        limit=limit,
    )
    print(report)
    latest_path, snapshot_path = _write_eval_report(report)
    print()
    print(f"Saved report: {latest_path}")
    print(f"Saved snapshot: {snapshot_path}")


def format_eval_report(
    run: EvalRun,
    include_categories: bool = True,
) -> str:
    sections = [
        f"Evaluating {len(run.questions)} questions",
        _overall_table(run),
        "Speed",
        _timing_table(run),
    ]
    if include_categories:
        category_title = f"hit@{CONFIG['category_hit_k']} by category"
        if run.category_metric_names:
            metric_names = list(dict.fromkeys(run.category_metric_names.values()))
            category_title = f"{' / '.join(metric_names)} by category"
        sections.extend(
            [
                category_title,
                _category_table(run),
            ]
        )
    return "\n\n".join(sections)


def _build_eval_report(
    run: EvalRun,
    show_ranks: bool,
    category: str | None,
    limit: int | None,
) -> str:
    lines = [
        f"Methods: {', '.join(run.methods)}",
        f"Questions: {len(run.questions)}",
        f"Limit: {limit if limit is not None else 'all'}",
        f"Category: {category or 'all'}",
        "",
        format_eval_report(run, include_categories=category is None),
    ]
    if show_ranks:
        lines.extend(["", _rank_table(run)])
    return "\n".join(lines)


def _write_eval_report(report: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    latest_path = REPORTS_DIR / "eval_latest.txt"
    snapshot_path = REPORTS_DIR / f"eval_{timestamp}.txt"
    content = report + "\n"
    latest_path.write_text(content, encoding="utf-8")
    snapshot_path.write_text(content, encoding="utf-8")
    return latest_path, snapshot_path


def run_eval(
    method_names: list[str],
    limit: int | None = None,
    category: str | None = None,
    warmup: int = 0,
    variant: str = BASE_CHUNK_VARIANT,
    span_metrics: bool = False,
    span_target: str = "gold",
    store_metrics: bool = False,
) -> EvalRun:
    initialize_eval_db()
    questions, relevance = load_eval_rows(
        limit=limit, category=category, variant=variant
    )
    if not questions:
        return EvalRun([], [], [], 0, {}, {}, [], {})

    resolved_methods = _resolve_methods(method_names)
    ensure_retrievers_ready(resolved_methods, variant)
    retrievers = {name: build_retriever(name, variant) for name in resolved_methods}
    warmup = min(warmup, max(len(questions) - 1, 0))
    warmup_questions = questions[:warmup]
    timed_questions = questions[warmup:]
    if warmup_questions:
        for name in resolved_methods:
            _retrieve_rankings(retrievers[name], warmup_questions)

    method_rankings = {}
    timings = {}
    for name in resolved_methods:
        started = time.perf_counter()
        method_rankings[name], effort = _retrieve_rankings(
            retrievers[name], timed_questions
        )
        elapsed = time.perf_counter() - started
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / len(timed_questions),
            "queries_per_query": effort["queries_per_query"],
            "total_queries": effort["total_queries"],
        }

    timed_question_ids = {row["id"] for row in timed_questions}
    timed_relevance = [
        row for row in relevance if row["question_id"] in timed_question_ids
    ]
    score_names, scores = score_eval_rankings(
        timed_relevance,
        method_rankings,
        resolved_methods,
    )
    run = EvalRun(
        questions=timed_questions,
        relevance=timed_relevance,
        methods=resolved_methods,
        warmup_count=warmup,
        rankings=method_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
        action_logs={
            name: log
            for name in resolved_methods
            if (log := retriever_action_log(retrievers[name])) is not None
        },
    )
    if span_metrics:
        run = add_span_scores(run, variant, span_target)
    # After the span pass, never before: `recall_per_share@k` divides
    # `char_recall@k` by the share, so the numerator has to exist first.
    return add_store_share_scores(run, variant) if store_metrics else run


def retriever_action_log(retriever: Retriever) -> list[dict] | None:
    """The agent's action log, or None for a retriever that keeps none.

    Duck-typed on purpose: every agent that keeps an `AgenticBatchStats` reports
    here, including agents that are not `AgenticRetriever` subclasses.
    """
    batch_stats = getattr(retriever, "batch_stats", None)
    if batch_stats is not None and isinstance(
        getattr(batch_stats, "action_log", None), list
    ):
        return [dict(entry) for entry in batch_stats.action_log]
    action_log = getattr(retriever, "action_log", None)
    return [dict(entry) for entry in action_log] if isinstance(action_log, list) else None


def add_span_scores(run: EvalRun, variant: str, target: str = "gold") -> EvalRun:
    """Add character-overlap metrics to a completed run.

    Kept apart from `score_eval_rankings` because it needs the target regions and
    this variant's chunk spans, neither of which a plain method comparison uses.
    A run with no targets comes back unchanged rather than gaining a column of
    real-looking zeroes.
    """
    targets, chunk_spans = load_span_labels(variant, target)
    question_ids = {row["id"] for row in run.questions}
    targets = {
        question_id: span
        for question_id, span in targets.items()
        if question_id in question_ids
    }
    if not targets or not chunk_spans:
        return run

    ks = tuple(CONFIG["metric_ks"])
    budgets = tuple(CONFIG.get("char_budgets") or ())
    scores = {name: dict(values) for name, values in run.scores.items()}
    span_names: list[str] = []
    for method_name in run.methods:
        span_scores = score_span_rankings(
            targets,
            chunk_spans,
            run.rankings[method_name],
            ks=ks,
            budgets=budgets,
        )
        scores[method_name].update(span_scores)
        span_names = list(span_scores)

    return replace(
        run,
        score_names=[*run.score_names, *span_names],
        scores=scores,
        span_questions=max(
            (span_coverage(targets, run.rankings[name]) for name in run.methods),
            default=0,
        ),
    )


def add_store_share_scores(run: EvalRun, variant: str) -> EvalRun:
    """Add the cost side of the trade: what share of the index each query read.

    Kept out of `add_span_scores` because it needs no targets at all — only the
    variant's chunk spans and the size of its store — so it covers every question
    that was run rather than only the anchored ones. It is also the one pass whose
    numbers a reader is supposed to want *low*, which is why the tables that show
    it have to be told so.

    `recall_per_share@k` is added only where `char_recall@k` is already present.
    A run with no span metrics keeps the share column and loses the ratio, rather
    than gaining a ratio computed against a missing numerator.
    """
    chunk_spans, store_chars = load_store_profile(variant)
    if not chunk_spans or store_chars <= 0:
        return run

    ks = tuple(CONFIG["metric_ks"])
    scores = {name: dict(values) for name, values in run.scores.items()}
    store_names: list[str] = []
    for method_name in run.methods:
        share = score_store_share(
            chunk_spans, run.rankings[method_name], store_chars, ks=ks
        )
        scores[method_name].update(share)
        efficiency = score_retrieval_efficiency(scores[method_name], ks=ks)
        scores[method_name].update(efficiency)
        store_names = [*share, *efficiency]

    return replace(
        run,
        score_names=[*run.score_names, *store_names],
        scores=scores,
        store_chars=store_chars,
        store_chunks=len(chunk_spans),
    )


def load_store_profile(variant: str) -> tuple[dict[str, tuple], int]:
    """This variant's chunk spans, and the total characters its store holds.

    `char_count` is summed rather than the spans measured, because those are two
    different numbers for an overlapping cut and only one of them describes the
    index: the store holds every copy, so the copies count. Chunks with no
    resolved span are still charged to the denominator — they are in the
    collection and were embedded — but cannot be charged to any numerator, which
    makes an unspanned variant read as cheap rather than as an error. Every
    variant a sweep runs is span-backfilled first.
    """
    with connect() as conn:
        chunk_spans = {
            row["id"]: (row["page_id"], row["start_char"], row["end_char"])
            for row in conn.execute(
                """
                select id, page_id, start_char, end_char
                from page_chunks
                where variant = ?
                  and start_char is not null
                  and end_char is not null
                """,
                (variant,),
            )
        }
        store_chars = int(
            conn.execute(
                "select coalesce(sum(char_count), 0) from page_chunks where variant = ?",
                (variant,),
            ).fetchone()[0]
        )
    return chunk_spans, store_chars


def load_span_labels(
    variant: str, target: str = "gold"
) -> tuple[dict[str, tuple], dict[str, tuple]]:
    """Target regions per question, and this variant's chunk spans.

    Two kinds of target, and the default needs no model at all:

    - `gold` — the span of the base chunk the question was generated from. That
      chunk is ground truth by construction rather than by anyone's judgement, and
      every approved question has one, so this covers the whole eval set for free.
    - `anchor` — the span of the quoted sentence that states the answer. A tighter
      target, so more discriminating, but it costs a model pass and only lands for
      about six questions in ten.

    Do not mix the two in one table: a paragraph-sized target and a
    sentence-sized one put recall and precision on different scales.

    Chunks with no resolved span are left out. Counting them as zero-length would
    understate how much text a method actually retrieved.
    """
    if target not in {"gold", "anchor"}:
        raise ValueError(f"Unknown span target: {target}")
    with connect() as conn:
        if target == "anchor":
            table = conn.execute(
                "select 1 from sqlite_master where type = 'table' "
                "and name = 'eval_answer_anchors'"
            ).fetchone()
            if not table:
                return {}, {}
            targets = {
                row["question_id"]: (row["page_id"], row["start_char"], row["end_char"])
                for row in conn.execute(
                    "select question_id, page_id, start_char, end_char "
                    "from eval_answer_anchors"
                )
            }
        else:
            targets = {
                row["question_id"]: (row["page_id"], row["start_char"], row["end_char"])
                for row in conn.execute(
                    """
                    select r.question_id,
                           g.page_id,
                           min(g.start_char) as start_char,
                           max(g.end_char) as end_char
                    from eval_relevant_chunks r
                    join page_chunks g on g.id = r.chunk_id
                    where g.variant = ?
                      and g.start_char is not null
                      and g.end_char is not null
                    group by r.question_id, g.page_id
                    order by r.question_id, g.page_id
                    """,
                    (BASE_CHUNK_VARIANT,),
                )
            }
        chunk_spans = {
            row["id"]: (row["page_id"], row["start_char"], row["end_char"])
            for row in conn.execute(
                """
                select id, page_id, start_char, end_char
                from page_chunks
                where variant = ?
                  and start_char is not null
                  and end_char is not null
                """,
                (variant,),
            )
        }
    return targets, chunk_spans


def load_eval_rows(
    limit: int | None = None,
    category: str | None = None,
    variant: str = BASE_CHUNK_VARIANT,
) -> tuple[list[dict], list[dict]]:
    """Approved questions and their gold labels for one chunk variant.

    Question texts are shared across variants; only the labels are per-variant.
    A question is included only if it has a label in this variant, because
    `score_rankings` averages over the questions it has labels for: a question
    left unlabelled would otherwise be scored as a guaranteed miss.
    """
    with connect() as conn:
        question_sql = """
                select q.id, q.question, q.answer, q.question_type,
                       q.question_language
                from eval_questions q
                where q.approved = 1
                  and exists (
                    select 1
                    from eval_relevant_chunks r
                    join page_chunks c on c.id = r.chunk_id
                    where r.question_id = q.id
                      and c.variant = :variant
                  )
                """
        params = {"variant": variant}
        if category is not None:
            question_sql += "\n                and q.question_type = :category"
            params["category"] = category
        question_sql += "\n                order by q.id"
        if limit is not None:
            question_sql += "\n                limit :limit"
            params["limit"] = limit
        questions = [
            dict(row)
            for row in conn.execute(
                question_sql,
                params,
            )
        ]
        question_ids = [row["id"] for row in questions]
        if not question_ids:
            return questions, []
        placeholders = ", ".join("?" for _ in question_ids)
        relevance = [
            dict(row)
            for row in conn.execute(
                f"""
                select r.question_id, r.chunk_id
                from eval_relevant_chunks r
                join page_chunks c on c.id = r.chunk_id
                where r.question_id in ({placeholders})
                  and c.variant = ?
                order by r.question_id, r.chunk_id
                """,
                [*question_ids, variant],
            )
        ]
    return questions, relevance


def _retrieve_rankings(
    retriever: Retriever,
    questions: list[dict],
) -> tuple[dict[str, list[str]], dict[str, float]]:
    batches = retriever.retrieve_batch(
        [row["question"] for row in questions],
        CONFIG["result_limit"],
    )
    rankings = {
        row["id"]: [chunk.id for chunk in batches[index]]
        for index, row in enumerate(questions)
    }
    return rankings, _query_effort(retriever, question_count=len(questions))


def _overall_table(run: EvalRun) -> str:
    rows = [
        [
            name,
            *(run.scores[name].get(score_name, "-") for score_name in run.score_names),
        ]
        for name in run.methods
    ]
    return "\n".join(
        [
            "Overall",
            bold_best_table(
                ["method", *run.score_names],
                rows,
                lower_is_better=cost_metrics(run.score_names),
            ),
        ]
    )


def cost_metrics(score_names: list[str]) -> list[str]:
    """Which of these columns a reader wants small.

    Public because every table that renders `store_share@k` needs the same
    answer, and bolding the largest share would put the bold on the most
    expensive cut in the grid.
    """
    return [name for name in score_names if name.startswith("store_share@")]


def score_eval_rankings(
    relevance: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    metric_ks = tuple(CONFIG["metric_ks"])
    standard_k = max(metric_ks)
    score_names = [f"hit@{k}" for k in metric_ks]
    score_names.extend([f"recall@{standard_k}", f"mrr@{CONFIG['mrr_k']}"])
    scores = {
        name: score_rankings(
            relevance,
            rankings[name],
            ks=metric_ks,
            mrr_k=CONFIG["mrr_k"],
            recall_k=standard_k,
        )
        for name in method_names
    }

    expanded_k = max(
        (
            len(ranked)
            for name in method_names
            if "agentic" in name
            for ranked in rankings[name].values()
        ),
        default=standard_k,
    )
    if expanded_k <= standard_k:
        return score_names, scores

    expanded_names = [
        f"hit@{expanded_k}",
        f"recall@{expanded_k}",
        f"mrr@{expanded_k}",
    ]
    score_names.extend(expanded_names)
    for name in method_names:
        if "agentic" not in name:
            continue
        expanded_scores = score_rankings(
            relevance,
            rankings[name],
            ks=(expanded_k,),
            mrr_k=expanded_k,
            recall_k=expanded_k,
        )
        scores[name].update(expanded_scores)
    return score_names, scores


def add_judge_adjusted_scores(
    run: EvalRun,
    summaries: dict[str, dict[str, int]],
    cutoff: int,
) -> EvalRun:
    score_name = f"judge_hit@{cutoff}"
    scores = {name: dict(values) for name, values in run.scores.items()}
    for method_name in run.methods:
        summary = summaries[method_name]
        assessed = summary["questions"]
        scores[method_name][score_name] = (
            (summary["strict_hits"] + summary["equivalent_misses"]) / assessed
            if assessed
            else 0.0
        )
    # `replace` rather than a fresh EvalRun: rebuilding field by field silently
    # drops any field added later, which is how the category and span columns
    # would go missing from a judged run.
    return replace(run, score_names=[*run.score_names, score_name], scores=scores)


def count_chunk_expansions(action_log: list[dict]) -> int:
    expansions = 0
    previous_count: int | None = None
    for entry in action_log:
        attempt = int(entry.get("attempt", 1))
        chunk_count = len(entry.get("chunks", []))
        if attempt <= 1:
            previous_count = chunk_count
            continue
        if previous_count is not None and chunk_count > previous_count:
            expansions += 1
        previous_count = chunk_count
    return expansions


def _timing_table(run: EvalRun) -> str:
    show_chunk_expansions = any(
        "chunk_expansions" in run.timings[name] for name in run.methods
    )
    headers = ["method", "seconds", "ms/query", "queries/query", "queries"]
    if show_chunk_expansions:
        headers.append("chunk expansions")

    rows = []
    for name in run.methods:
        row = [
            name,
            f"{run.timings[name]['seconds']:.2f}",
            f"{run.timings[name]['ms_per_query']:.1f}",
            f"{run.timings[name]['queries_per_query']:.2f}",
            int(run.timings[name]["total_queries"]),
        ]
        if show_chunk_expansions:
            row.append(int(run.timings[name].get("chunk_expansions", 0)))
        rows.append(row)

    return plain_table(
        headers,
        rows,
    )


def _query_effort(retriever: Retriever, question_count: int) -> dict[str, float]:
    if question_count == 0:
        return {"queries_per_query": 0.0, "total_queries": 0.0}

    if hasattr(retriever, "average_queries_per_question") and hasattr(
        retriever, "total_queries"
    ):
        return {
            "queries_per_query": float(retriever.average_queries_per_question()),
            "total_queries": float(retriever.total_queries()),
        }

    return {
        "queries_per_query": 1.0,
        "total_queries": float(question_count),
    }


def _category_table(run: EvalRun) -> str:
    question_type_by_id = {row["id"]: row["question_type"] for row in run.questions}
    types = sorted(set(question_type_by_id.values()))
    if run.category_metric_names is not None and run.category_scores is not None:
        metric_names = set(run.category_metric_names.values())
        if len(metric_names) == 1:
            rows = [
                [
                    method_name,
                    *(
                        run.category_scores[method_name][question_type]
                        for question_type in types
                    ),
                ]
                for method_name in run.methods
            ]
            return bold_best_table(["method", *types], rows)
        rows = [
            [
                method_name,
                run.category_metric_names[method_name],
                *(
                    run.category_scores[method_name][question_type]
                    for question_type in types
                ),
            ]
            for method_name in run.methods
        ]
        return bold_best_table(["method", "metric", *types], rows)

    by_method = category_hit_scores(run)
    rows = [
        [method_name, *(by_method[method_name][question_type] for question_type in types)]
        for method_name in run.methods
    ]
    return bold_best_table(["method", *types], rows)


def category_question_types(run: EvalRun) -> list[str]:
    """Question types present in this run, in table order."""
    return sorted({row["question_type"] for row in run.questions})


def category_hit_scores(run: EvalRun) -> dict[str, dict[str, float]]:
    """`hit@category_hit_k` per method per question type.

    Public because a chunk-variant grid needs the same numbers without the
    surrounding single-run report: `format_eval_report` renders one run, while a
    variant sweep needs one cell's category scores kept alongside four others.
    """
    question_type_by_id = {row["id"]: row["question_type"] for row in run.questions}
    relevance_by_type = defaultdict(list)
    for row in run.relevance:
        relevance_by_type[question_type_by_id[row["question_id"]]].append(row)

    metric = f"hit@{CONFIG['category_hit_k']}"
    scores: dict[str, dict[str, float]] = {}
    for method_name in run.methods:
        scores[method_name] = {}
        for question_type in category_question_types(run):
            rows = relevance_by_type[question_type]
            type_question_ids = {item["question_id"] for item in rows}
            rankings = {
                question_id: run.rankings[method_name][question_id]
                for question_id in type_question_ids
            }
            scores[method_name][question_type] = score_rankings(
                rows,
                rankings,
                ks=(CONFIG["category_hit_k"],),
                mrr_k=CONFIG["mrr_k"],
            )[metric]
    return scores


def _rank_table(run: EvalRun) -> str:
    relevant_by_question = defaultdict(set)
    for row in run.relevance:
        relevant_by_question[row["question_id"]].add(row["chunk_id"])

    rows = []
    for question in run.questions:
        row = [question["question_type"], question["question"][:80]]
        for method_name in run.methods:
            rank = _first_rank(
                run.rankings[method_name][question["id"]],
                relevant_by_question[question["id"]],
            )
            row.append(rank or "-")
        rows.append(row)
    return plain_table(["type", "question", *run.methods], rows)


def _first_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return index
    return None


def _resolve_methods(method_names: list[str]) -> list[str]:
    if not method_names:
        return CONFIG["default_methods"]
    if method_names == ["all"]:
        return list_retrievers()
    unknown = sorted(set(method_names) - set(list_retrievers()))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return method_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(CONFIG["default_methods"]))
    parser.add_argument(
        "--agentic-only",
        action="store_true",
        help="Run only agentic retrieval methods (qwen_agentic,qwen_hybrid_agentic).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--show-ranks", action="store_true")
    args = parser.parse_args()

    load_local_env()
    methods = (
        AGENTIC_METHODS
        if args.agentic_only
        else [name.strip() for name in args.methods.split(",") if name.strip()]
    )

    evaluate(
        methods,
        show_ranks=args.show_ranks,
        limit=args.limit,
        category=args.category,
    )


if __name__ == "__main__":
    main()
